"""Local face recognition with OpenCV YuNet and SFace.

Models are downloaded only by an explicit download_models() call.
"""
from pathlib import Path
import os
import tempfile
import urllib.request

import cv2
import numpy as np

MODEL_REVISION = "48b10fa505aeb51fbd127fc60a5c660040e823b7"
MODEL_FILES = {
    "detector": ("face_detection_yunet", "face_detection_yunet_2023mar.onnx", 100_000, 2_000_000),
    "recognizer": ("face_recognition_sface", "face_recognition_sface_2021dec.onnx", 30_000_000, 50_000_000),
}


def _valid_model(path, minimum, maximum):
    path = Path(path)
    if not path.is_file() or not minimum <= path.stat().st_size <= maximum:
        return False
    with path.open("rb") as stream:
        return not stream.read(200).startswith(b"version https://git-lfs.github.com/spec/")


def download_models(directory):
    """Fetch pinned official OpenCV Zoo models, atomically; return paths by role."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    result = {}
    for role, (folder, filename, minimum, maximum) in MODEL_FILES.items():
        target = directory / filename
        if not _valid_model(target, minimum, maximum):
            url = f"https://media.githubusercontent.com/media/opencv/opencv_zoo/{MODEL_REVISION}/models/{folder}/{filename}"
            fd, temporary = tempfile.mkstemp(prefix=filename + ".", suffix=".tmp", dir=directory)
            try:
                with os.fdopen(fd, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
                    size = 0
                    while chunk := response.read(1024 * 1024):
                        size += len(chunk)
                        if size > maximum:
                            raise ValueError(f"Unexpected model size: {filename}")
                        output.write(chunk)
                if not _valid_model(temporary, minimum, maximum):
                    raise ValueError(f"Invalid model download: {filename}")
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        result[role] = str(target)
    return result


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    if not vector.size or not np.isfinite(vector).all():
        raise ValueError("Face embedding must be a finite, nonempty vector")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8 or not np.isfinite(norm):
        raise ValueError("Face embedding has invalid norm")
    return vector / norm


class FaceEngine:
    def __init__(self, detector_path, recognizer_path, threshold=.45, margin=.05, min_face_size=70):
        if not -1 <= threshold <= 1 or not 0 <= margin <= 2 or min_face_size <= 0:
            raise ValueError("Invalid recognition configuration")
        for path in (detector_path, recognizer_path):
            if not Path(path).is_file():
                raise FileNotFoundError(f"Missing model: {path}; run download-models first")
        self.threshold = threshold
        self.margin = margin
        self.min_face_size = min_face_size
        self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), .9, .3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")

    def detect(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Empty image")
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(frame)
        if faces is None:
            return []
        return [face for face in faces if face[2] >= self.min_face_size and face[3] >= self.min_face_size]

    def embedding(self, frame, face):
        aligned = self.recognizer.alignCrop(frame, face)
        return _unit(self.recognizer.feature(aligned))

    def identify(self, vector, gallery):
        """Return a unique accepted identity; similarity is not a probability."""
        vector = _unit(vector)
        scores = {}
        for employee_id, name, candidate in gallery:
            candidate = _unit(candidate)
            if candidate.shape != vector.shape:
                raise ValueError("Incompatible gallery embedding dimensions")
            score = float(np.clip(np.dot(vector, candidate), -1., 1.))
            if employee_id not in scores or score > scores[employee_id][1]:
                scores[employee_id] = (name, score)
        ranked = sorted(scores.items(), key=lambda item: item[1][1], reverse=True)
        if not ranked:
            return None
        employee_id, (name, score) = ranked[0]
        if score < self.threshold:
            return None
        if len(ranked) > 1 and score - ranked[1][1][1] <= self.margin:
            return None
        return employee_id, name, score


def open_capture(source):
    """Open webcam, file or network source. Does not log source credentials."""
    if isinstance(source, str) and source.isdecimal():
        source = int(source)
    if isinstance(source, str) and source.lower().startswith(("rtsp://", "rtsps://", "http://", "https://")):
        try:
            capture = cv2.VideoCapture(source, cv2.CAP_FFMPEG, [
                cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
            ])
        except (cv2.error, AttributeError) as exc:
            raise RuntimeError("OpenCV build lacks network video timeout support") from exc
    else:
        capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        if isinstance(source, int):
            raise RuntimeError(f"Không mở được webcam {source}. Kiểm tra camera có được hệ điều hành nhận, quyền truy cập và ứng dụng khác đang sử dụng. Có thể thử --source 1.")
        raise RuntimeError("Cannot open camera/video; check source and connection")
    return capture


def enroll_images(engine, store, employee_id, name, paths):
    """Validate every input image before storing an employee or face samples."""
    paths = list(paths)
    if not paths:
        raise ValueError("Provide at least one registration image")
    vectors = []
    for path in paths:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"Cannot read registration image: {path}")
        faces = engine.detect(frame)
        if len(faces) != 1:
            raise ValueError(f"Registration image must contain exactly one usable face: {path}")
        vectors.append(engine.embedding(frame, faces[0]).tolist())
    store.add_employee(employee_id, name)
    for vector in vectors:
        store.save_embedding(employee_id, vector)
    return len(vectors)


def enroll_camera(engine, store, employee_id, name, source='0', samples=5,
                  timeout=30.0, interval=.8, warmup=2.0):
    """Collect consistent single-face samples, committing only on completion."""
    import time
    if not 1 <= samples <= 30 or timeout <= warmup or interval <= 0:
        raise ValueError('Invalid camera enrollment settings')
    capture = open_capture(source)
    vectors = []
    started = time.monotonic()
    next_sample = started + warmup
    print('Nhìn thẳng webcam, chỉ để một người trong hình. Đang lấy mẫu...', flush=True)
    try:
        while time.monotonic() - started < timeout:
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError('Không đọc được hình webcam. Kiểm tra quyền camera hoặc ứng dụng đang dùng camera.')
            now = time.monotonic()
            if now < next_sample:
                continue
            faces = engine.detect(frame)
            if len(faces) != 1:
                continue
            vector = engine.embedding(frame, faces[0])
            if vectors and engine.identify(vector, [(employee_id, name, vectors[0])]) is None:
                continue
            vectors.append(vector.tolist())
            next_sample = now + interval
            print(f'Đã lấy {len(vectors)}/{samples} mẫu', flush=True)
            if len(vectors) == samples:
                break
        if len(vectors) < samples:
            raise ValueError('Chưa lấy đủ mẫu mặt trong thời gian chờ. Nhìn rõ vào camera, tăng ánh sáng và thử lại. Chưa lưu mẫu mới.')
    finally:
        capture.release()
    store.add_employee(employee_id, name)
    for vector in vectors:
        store.save_embedding(employee_id, vector)
    return len(vectors)
