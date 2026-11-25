"""Read live video continuously, retaining only the newest decoded frame."""
from datetime import datetime, timezone
import threading
import time

from .vision import open_capture


class LatestCapture:
    """Single-slot capture with acquisition timestamps and bounded consumer waits.

    Intended for live cameras, not prerecorded files. The producer drains the
    decoder while recognition runs; old unconsumed frames are deliberately lost.
    """

    def __init__(self, source):
        self._capture = open_capture(source)
        self._condition = threading.Condition()
        self._stopped = False
        self._failed = False
        self._sequence = 0
        self._consumed = 0
        self._latest = None
        self._thread = threading.Thread(target=self._reader, name="camera-reader", daemon=True)
        self._thread.start()

    def _reader(self):
        try:
            while True:
                with self._condition:
                    if self._stopped:
                        break
                ok, frame = self._capture.read()
                acquired_at = datetime.now(timezone.utc)
                tick = time.monotonic()
                with self._condition:
                    if self._stopped:
                        break
                    if not ok or frame is None:
                        self._failed = True
                        self._condition.notify_all()
                        break
                    self._sequence += 1
                    self._latest = (frame, acquired_at, tick)
                    self._condition.notify_all()
        except Exception:
            with self._condition:
                self._failed = True
                self._condition.notify_all()
        finally:
            self._capture.release()

    def read(self, timeout=12.0):
        """Return (ok, frame, aware UTC datetime, monotonic tick), once per frame.

        This class supports one consumer. Failure, closure or a wait timeout
        returns (False, None, None, None). A frame is never returned twice.
        """
        timeout = float(timeout)
        if not 0 <= timeout <= 12:
            raise ValueError("Capture read timeout must be between 0 and 12 seconds")
        with self._condition:
            self._condition.wait_for(
                lambda: self._sequence > self._consumed or self._failed or self._stopped,
                timeout=timeout,
            )
            if self._stopped or self._failed or self._sequence == self._consumed:
                return False, None, None, None
            self._consumed = self._sequence
            frame, acquired_at, tick = self._latest
            return True, frame, acquired_at, tick

    def close(self):
        """Signal shutdown and wait at most twelve seconds for the reader.

        The producer owns release() to avoid concurrent FFmpeg read/release.
        Network streams have a ten-second read timeout in open_capture().
        """
        with self._condition:
            self._stopped = True
            self._condition.notify_all()
        if threading.current_thread() is not self._thread:
            self._thread.join(timeout=12)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
