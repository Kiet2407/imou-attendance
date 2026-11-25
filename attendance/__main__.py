import argparse
import csv
from datetime import datetime, timedelta
import html
import io
import json
import logging
import os
import re
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

from .db import Store
from .tracking import EntryTracker


def aware_datetime(value):
    result = datetime.fromisoformat(value)
    if result.utcoffset() is None:
        raise argparse.ArgumentTypeError('Timestamp must include timezone, e.g. +07:00')
    return result


def parser():
    p = argparse.ArgumentParser(description='Chấm công một cửa bằng OpenCV')
    p.add_argument('--db', default='data/attendance.sqlite3')
    sub = p.add_subparsers(dest='command', required=True)
    download = sub.add_parser('download-models')
    download.add_argument('--models', default='models')
    employee = sub.add_parser('employee', help='Thêm nhân viên chưa có ảnh')
    employee.add_argument('--id', required=True)
    employee.add_argument('--name', required=True)
    enroll = sub.add_parser('enroll', help='Đăng ký từ ảnh thật, mỗi ảnh đúng một mặt')
    enroll.add_argument('--id', required=True)
    enroll.add_argument('--name', required=True)
    enroll.add_argument('--images', nargs='+', required=True)
    enroll.add_argument('--models', default='models')
    webcam = sub.add_parser('enroll-camera', help='Đăng ký khuôn mặt trực tiếp từ webcam')
    webcam.add_argument('--id', required=True)
    webcam.add_argument('--name', required=True)
    webcam.add_argument('--source', default='0')
    webcam.add_argument('--models', default='models')
    webcam.add_argument('--samples', type=int, default=5)
    webcam.add_argument('--timeout', type=float, default=30)
    run = sub.add_parser('run')
    run.add_argument('--source', help='Mặc định đọc biến môi trường CAMERA_SOURCE')
    run.add_argument('--models', default='models')
    run.add_argument('--axis', choices=['x','y'], default='x')
    run.add_argument('--line', type=float, default=.5)
    run.add_argument('--direction', type=int, choices=[-1,1], default=1)
    run.add_argument('--threshold', type=float, default=.45)
    run.add_argument('--margin', type=float, default=.05)
    run.add_argument('--confirmations', type=int, default=3)
    run.add_argument('--camera', default='entrance')
    run.add_argument('--video-start', type=aware_datetime,
                     help='Bắt buộc với file: thời điểm bắt đầu quay, có múi giờ')
    run.add_argument('--preview', help='Ghi ảnh JPEG có đường ranh để chỉnh vị trí')
    report = sub.add_parser('report')
    report.add_argument('--date', default=datetime.now(ZoneInfo('Asia/Ho_Chi_Minh')).date().isoformat())
    report.add_argument('--timezone', default='Asia/Ho_Chi_Minh')
    report.add_argument('--format', choices=['csv','html'], default='csv')
    report.add_argument('--output', required=True)
    return p


def engine_for(args):
    from .vision import FaceEngine
    root = Path(args.models)
    return FaceEngine(str(root/'face_detection_yunet_2023mar.onnx'),
                      str(root/'face_recognition_sface_2021dec.onnx'),
                      threshold=getattr(args,'threshold',.45),
                      margin=getattr(args,'margin',.05))


def report_file(store, args):
    rows = store.daily_report(args.date, args.timezone)
    columns = ['Mã nhân viên','Họ tên','Giờ đến đầu tiên','Camera','Trạng thái']
    values = [[r['employee_id'],r['name'],r['first_arrival'] or '',
               r['camera'] or '', 'Đã ghi nhận' if r['first_arrival'] else 'Chưa ghi nhận'] for r in rows]
    if args.format == 'csv':
        output = io.StringIO(newline='')
        writer = csv.writer(output)
        writer.writerow(columns)
        # Avoid spreadsheet formula execution when opening exported text.
        writer.writerows([["'"+v if v.startswith(('=','+','-','@','\t','\r')) else v for v in row] for row in values])
        content = '\ufeff'+output.getvalue()
    else:
        cells = lambda row, tag: '<tr>'+''.join(f'<{tag}>{html.escape(v)}</{tag}>' for v in row)+'</tr>'
        content = ('<!doctype html><html lang="vi"><meta charset="utf-8">'
                   '<meta name="viewport" content="width=device-width,initial-scale=1">'
                   '<title>Bảng chấm công</title><style>body{font:16px system-ui;margin:40px}'
                   'table{border-collapse:collapse}th,td{padding:12px;border:1px solid #ccc}'
                   'th{background:#eef4fa}</style><h1>Chấm công '+html.escape(args.date)+'</h1>'
                   '<p>Múi giờ: '+html.escape(args.timezone)+'</p><table>'+cells(columns,'th')+
                   ''.join(cells(row,'td') for row in values)+'</table></html>')
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding='utf-8')
    print(f'Đã xuất {len(rows)} nhân viên: {target}')


def run_camera(store, args):
    import cv2
    from .vision import open_capture
    from .live import LatestCapture
    source = args.source or os.environ.get('CAMERA_SOURCE')
    if source is None:
        raise ValueError('Cần --source hoặc biến môi trường CAMERA_SOURCE')
    is_file = Path(source).is_file() if '://' not in source else False
    if is_file and not args.video_start:
        raise ValueError('File video cần --video-start để không ghi nhầm giờ hiện tại')
    if not is_file and args.video_start:
        raise ValueError('--video-start chỉ dùng cho file video')
    engine = engine_for(args)
    gallery = store.embeddings()
    if not gallery:
        raise ValueError('Chưa có mẫu nhân viên. Hãy chạy enroll trước.')
    def tracker():
        return EntryTracker(axis=args.axis,line=args.line,direction=args.direction,
                            confirmations=args.confirmations)
    gate = tracker()
    cap = open_capture(source) if is_file else LatestCapture(source)
    frame_index = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) if is_file else 0
        if is_file and (fps <= 0 or not __import__('math').isfinite(fps)):
            raise ValueError('Không xác định được FPS video')
        while True:
            if is_file:
                ok, frame = cap.read()
            else:
                ok, frame, occurred_at, tick = cap.read()
            if not ok:
                if is_file:
                    break
                logging.warning('Mất hình; kết nối lại sau 2 giây. Không ghi nhận trong khoảng mất hình.')
                cap.close()
                time.sleep(2)
                gate = tracker()
                try:
                    cap = LatestCapture(source)
                except (RuntimeError, ValueError):
                    continue
                continue
            if is_file:
                elapsed = frame_index/fps
                occurred_at = args.video_start + timedelta(seconds=elapsed)
                tick = elapsed
            frame_index += 1
            height, width = frame.shape[:2]
            detections = []
            for face in engine.detect(frame):
                x,y,w,h = face[:4]
                identity = engine.identify(engine.embedding(frame,face),gallery)
                detections.append((((x+w/2)/width,(y+h/2)/height),identity))
            for employee_id,name,score in gate.update(detections,tick):
                store.record_entry(employee_id,occurred_at,args.camera,float(score))
                print(json.dumps({'employee_id':employee_id,'name':name,
                                  'time':occurred_at.isoformat(),'score':float(score)},ensure_ascii=False),flush=True)
            if args.preview and frame_index % 15 == 1:
                preview = frame.copy()
                pos = int(args.line*(width if args.axis=='x' else height))
                a,b = ((pos,0),(pos,height)) if args.axis=='x' else ((0,pos),(width,pos))
                cv2.line(preview,a,b,(0,255,255),2)
                for point,identity in detections:
                    center=(int(point[0]*width),int(point[1]*height))
                    cv2.circle(preview,center,8,(0,255,0) if identity else (0,0,255),2)
                target = Path(args.preview)
                target.parent.mkdir(parents=True,exist_ok=True)
                if not cv2.imwrite(str(target),preview):
                    raise ValueError('Không ghi được ảnh preview')
    finally:
        if is_file:
            cap.release()
        else:
            cap.close()


def main():
    args = parser().parse_args()
    logging.basicConfig(level=logging.INFO,format='%(levelname)s: %(message)s')
    store = None
    try:
        if args.command == 'download-models':
            from .vision import download_models
            download_models(args.models)
            print('Đã tải model.')
            return 0
        store = Store(args.db)
        if args.command == 'employee':
            store.add_employee(args.id,args.name)
        elif args.command == 'enroll':
            from .vision import enroll_images
            enroll_images(engine_for(args),store,args.id,args.name,args.images)
            print('Đã đăng ký mẫu khuôn mặt.')
        elif args.command == 'enroll-camera':
            from .vision import enroll_camera
            enroll_camera(engine_for(args),store,args.id,args.name,args.source,args.samples,args.timeout)
            print('Đã đăng ký khuôn mặt từ camera.')
        elif args.command == 'report':
            report_file(store,args)
        elif args.command == 'run':
            run_camera(store,args)
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        # Show actionable errors while redacting source URLs with credentials.
        detail = re.sub(r'(?i)(?:rtsps?|https?)://[^\s]+', '[địa chỉ camera đã ẩn]', str(exc))
        print(f'Lỗi: {detail}',file=sys.stderr)
        return 1
    finally:
        if store:
            store.close()


if __name__ == '__main__':
    sys.exit(main())
