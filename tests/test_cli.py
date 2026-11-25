import argparse
import contextlib
import io
from datetime import datetime
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from attendance.__main__ import main, parser, report_file, run_camera
from attendance.db import Store


class CliTests(unittest.TestCase):
    def test_report_html_escapes_employee_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(':memory:')
            self.addCleanup(store.close)
            store.add_employee('a','<script>alert(1)</script>')
            args=parser().parse_args(['report','--date','2026-11-25','--format','html','--output',tmp+'/report.html'])
            report_file(store,args)
            text=Path(args.output).read_text()
            self.assertIn('&lt;script&gt;',text)
            self.assertNotIn('<script>',text)
            self.assertIn('Chưa ghi nhận',text)

    def test_csv_formula_is_escaped(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=Store(':memory:')
            self.addCleanup(store.close)
            store.add_employee('a','=1+1')
            args=parser().parse_args(['report','--output',tmp+'/out.csv'])
            report_file(store,args)
            self.assertIn("'=1+1",Path(args.output).read_text())

    def test_real_video_capture_to_database_with_fake_recognition(self):
        # Tests real OpenCV decode + tracking + UTC DB + local report;
        # deliberately does not assert facial-recognition accuracy.
        with tempfile.TemporaryDirectory() as tmp:
            path=tmp+'/door.avi'
            writer=cv2.VideoWriter(path,cv2.VideoWriter_fourcc(*'MJPG'),10,(200,200))
            self.assertTrue(writer.isOpened())
            for _ in range(8):
                writer.write(np.zeros((200,200,3),dtype=np.uint8))
            writer.release()
            class FakeEngine:
                index=0
                def detect(self,frame):
                    center=(.25+self.index*.05)*200
                    self.index+=1
                    return [np.array([center-40,60,80,80])]
                def embedding(self,frame,face):
                    return [1,0]
                def identify(self,vector,gallery):
                    return ('a','An',.8)
            store=Store(':memory:')
            self.addCleanup(store.close)
            store.add_employee('a','An')
            store.save_embedding('a',[1,0])
            args=parser().parse_args(['run','--source',path,'--video-start','2026-11-25T08:00:00+07:00'])
            with patch('attendance.__main__.engine_for',return_value=FakeEngine()):
                run_camera(store,args)
            report=store.daily_report('2026-11-25')
            self.assertEqual(report[0]['first_arrival'],'2026-11-25T08:00:00.600000+07:00')

    def test_file_requires_original_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'file.avi'
            path.touch()
            args=parser().parse_args(['run','--source',str(path)])
            store=Store(':memory:')
            self.addCleanup(store.close)
            with self.assertRaisesRegex(ValueError,'video-start'):
                run_camera(store,args)

    def test_missing_enrollment_message_visible(self):
        with tempfile.TemporaryDirectory() as tmp:
            errors=io.StringIO()
            with patch('sys.argv',['attendance','--db',tmp+'/empty.sqlite3','run','--source','0']), patch('attendance.__main__.engine_for'), contextlib.redirect_stderr(errors):
                self.assertEqual(main(),1)
            self.assertIn('Chưa có mẫu nhân viên',errors.getvalue())

    def test_error_camera_url_is_redacted(self):
        errors=io.StringIO()
        with patch('sys.argv',['attendance','--db',':memory:','run','--source','0']), patch('attendance.__main__.run_camera',side_effect=RuntimeError('Cannot open rtsp://user:secret@camera/stream')), contextlib.redirect_stderr(errors):
            self.assertEqual(main(),1)
        self.assertNotIn('secret',errors.getvalue())
        self.assertIn('địa chỉ camera đã ẩn',errors.getvalue())
