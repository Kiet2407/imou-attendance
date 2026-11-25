import unittest
from unittest.mock import Mock, patch
import numpy as np
from attendance.db import Store
from attendance.vision import enroll_camera


class CameraEnrollTests(unittest.TestCase):
    def setUp(self):
        self.store=Store(':memory:')
        self.addCleanup(self.store.close)
        self.cap=Mock()
        self.cap.read.return_value=(True,np.zeros((100,100,3),dtype=np.uint8))
        self.engine=Mock()
        self.engine.detect.return_value=[np.zeros(15)]
        self.engine.embedding.return_value=np.array([1.,0.])
        self.engine.identify.return_value=('a','An',1.)
        self.tick=0

    def clock(self):
        self.tick+=.5
        return self.tick

    def test_capture_registers_samples_and_releases(self):
        with patch('attendance.vision.open_capture',return_value=self.cap), patch('time.monotonic',side_effect=self.clock):
            result=enroll_camera(self.engine,self.store,'a','An',samples=3,timeout=20,warmup=0)
        self.assertEqual(result,3)
        self.assertEqual(len(self.store.embeddings()),3)
        self.cap.release.assert_called_once()

    def test_timeout_leaves_no_partial_registration(self):
        self.engine.detect.return_value=[]
        with patch('attendance.vision.open_capture',return_value=self.cap), patch('time.monotonic',side_effect=self.clock):
            with self.assertRaisesRegex(ValueError,'Chưa lấy đủ'):
                enroll_camera(self.engine,self.store,'a','An',samples=3,timeout=3,warmup=0)
        self.assertEqual(self.store.daily_report('2026-11-25'),[])
        self.cap.release.assert_called_once()

    def test_read_failure_releases_without_saving(self):
        self.cap.read.return_value=(False,None)
        with patch('attendance.vision.open_capture',return_value=self.cap):
            with self.assertRaises(RuntimeError):
                enroll_camera(self.engine,self.store,'a','An')
        self.assertEqual(self.store.embeddings(),[])
        self.cap.release.assert_called_once()

    def test_different_person_samples_rejected(self):
        self.engine.identify.return_value=None
        with patch('attendance.vision.open_capture',return_value=self.cap), patch('time.monotonic',side_effect=self.clock):
            with self.assertRaises(ValueError):
                enroll_camera(self.engine,self.store,'a','An',samples=3,timeout=5,warmup=0)
        self.assertEqual(self.store.embeddings(),[])
