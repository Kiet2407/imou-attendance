import queue
import threading
import unittest
from unittest.mock import patch

from attendance.live import LatestCapture


class ControlledCapture:
    def __init__(self):
        self.frames = queue.Queue()
        self.released = threading.Event()

    def read(self):
        item = self.frames.get(timeout=2)
        return (False, None) if item is None else (True, item)

    def release(self):
        self.released.set()


class LatestCaptureTests(unittest.TestCase):
    def setUp(self):
        self.source = ControlledCapture()
        with patch('attendance.live.open_capture', return_value=self.source):
            self.capture = LatestCapture('mock')

    def tearDown(self):
        self.source.frames.put(None)
        self.capture.close()

    def wait_sequence(self, sequence):
        with self.capture._condition:
            self.assertTrue(self.capture._condition.wait_for(
                lambda: self.capture._sequence >= sequence, timeout=1))

    def test_latest_only_and_no_repeated_frame(self):
        self.source.frames.put('old')
        self.source.frames.put('new')
        self.wait_sequence(2)
        ok, frame, acquired_at, tick = self.capture.read(.1)
        self.assertTrue(ok)
        self.assertEqual(frame, 'new')
        self.assertIsNotNone(acquired_at.utcoffset())
        self.assertGreater(tick, 0)
        self.assertEqual(self.capture.read(.01), (False, None, None, None))
        self.source.frames.put('next')
        self.assertEqual(self.capture.read(1)[1], 'next')

    def test_read_failure_notifies_and_releases(self):
        self.source.frames.put(None)
        self.assertEqual(self.capture.read(1), (False, None, None, None))
        self.capture.close()
        self.assertTrue(self.source.released.is_set())
        self.assertFalse(self.capture._thread.is_alive())

    def test_close_releases_and_future_reads_fail(self):
        self.source.frames.put(None)
        self.capture.close()
        self.assertTrue(self.source.released.is_set())
        self.assertEqual(self.capture.read(0), (False, None, None, None))
        self.assertFalse(self.capture._thread.is_alive())

    def test_invalid_timeout_rejected(self):
        for value in (-1, 13, float('nan')):
            with self.assertRaises(ValueError):
                self.capture.read(value)
