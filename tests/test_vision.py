import unittest
from attendance.vision import FaceEngine


class RecognitionTests(unittest.TestCase):
    def setUp(self):
        self.engine = FaceEngine.__new__(FaceEngine)
        self.engine.threshold = .8
        self.engine.margin = .05

    def test_multiple_samples_same_identity_not_ambiguous(self):
        result = self.engine.identify([1, 0], [('a', 'An', [1, 0]), ('a', 'An', [.99, .01]), ('b', 'Binh', [0, 1])])
        self.assertEqual(result[:2], ('a', 'An'))

    def test_ambiguous_identity_rejected(self):
        self.assertIsNone(self.engine.identify([1, 0], [('a', 'An', [1, 0]), ('b', 'Binh', [.99, .01])]))

    def test_unknown_and_empty_gallery(self):
        self.assertIsNone(self.engine.identify([1, 0], [('a', 'An', [0, 1])]))
        self.assertIsNone(self.engine.identify([1, 0], []))

    def test_bad_embeddings_rejected(self):
        for vector in ([0, 0], [float('nan'), 1], []):
            with self.assertRaises(ValueError):
                self.engine.identify(vector, [])
        with self.assertRaises(ValueError):
            self.engine.identify([1, 0], [('a', 'An', [1, 0, 0])])
