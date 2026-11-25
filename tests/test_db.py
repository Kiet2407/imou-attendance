import sqlite3
import unittest
from datetime import datetime, timezone

from attendance.db import Store


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")
        self.addCleanup(self.store.close)
        self.store.add_employee("E01", "Nguyễn An")
        self.store.add_employee("E02", "Trần Bình")

    def test_missing_employee_is_included(self):
        report = self.store.daily_report("2026-11-25")
        self.assertEqual(len(report), 2)
        self.assertIsNone(report[0]["first_arrival"])
        self.assertIsNone(report[0]["camera"])

    def test_earliest_event_with_matching_camera_even_if_inserted_later(self):
        self.store.record_entry("E01", datetime.fromisoformat("2026-11-25T09:00:00+07:00"), "second")
        self.store.record_entry("E01", datetime.fromisoformat("2026-11-25T01:00:00+00:00"), "first")
        report = self.store.daily_report("2026-11-25")
        self.assertEqual(report[0]["first_arrival"], "2026-11-25T08:00:00+07:00")
        self.assertEqual(report[0]["camera"], "first")

    def test_day_boundaries_use_office_timezone(self):
        for timestamp in ("2026-11-24T16:59:59+00:00", "2026-11-24T17:00:00+00:00", "2026-11-25T17:00:00+00:00"):
            self.store.record_entry("E01", datetime.fromisoformat(timestamp))
        report = self.store.daily_report("2026-11-25")
        self.assertEqual(report[0]["first_arrival"], "2026-11-25T00:00:00+07:00")
        next_report = self.store.daily_report("2026-11-26")
        self.assertEqual(next_report[0]["first_arrival"], "2026-11-26T00:00:00+07:00")

    def test_utc_report_is_supported(self):
        self.store.record_entry("E01", datetime.fromisoformat("2026-11-24T23:00:00+00:00"))
        self.assertIsNone(self.store.daily_report("2026-11-25", "UTC")[0]["first_arrival"])

    def test_updating_employee_preserves_embeddings_and_events(self):
        self.store.save_embedding("E01", [1, 0.5])
        self.store.record_entry("E01", datetime(2026, 11, 25, 1, tzinfo=timezone.utc))
        self.store.add_employee("E01", "Tên mới")
        self.assertEqual(len(self.store.daily_report("2026-11-25")), 2)
        self.assertEqual(self.store.embeddings(), [("E01", "Tên mới", [1.0, 0.5])])
        self.assertIsNotNone(self.store.daily_report("2026-11-25")[0]["first_arrival"])

    def test_multiple_face_samples(self):
        self.store.save_embedding("E01", [1, 0])
        self.store.save_embedding("E01", [0.9, 0.1])
        self.assertEqual(len(self.store.embeddings()), 2)

    def test_naive_datetime_is_rejected(self):
        with self.assertRaises(ValueError):
            self.store.record_entry("E01", datetime(2026, 11, 25, 8))

    def test_unknown_employee_is_rejected(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.save_embedding("unknown", [1])
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.record_entry("unknown", datetime.now(timezone.utc))

    def test_invalid_embeddings_rejected(self):
        for vector in ([], [0, 0], [float("nan")], [float("inf")], None):
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                self.store.save_embedding("E01", vector)

    def test_empty_employee_rejected(self):
        with self.assertRaises(ValueError):
            self.store.add_employee(" ", "Name")


if __name__ == "__main__":
    unittest.main()
