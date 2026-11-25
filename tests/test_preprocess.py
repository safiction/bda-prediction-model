import json
import unittest
from collections import Counter
from datetime import date
from pathlib import Path

from src.preprocess import clean_row
from src.temporal import replay_dates


class CleaningTests(unittest.TestCase):
    def test_malformed_rows_are_dropped(self):
        stats = Counter()
        self.assertIsNone(clean_row({"record_id": "1", "charge": None}, "cdr",
                                   date(2025, 10, 1), Path("cdr.csv"), 1, date(2025, 10, 2), stats))
        self.assertEqual(stats["dropped_malformed"], 1)

    def test_numeric_defects_preserve_activity_and_late_records_wait(self):
        row = dict(record_id="1", subscriber_key="SUB000000001", event_ts="01.10.2025 12:00:00",
                   ingest_ts="2025-10-02 00:01:00", duration_sec="0", bytes_up="-1",
                   bytes_down="nan", charge="0", service_type=" VOICE ", roaming_flag="False")
        stats = Counter()
        args = (row, "cdr", date(2025, 10, 1), Path("cdr.csv"), 1)
        self.assertIsNone(clean_row(*args, date(2025, 10, 1), stats))
        clean = clean_row(*args, date(2025, 10, 2), stats)
        self.assertEqual(clean["duration_sec"], 0)
        self.assertEqual(clean["charge"], 0)
        self.assertIsNone(clean["bytes_up"])
        self.assertIsNone(clean["bytes_down"])
        self.assertEqual(clean["service_type"], "voice")
        self.assertEqual(clean["available_date"], "2025-10-02")

    def test_forbidden_ticket_fields_are_removed(self):
        row = dict(ticket_id="1", subscriber_key="SUB000000001", created_ts="2025-10-01 12:00:00",
                   closed_ts="2025-10-01 10:00:00", text="not a feature", retention_contact="True",
                   category=" QUALITY ", channel="PHONE", resolution_code="RC1")
        clean = clean_row(row, "tickets", date(2025, 10, 1), Path("tickets.csv"), 1,
                          date(2025, 10, 2), Counter())
        self.assertNotIn("text", clean)
        self.assertNotIn("retention_contact", clean)
        self.assertEqual(clean["category"], "quality")
        self.assertIsNone(clean["closed_ts"])
        self.assertEqual(clean["bad_resolution_duration"], "true")

    def test_day_91_requires_no_new_event_partition(self):
        config = json.loads(Path("config.json").read_text())
        event_end, availability = replay_dates(date(2025, 10, 1), 91, config)
        self.assertEqual(str(event_end), "2025-12-29")
        self.assertEqual(str(availability), "2025-12-30")
        with self.assertRaises(ValueError):
            replay_dates(date(2025, 10, 1), 92, config)


if __name__ == "__main__":
    unittest.main()
