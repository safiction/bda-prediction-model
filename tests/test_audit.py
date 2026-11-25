import csv
import contextlib
import io
import json
import sqlite3
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

from src.audit import create_database, measure_snapshot, run_audit, scan_file
from src.data import discover_files, inspect_file, parse_timestamp
from src.temporal import outcome_is_complete, snapshot_schedule


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        self.start = date(2025, 10, 1)
        self.connection = sqlite3.connect(":memory:")
        create_database(self.connection)
        self.addCleanup(self.connection.close)

    def add_event(self, record_id, subscriber, day, available=None):
        self.connection.execute("INSERT INTO cdr VALUES (?, ?, ?, ?, ?)",
                                (record_id, b"x", subscriber, day.isoformat(),
                                 (available or day).isoformat()))

    def test_snapshot_boundaries_and_delayed_availability(self):
        snapshot = self.start + timedelta(days=13)
        end = snapshot + timedelta(days=14)
        self.add_event("1", "SUB000000001", snapshot)
        self.add_event("2", "SUB000000001", end)
        self.add_event("3", "SUB000000002", snapshot)
        self.add_event("4", "SUB000000002", end + timedelta(days=1))
        self.add_event("5", "SUB000000003", snapshot + timedelta(days=1))
        self.add_event("6", "SUB000000004", snapshot, snapshot + timedelta(days=1))
        dates = {self.start + timedelta(days=i) for i in range(90)}
        result = measure_snapshot(self.connection, self.start, 14, self.config, dates)
        self.assertEqual(result["eligible"], 2)
        self.assertEqual(result["diagnostic_active"], 1)
        self.assertEqual(result["diagnostic_inactive"], 1)
        self.assertTrue(result["labels_ready_at_horizon"])
        self.add_event("7", "SUB000000001", end, end + timedelta(days=2))
        delayed = measure_snapshot(self.connection, self.start, 14, self.config, dates)
        self.assertEqual(delayed["eligible"], result["eligible"])
        self.assertFalse(delayed["labels_ready_at_horizon"])
        self.assertEqual(delayed["unknown_at_horizon"], 2)

    def test_missing_outcome_day_is_unknown(self):
        dates = {self.start + timedelta(days=i) for i in range(27)}
        self.assertFalse(outcome_is_complete(self.start + timedelta(days=13), 14,
                                             self.start + timedelta(days=89), dates))
        result = measure_snapshot(self.connection, self.start, 14, self.config, dates)
        self.assertIsNone(result["diagnostic_inactive"])

    def test_training_outcomes_precede_validation_outcomes(self):
        schedule = snapshot_schedule(self.config)
        self.assertEqual(list(schedule), [14, 21, 28, 35, 42, 46, 53, 60, 76])
        for replay in self.config["training_replay_days"]:
            training = [day for day, roles in schedule.items() if (f"train at {replay}", replay) in roles]
            self.assertLessEqual(max(training) + 14, replay - 14)

    def test_discovery_excludes_other_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            for source in ("cdr", "tickets", "telemetry", "reference"):
                path = Path(temporary) / "raw" / source / "dt=2025-10-01" / f"{source}.csv"
                path.parent.mkdir(parents=True)
                path.write_text("test", encoding="utf-8")
            self.assertEqual({source for _, source, _ in discover_files(temporary)}, {"cdr", "tickets"})

    def test_encoding_fallback_checks_entire_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tickets.csv"
            path.write_bytes(b"x" * (1024 * 1024 + 1) + "Пример".encode("cp1251"))
            encoding, fingerprint = inspect_file(path, self.config["encodings"])
            self.assertEqual(encoding, "cp1251")
            self.assertEqual(len(fingerprint), 64)

    def test_mixed_timestamp_formats(self):
        self.assertEqual(parse_timestamp("01.10.2025 11:02:38"),
                         parse_timestamp("2025-10-01 11:02:38"))
        self.assertIsNone(parse_timestamp("not a timestamp"))
        self.assertIsNone(parse_timestamp("2025-10-01 11:02:38+03:00"))

    def test_deduplication_keeps_activity_despite_numeric_defects(self):
        columns = ["record_id", "subscriber_key", "event_ts", "ingest_ts",
                   "duration_sec", "bytes_up", "bytes_down", "charge"]
        row = ["1", "SUB000000001", "2025-10-01 12:00:00", "2025-10-01 12:01:00", "0", "-1", "0", "0"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cdr.csv"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(columns)
                writer.writerow(row)
                writer.writerow(row)
                writer.writerow(row[:-1] + ["2"])
                writer.writerow(row[1:])
            stats = scan_file(self.connection, self.start, "cdr", path, self.config)["quality"]
            self.assertEqual(stats["rows"], 4)
            self.assertEqual(stats["malformed_csv_rows"], 1)
            self.assertEqual(stats["duplicate_id_rows"], 2)
            self.assertEqual(stats["conflicting_duplicate_rows"], 1)
            self.assertEqual(stats["invalid_bytes_up"], 3)
            self.assertEqual(self.connection.execute("SELECT COUNT(*) FROM cdr").fetchone()[0], 1)

    def test_small_audit_writes_reproducible_reports(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for day in range(4):
                event_date = self.start + timedelta(days=day)
                for source in ("cdr", "tickets"):
                    path = root / "raw" / source / f"dt={event_date}" / f"{source}.csv"
                    path.parent.mkdir(parents=True)
                    with path.open("w", encoding="utf-8", newline="") as stream:
                        writer = csv.writer(stream)
                        if source == "cdr":
                            writer.writerow(["record_id", "subscriber_key", "event_ts", "ingest_ts",
                                             "duration_sec", "bytes_up", "bytes_down", "charge"])
                            for index, subscriber in enumerate((1, 2 if day % 2 == 0 else 3)):
                                writer.writerow([f"{day}-{index}", f"SUB{subscriber:09}",
                                                 f"{event_date} 12:00:00", f"{event_date} 12:01:00",
                                                 0, 0, 0, 0])
                        else:
                            writer.writerow(["ticket_id", "subscriber_key", "created_ts", "closed_ts"])
                            writer.writerow([str(day), "SUB000000001", f"{event_date} 12:00:00", ""])
            config = dict(self.config, data_dir=str(root), audit_dir=str(root / "audit"),
                          expected_days=4, feature_window_days=1, outcome_window_days=1,
                          snapshot_stride_days=1, training_replay_days=[3],
                          final_snapshot_day=3, final_replay_day=4)
            with contextlib.redirect_stdout(io.StringIO()):
                first = run_audit(config)
                second = run_audit(config)
            self.assertEqual(first, second)
            self.assertEqual(first["gate"], "passed")
            self.assertTrue((root / "audit/report.md").is_file())
            self.assertEqual(first["unique_valid_ids"]["cdr"], 8)
            self.assertTrue(all(row["diagnostic_inactive"] == 1 for row in first["snapshots"]))


if __name__ == "__main__":
    unittest.main()
