import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path

from src.features import KEY_COLUMNS, build_features, validate_feature_columns
from src.ingest import atomic_json, start_spark
from src.labels import build_labels
from src.preprocess import daily_cdr
from src.snapshots import build_snapshots
from src.temporal import temporal_split_manifest


class SplitAndSchemaTests(unittest.TestCase):
    def test_chronological_manifest_and_feature_allowlist(self):
        config = json.loads(Path("config.json").read_text())
        manifest = temporal_split_manifest(date(2025, 10, 1), config)
        for run in manifest["runs"]:
            self.assertLess(max(row["outcome_end"] for row in run["training"]), run["validation"]["outcome_start"])
            self.assertNotIn(76, [row["snapshot_day"] for row in run["training"]])
        self.assertEqual(manifest["final_delivery_clock_day"], 91)
        validate_feature_columns(KEY_COLUMNS + config["feature_allowlist"], config)
        for forbidden in ("target", "retention_contact", "text", "registration_date"):
            with self.assertRaises(ValueError):
                validate_feature_columns(KEY_COLUMNS + config["feature_allowlist"] + [forbidden], config)


@unittest.skipIf(os.name == "nt" or importlib.util.find_spec("pyspark") is None,
                 "Spark temporal tests require the Linux Docker runtime")
class HistoricalSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = json.loads(Path("config.json").read_text())
        cls.spark = start_spark(cls.config)
        cls.start = date(2025, 10, 1)
        cls.snapshot = date(2025, 10, 14)
        rows = []

        def event(identifier, subscriber, day, service="voice", duration=0., up=0., down=0., charge=0., roaming=False, available=None):
            when = datetime(2025, 10, day, 12)
            arrival = datetime(2025, 10, available or day, 12, 1)
            rows.append((identifier, f"SUB{subscriber:09}", when, arrival, service, duration,
                         up, down, charge, roaming, arrival.date(), when.date()))

        event("a1", 1, 1, duration=0., up=10., down=20.)
        event("a2", 1, 1, service="data", up=5., down=None, charge=None, roaming=None, available=2)
        event("a3", 1, 8, duration=10., charge=4., roaming=True)
        event("a4", 1, 14, duration=20., up=1., down=1., charge=2.)
        event("late", 1, 14, service="data", up=1000., down=1000., charge=1000., available=15)
        event("b", 2, 14, service="sms")
        event("c", 3, 14)
        event("only_late", 5, 14, available=15)
        event("first_outcome", 1, 15)
        event("last_outcome", 2, 28)
        event("outside_outcome", 3, 29)
        event("future_only", 4, 15)
        cls.cdr = cls.spark.createDataFrame(rows, "record_id string, subscriber_key string, event_ts timestamp, ingest_ts timestamp, "
            "service_type string, duration_sec double, bytes_up double, bytes_down double, charge double, roaming_flag boolean, "
            "available_date date, event_date date").cache()
        tickets = []

        def ticket(identifier, subscriber, created, closed, category, available):
            tickets.append((identifier, f"SUB{subscriber:09}", datetime(2025, 10, created, 12),
                datetime(2025, 10, closed, 12) if closed else None, category, "rc1",
                date(2025, 10, available), date(2025, 10, available), "tickets.csv", available))

        ticket("1", 1, 1, 15, "quality", 1)
        ticket("1", 1, 1, 10, "billing", 15)
        ticket("2", 1, 8, 10, "billing", 8)
        ticket("3", 2, 14, None, "connection", 14)
        ticket("4", 1, 15, None, "quality", 15)
        cls.tickets = cls.spark.createDataFrame(tickets, "ticket_id string, subscriber_key string, created_ts timestamp, "
            "closed_ts timestamp, category string, resolution_code string, available_date date, source_partition date, "
            "source_file string, source_row long").cache()
        cls.cdr.count()
        cls.tickets.count()

    @classmethod
    def tearDownClass(cls):
        cls.cdr.unpersist()
        cls.tickets.unpersist()
        cls.spark.stop()

    def features(self, cdr=None, tickets=None):
        return build_features(daily_cdr(self.cdr if cdr is None else cdr),
                              self.tickets if tickets is None else tickets, self.snapshot, self.config)

    def test_homework_aggregations_with_correct_historical_definitions(self):
        rows = {row.subscriber_key: row.asDict() for row in self.features().collect()}
        self.assertEqual(set(rows), {f"SUB{i:09}" for i in (1, 2, 3)})
        a = rows["SUB000000001"]
        self.assertEqual(a["event_count"], 4)
        self.assertEqual(a["active_days"], 3)
        self.assertEqual(a["days_since_last_activity"], 0)
        self.assertEqual(a["voice_count"], 3)
        self.assertEqual(a["avg_voice_duration_sec"], 10.)
        self.assertAlmostEqual(a["avg_traffic_bytes_per_event"], 32 / 3)
        self.assertEqual(a["total_bytes_up"], 16.)
        self.assertEqual(a["avg_charge"], 2.)
        self.assertAlmostEqual(a["roaming_share"], 1 / 3)
        self.assertEqual((a["events_last_7d"], a["events_previous_7d"]), (2, 2))
        self.assertEqual((a["ticket_count"], a["closed_ticket_count"], a["unresolved_ticket_count"]), (2, 1, 1))
        self.assertEqual(a["avg_resolution_hours"], 48.)
        self.assertEqual(rows["SUB000000003"]["ticket_count"], 0)
        self.assertIsNone(rows["SUB000000003"]["avg_resolution_hours"])
        self.assertEqual(rows["SUB000000003"]["resolution_missing"], 1)

    def test_future_events_and_ticket_versions_do_not_change_past_features(self):
        past_cdr = self.cdr.filter("event_date <= '2025-10-14' AND available_date <= '2025-10-14'")
        past_tickets = self.tickets.filter("available_date <= '2025-10-14'")
        expected = self.features(past_cdr, past_tickets).orderBy("subscriber_key").collect()
        self.assertEqual(self.features().orderBy("subscriber_key").collect(), expected)
        from pyspark.sql import functions as F
        with self.assertRaises(ValueError):
            self.features(tickets=self.tickets.withColumn("retention_contact", F.lit(True)))

    def test_outcome_boundaries_and_unknown_horizons(self):
        cohort = self.features()
        complete = {self.start + timedelta(days=i) for i in range(28)}
        labels = build_labels(self.cdr, cohort, self.snapshot, date(2025, 10, 29), complete)
        self.assertEqual({row.subscriber_key: row.target for row in labels.collect()},
                         {"SUB000000001": 0, "SUB000000002": 0, "SUB000000003": 1})
        missing = complete - {date(2025, 10, 20)}
        self.assertTrue(all(row.target is None for row in build_labels(
            self.cdr, cohort, self.snapshot, date(2025, 10, 29), missing).collect()))
        self.assertTrue(all(row.target is None for row in build_labels(
            self.cdr, cohort, self.snapshot, date(2025, 10, 27), complete).collect()))

    def test_snapshot_parquets_pending_status_and_repeated_build(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            batch = root / "processed/batch"
            self.cdr.write.parquet(str(batch / "clean/cdr"))
            self.tickets.write.parquet(str(batch / "clean/tickets"))
            daily_cdr(self.cdr).write.parquet(str(batch / "daily/cdr"))
            manifest = {"files": [{"source": source, "partition": str(self.start + timedelta(days=i))}
                                   for source in ("cdr", "tickets") for i in range(29)]}
            atomic_json(batch / "manifest.json", manifest)
            state_dir = root / "state"
            state_dir.mkdir()
            atomic_json(state_dir / "ingest.json", {"output_dir": str(batch), "fingerprint": "a" * 64,
                "available_through": "2025-10-29", "loaded_event_end": "2025-10-29",
                "complete_cdr_dates": [str(self.start + timedelta(days=i)) for i in range(28)]})
            config = dict(self.config, state_dir=str(state_dir), processed_dir=str(root / "processed"),
                          snapshots_dir=str(root / "snapshots"))
            with contextlib.redirect_stdout(io.StringIO()):
                result = build_snapshots(config, [14, 21], self.spark)
                repeated = build_snapshots(config, [14, 21], self.spark)
            self.assertEqual(result, repeated)
            mature, pending = result["snapshots"]
            self.assertEqual(mature["class_counts"], {"0": 2, "1": 1})
            self.assertEqual(mature["label_status"], "mature")
            self.assertEqual(pending["label_status"], "pending")
            self.assertEqual(pending["class_counts"], {"unknown": pending["feature_rows"]})
            saved = self.spark.read.parquet(str(Path(mature["output_dir"]) / "features"))
            self.assertEqual(saved.select(*KEY_COLUMNS).distinct().count(), mature["feature_rows"])
            self.assertTrue((Path(result["output_dir"]) / "split_manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
