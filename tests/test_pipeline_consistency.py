import contextlib
import csv
import importlib.util
import io
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.ingest import ingest, start_spark
from src.preprocess import CDR_COLUMNS, TICKET_COLUMNS, tickets_as_of


@unittest.skipIf(os.name == "nt" or importlib.util.find_spec("pyspark") is None,
                 "Spark integration test requires the Linux Docker runtime")
class PipelineConsistencyTests(unittest.TestCase):
    def test_replay_parquet_and_repeat_ingestion(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = json.loads(Path("config.json").read_text())
            config.update(data_dir=str(root / "data"), processed_dir=str(root / "processed"),
                          state_dir=str(root / "state"), expected_days=3)
            for day in range(1, 4):
                for source in ("cdr", "tickets"):
                    path = root / "data" / "raw" / source / f"dt=2025-10-0{day}" / f"{source}.csv"
                    path.parent.mkdir(parents=True)
                    with path.open("w", newline="", encoding="utf-8") as stream:
                        columns = CDR_COLUMNS if source == "cdr" else [c for c in TICKET_COLUMNS if c != "bad_resolution_duration"]
                        writer = csv.DictWriter(stream, fieldnames=columns)
                        writer.writeheader()
                        if source == "cdr":
                            row = dict.fromkeys(columns, "0")
                            row.update(record_id=str(day), subscriber_key=f"SUB{day:09}",
                                       event_ts=f"2025-10-0{day} 12:00:00", ingest_ts=f"2025-10-0{day} 12:01:00",
                                       service_type=" VOICE ", direction="out", roaming_flag="false", bytes_up="-1")
                            writer.writerow(row)
                            if day == 1:
                                writer.writerow(row)
                                late = dict(row, record_id="late", subscriber_key="SUB000000009",
                                            ingest_ts="2025-10-03 00:01:00")
                                writer.writerow(late)
                                stream.write("broken,row\n")
                        else:
                            writer.writerow(dict(ticket_id="1", subscriber_key="SUB000000001",
                                created_ts="2025-10-01 12:00:00", closed_ts="2025-10-03 12:00:00",
                                channel="PHONE", category="quality" if day == 1 else "billing", resolution_code="RC1"))
            spark = start_spark(config)
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    first = ingest(config, 2, spark)
                    repeated = ingest(config, 2, spark)
                self.assertEqual(first, repeated)
                self.assertEqual(first["quality"]["cdr"]["dropped_malformed"], 1)
                self.assertEqual(first["quality"]["cdr"]["output_rows"], 2)
                self.assertNotIn("2025-10-01", first["complete_cdr_dates"])
                output = Path(first["output_dir"])
                cdr = spark.read.parquet(str(output / "clean/cdr"))
                self.assertEqual(cdr.select("record_id").distinct().count(), cdr.count())
                self.assertEqual({row.record_id for row in cdr.select("record_id").collect()}, {"1", "2"})
                daily = spark.read.parquet(str(output / "daily/cdr"))
                totals = daily.selectExpr("sum(event_count) AS events", "sum(voice_duration_valid_count) AS durations").first()
                self.assertEqual(totals.events, 2)
                self.assertEqual(totals.durations, 2)
                tickets = spark.read.parquet(str(output / "clean/tickets"))
                self.assertNotIn("text", tickets.columns)
                self.assertNotIn("retention_contact", tickets.columns)
                historical = tickets_as_of(tickets, date(2025, 10, 1)).first()
                self.assertEqual(historical.category, "quality")
                self.assertIsNone(historical.closed_ts)
                self.assertIsNone(historical.resolution_code)
                self.assertEqual(tickets_as_of(tickets, date(2025, 10, 2)).first().category, "billing")
                state_path = root / "state/ingest.json"
                with contextlib.redirect_stdout(io.StringIO()), patch("src.ingest.daily_cdr", side_effect=RuntimeError("test failure")):
                    with self.assertRaisesRegex(RuntimeError, "test failure"):
                        ingest(config, 3, spark)
                self.assertEqual(json.loads(state_path.read_text()), first)
                self.assertFalse((root / "state/ingest.lock").exists())
                with contextlib.redirect_stdout(io.StringIO()):
                    advanced = ingest(config, 3, spark)
                self.assertEqual(advanced["quality"]["cdr"]["output_rows"], 4)
                self.assertIn("2025-10-01", advanced["complete_cdr_dates"])
                manifest = json.loads((output / "manifest.json").read_text())
                self.assertTrue(all(item["partition"] <= "2025-10-02" for item in manifest["files"]))
            finally:
                spark.stop()


if __name__ == "__main__":
    unittest.main()
