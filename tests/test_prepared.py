import hashlib
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.ingest import atomic_json
from src.prepared import reuse_prepared
from src.temporal import date_for_day, snapshot_windows
from src.worker import run_job, submit_job


class PreparedDataTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(Path("config.json").read_text(encoding="utf-8"))
        for key in ("data_dir", "processed_dir", "snapshots_dir", "state_dir", "artifacts_dir", "audit_dir"):
            config[key] = str(root / key)
            Path(config[key]).mkdir()
        batch = Path(config["processed_dir"]) / "day=061_ready"
        batch.mkdir()
        files = []
        for kind in ("cdr", "tickets"):
            for day in range(1, 62):
                partition = str(date_for_day(date(2025, 10, 1), day))
                path = Path(config["data_dir"]) / f"raw/{kind}/dt={partition}/{kind}.csv"
                path.parent.mkdir(parents=True)
                path.write_bytes(b"original input")
                files.append({"source": kind, "partition": partition, "path": path.as_posix(),
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            (batch / "clean" / kind).mkdir(parents=True)
        (batch / "daily/cdr").mkdir(parents=True)
        atomic_json(batch / "manifest.json", {"config": config, "files": files})
        atomic_json(batch / "quality.json", {"through_day": 61, "available_through": "2025-11-30",
                    "loaded_event_end": "2025-11-30", "fingerprint": "ready", "output_dir": str(batch)})
        snapshots = Path(config["snapshots_dir"]) / "ready_features"
        for day in (14, 21, 28, 46):
            metadata = snapshot_windows(date(2025, 10, 1), day, config)
            folder = snapshots / f"snapshot_date={metadata['snapshot_date']}"
            (folder / "features").mkdir(parents=True)
            (folder / "labels").mkdir()
            atomic_json(folder / "manifest.json", {**metadata, "input_fingerprint": "ready",
                        "available_through": "2025-11-30", "feature_names": config["feature_allowlist"],
                        "label_status": "mature", "missing_outcome_dates": []})
        path = root / "config.json"
        atomic_json(path, config)
        return config, path, files

    def test_exact_cutoff_reused_and_modified_or_missing_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, files = self.fixture(Path(temporary))
            self.assertFalse(reuse_prepared(config, 68, [14]))
            self.assertTrue(reuse_prepared(config, 61, [14, 46]))
            state = Path(config["state_dir"])
            self.assertEqual(json.loads((state / "ingest.json").read_text())["through_day"], 61)
            before = (state / "snapshots.json").read_bytes()
            Path(files[0]["path"]).write_bytes(b"changed input")
            self.assertFalse(reuse_prepared(config, 61, [14, 46]))
            self.assertEqual((state / "snapshots.json").read_bytes(), before)
            Path(files[0]["path"]).unlink()
            self.assertFalse(reuse_prepared(config, 61, [14]))

    def test_different_feature_rules_and_unavailable_snapshots_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _, _ = self.fixture(Path(temporary))
            self.assertFalse(reuse_prepared(config, 61, [53]))
            config["feature_allowlist"] = config["feature_allowlist"][:-1]
            self.assertFalse(reuse_prepared(config, 61, [14]))
            self.assertFalse((Path(config["state_dir"]) / "ingest.json").exists())

    def test_worker_skips_cleaning_and_spark_for_a_prepared_batch(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path, _ = self.fixture(Path(temporary))
            with patch("src.worker.subprocess.Popen"):
                job = submit_job("prepare", path)
            with patch("src.worker.run_command", side_effect=AssertionError("Prepared data must not start a subprocess")):
                result = run_job(job["job_id"], path)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(json.loads((Path(config["state_dir"]) / "replay.json").read_text())["through_day"], 61)
