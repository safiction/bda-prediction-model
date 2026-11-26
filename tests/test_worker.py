import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.final import read_json
from src.ingest import atomic_json
from src.worker import actions, current_job, load_config, replay_day, run_job, submit_job


class WorkerTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(Path("config.json").read_text())
        for name in ("data_dir", "audit_dir", "processed_dir", "snapshots_dir", "state_dir", "artifacts_dir"):
            config[name] = str(root / name)
        state = Path(config["state_dir"])
        state.mkdir()
        atomic_json(state / "ingest.json", {"through_day": 61, "fingerprint": "batch61"})
        atomic_json(state / "snapshots.json", {"input_fingerprint": "batch61", "snapshots":
            [{"snapshot_day": day, "label_status": "mature"} for day in (14, 21, 28, 46)]})
        run = Path(config["artifacts_dir"]) / "runs/first"
        run.mkdir(parents=True)
        atomic_json(run / "run.json", {"status": "completed", "milestone": 60, "input_fingerprint": "batch61", "candidates": {}})
        path = root / "config.json"
        atomic_json(path, config)
        return config, path

    def test_single_job_survives_refresh_and_prevents_duplicate_submission(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path = self.fixture(Path(temporary))
            with patch("src.worker.subprocess.Popen") as process:
                job = submit_job("prepare", path)
                self.assertEqual(current_job(load_config(path))["job_id"], job["job_id"])
                with self.assertRaises(ValueError):
                    submit_job("prepare", path)
                self.assertEqual(process.call_count, 1)

    def test_training_numbers_follow_runs_instead_of_dataset_days(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, _ = self.fixture(Path(temporary))
            state = Path(config["state_dir"])
            options = actions(config)
            self.assertTrue(options["prepare"]["enabled"])
            self.assertEqual(options["prepare"]["training_number"], 2)
            self.assertEqual(options["prepare"]["label"], "Открыть данные")
            atomic_json(state / "ingest.json", {"through_day": 68, "fingerprint": "batch68"})
            atomic_json(state / "snapshots.json", {"input_fingerprint": "batch68", "snapshots":
                        [{"snapshot_day": day, "label_status": "mature"} for day in (14, 21, 28, 35, 53)]})
            options = actions(config)
            self.assertTrue(options["train"]["enabled"])
            self.assertFalse(options["prepare"]["enabled"])
            self.assertEqual(options["train"]["training_number"], 2)
            self.assertEqual(options["train"]["label"], "Обучить модели")

    def test_missing_prepared_data_keeps_clock_and_never_starts_preprocessing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path = self.fixture(Path(temporary))
            with patch("src.worker.subprocess.Popen"):
                job = submit_job("prepare", path)

            before = (Path(config["state_dir"]) / "ingest.json").read_bytes()
            with patch("src.worker.reuse_prepared", return_value=False), patch("src.worker.run_command") as command:
                result = run_job(job["job_id"], path)
            command.assert_not_called()
            self.assertEqual(result["status"], "failed")
            self.assertIn("Подготовленные данные", result["error"])
            self.assertEqual((Path(config["state_dir"]) / "ingest.json").read_bytes(), before)
            self.assertEqual(replay_day(config), 61)
            self.assertTrue(actions(config)["prepare"]["enabled"])
            self.assertFalse((Path(config["state_dir"]) / "jobs/active.json").exists())

    def test_interrupted_worker_recovery_releases_only_its_stale_stage_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path = self.fixture(Path(temporary))
            with patch("src.worker.subprocess.Popen"):
                job = submit_job("prepare", path)
            jobs = Path(config["state_dir"]) / "jobs"
            job.update(created_timestamp=1, pid=999999, stage_pid=999998, status="running")
            atomic_json(jobs / f"{job['job_id']}.json", job)
            (Path(config["state_dir"]) / "ingest.lock").write_text("999998")
            with patch("src.worker.process_alive", return_value=False):
                self.assertIsNone(current_job(config))
            self.assertEqual(read_json(jobs / f"{job['job_id']}.json")["status"], "failed")
            self.assertFalse((Path(config["state_dir"]) / "ingest.lock").exists())

    def test_launch_failure_recorded_and_bad_actions_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path = self.fixture(Path(temporary))
            with self.assertRaises(ValueError):
                submit_job("arbitrary_command", path)
            with patch("src.worker.subprocess.Popen", side_effect=OSError("launch test failure")):
                with self.assertRaises(OSError):
                    submit_job("prepare", path)
            jobs = Path(config["state_dir"]) / "jobs"
            records = [read_json(item) for item in jobs.glob("*.json")]
            self.assertEqual(records[0]["status"], "failed")
            self.assertFalse((jobs / "active.json").exists())

    def test_reader_does_not_release_claim_while_completed_worker_is_finalizing(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, path = self.fixture(Path(temporary))
            with patch("src.worker.subprocess.Popen"):
                job = submit_job("prepare", path)
            jobs = Path(config["state_dir"]) / "jobs"
            job.update(status="completed", pid=12345)
            atomic_json(jobs / f"{job['job_id']}.json", job)
            with patch("src.worker.process_alive", return_value=True):
                self.assertIsNotNone(current_job(config))
                self.assertTrue((jobs / "active.json").exists())
            with patch("src.worker.process_alive", return_value=False):
                self.assertIsNone(current_job(config))


if __name__ == "__main__":
    unittest.main()
