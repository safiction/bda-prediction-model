import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.evaluate import choose_model, evaluate_predictions
from src.ingest import atomic_json
from src.predict import load_model, predict_probabilities
from src.temporal import date_for_day, temporal_split_manifest
from src.train import load_training_data, train


class SelectionRuleTests(unittest.TestCase):
    def test_strict_improvement_and_ties_keep_incumbent(self):
        config = json.loads(Path("config.json").read_text())
        scores = {"logistic": {"pr_auc": 0.4}, "xgboost": {"pr_auc": 0.5}}
        self.assertEqual(choose_model(scores, None, None, config)[0], "xgboost")
        self.assertEqual(choose_model(scores, "current", {"pr_auc": 0.5}, config)[0], "current")
        self.assertEqual(choose_model(scores, "current", {"pr_auc": 0.4}, config)[0], "xgboost")


DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("numpy", "pandas", "pyarrow", "sklearn", "xgboost"))


@unittest.skipUnless(DEPENDENCIES, "Model tests require the pinned Docker dependencies")
class ModelTests(unittest.TestCase):
    def fixture(self, root, clock=61):
        import numpy as np
        import pandas as pd
        config = json.loads(Path("config.json").read_text())
        for name in ("processed_dir", "snapshots_dir", "state_dir", "artifacts_dir"):
            config[name] = str(root / name)
        state = Path(config["state_dir"])
        state.mkdir(exist_ok=True)
        batch = Path(config["processed_dir"]) / f"batch{clock}"
        snapshots = Path(config["snapshots_dir"]) / f"batch{clock}"
        batch.mkdir(parents=True)
        snapshots.mkdir(parents=True)
        start = date(2025, 10, 1)
        available = str(date_for_day(start, clock))
        fingerprint = f"fixture_{clock}"
        atomic_json(batch / "manifest.json", {"files": [{"source": name, "partition": str(start), "sha256": "fixture"}
                                                       for name in ("cdr", "tickets")]})
        milestone = clock - config["label_delivery_lag_days"]
        split = next(item for item in temporal_split_manifest(start, config)["runs"] if item["milestone_day"] == milestone)
        for windows in split["training"] + [split["validation"]]:
            day = windows["snapshot_day"]
            size = 44 if day == 53 else 40
            rng = np.random.default_rng(day)
            target = np.arange(size) % 2
            values = {name: np.abs(rng.normal(size=size)) + target * 0.2 for name in config["feature_allowlist"]}
            values["event_count"] = np.arange(size, dtype=float) + (1000 if day in (46, 53) else 1)
            values["avg_resolution_hours"][:] = np.nan
            features = pd.DataFrame(values)
            features.insert(0, "snapshot_date", windows["snapshot_date"])
            features.insert(0, "subscriber_key", [f"SUB{i:09}" for i in range(size)])
            labels = features[["subscriber_key", "snapshot_date"]].copy()
            labels["target"] = target
            folder = snapshots / f"snapshot_date={windows['snapshot_date']}"
            folder.mkdir()
            features.to_parquet(folder / "features.parquet", index=False)
            labels.to_parquet(folder / "labels.parquet", index=False)
            (folder / "features").mkdir()
            (folder / "labels").mkdir()
            (folder / "features.parquet").replace(folder / "features/part.parquet")
            (folder / "labels.parquet").replace(folder / "labels/part.parquet")
            atomic_json(folder / "manifest.json", {**windows, "input_fingerprint": fingerprint,
                "available_through": available, "label_status": "mature", "missing_outcome_dates": [],
                "feature_rows": size, "feature_names": config["feature_allowlist"],
                "feature_types": {name: "double" for name in config["feature_allowlist"]},
                "class_counts": {"0": size // 2, "1": size // 2}})
        atomic_json(state / "ingest.json", {"through_day": clock, "available_through": available,
                                           "output_dir": str(batch), "fingerprint": fingerprint})
        atomic_json(state / "snapshots.json", {"available_through": available,
                                              "output_dir": str(snapshots), "input_fingerprint": fingerprint})
        return config

    def test_metrics_use_raw_probabilities_and_report_undefined_values(self):
        import numpy as np
        metrics = evaluate_predictions([0, 1], [0.1, 0.2])
        self.assertAlmostEqual(metrics["log_loss"], -(np.log(0.9) + np.log(0.2)) / 2)
        self.assertEqual(metrics["pr_auc"], 1.)
        self.assertIsNone(metrics["precision"])
        self.assertEqual(metrics["recall"], 0.)
        self.assertEqual(metrics["confusion_matrix"], {"tn": 1, "fp": 0, "fn": 1, "tp": 0})
        loss32 = evaluate_predictions([0, 1], np.array([1, 0], dtype=np.float32))["log_loss"]
        loss64 = evaluate_predictions([0, 1], np.array([1, 0], dtype=np.float64))["log_loss"]
        self.assertEqual(loss32, loss64)
        single = evaluate_predictions([0, 0], [0.1, 0.2])
        self.assertIsNone(single["roc_auc"])
        self.assertIsNone(single["pr_auc"])
        self.assertIsNone(single["recall"])

    def test_reload_and_preprocessing_fit_training_only(self):
        import numpy as np
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            config = self.fixture(Path(temporary))
            with contextlib.redirect_stdout(io.StringIO()):
                result = train(config, 60)
            training, validation, _ = load_training_data(config, 60)
            x_validation = validation[config["feature_allowlist"]]
            for version in result["candidates"]:
                folder = Path(config["artifacts_dir"]) / "models" / version
                pipeline, metadata = load_model(folder)
                imputer = pipeline.named_steps["imputer"]
                event_position = config["feature_allowlist"].index("event_count")
                self.assertEqual(imputer.statistics_[event_position], training.event_count.median())
                self.assertLess(imputer.statistics_[event_position], validation.event_count.min())
                self.assertEqual(len(imputer.statistics_), len(config["feature_allowlist"]))
                if "scaler" in pipeline.named_steps:
                    self.assertEqual(pipeline.named_steps["scaler"].n_samples_seen_, len(training))
                saved = pd.read_parquet(folder / "validation.parquet")
                np.testing.assert_allclose(predict_probabilities(pipeline, x_validation, metadata["feature_names"]),
                                           saved.probability, rtol=1e-10, atol=1e-12)
                reversed_columns = list(reversed(metadata["feature_names"]))
                np.testing.assert_allclose(predict_probabilities(pipeline, x_validation[reversed_columns], metadata["feature_names"]),
                                           saved.probability, rtol=1e-10, atol=1e-12)
                with self.assertRaises(ValueError):
                    predict_probabilities(pipeline, x_validation.assign(retention_contact=True), metadata["feature_names"])

    def test_final_outcomes_never_read_and_later_replay_rejected(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            final_date = str(date_for_day(date(2025, 10, 1), 76))
            state = json.loads((Path(config["state_dir"]) / "snapshots.json").read_text())
            (Path(state["output_dir"]) / f"snapshot_date={final_date}" / "labels").mkdir(parents=True)
            original = pd.read_parquet

            def guarded(path, *args, **kwargs):
                if final_date in str(path):
                    raise AssertionError("Final outcomes must not be read")
                return original(path, *args, **kwargs)

            with patch("pandas.read_parquet", side_effect=guarded), contextlib.redirect_stdout(io.StringIO()):
                result = train(config, 60)
            self.assertEqual(result["status"], "completed")
            state_path = Path(config["state_dir"]) / "ingest.json"
            state = json.loads(state_path.read_text())
            state["through_day"] = 91
            atomic_json(state_path, state)
            with self.assertRaisesRegex(ValueError, "later replay data"):
                load_training_data(config, 60)

    def test_incumbent_revalidated_on_current_cohort_and_failures_preserve_active_model(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()):
                first = train(config, 60)
                config = self.fixture(root, 68)
                second = train(config, 67)
            self.assertEqual(second["parent_version"], first["selected_version"])
            self.assertEqual(second["incumbent"]["validation_rows"], 44)
            saved = pd.read_parquet(Path(config["artifacts_dir"]) / "runs" / second["run_id"] / "incumbent_validation.parquet")
            self.assertEqual(len(saved), 44)
            self.assertTrue(saved.snapshot_date.eq(str(date_for_day(date(2025, 10, 1), 53))).all())
            active = Path(config["state_dir"]) / "active_model.json"
            before = active.read_bytes()
            with patch("sklearn.linear_model.LogisticRegression.fit", side_effect=RuntimeError("failed fit")), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "failed fit"):
                    train(config, 67)
            self.assertEqual(active.read_bytes(), before)
            self.assertFalse((Path(config["state_dir"]) / "ingest.lock").exists())
            statuses = [json.loads(path.read_text())["status"] for path in (Path(config["artifacts_dir"]) / "runs").glob("*/run.json")]
            self.assertIn("failed", statuses)

    def test_unknown_labels_and_forbidden_input_columns_rejected(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            config = self.fixture(Path(temporary))
            state = json.loads((Path(config["state_dir"]) / "snapshots.json").read_text())
            first_date = str(date_for_day(date(2025, 10, 1), 14))
            folder = Path(state["output_dir"]) / f"snapshot_date={first_date}"
            features = pd.read_parquet(folder / "features")
            features.assign(retention_contact=True).to_parquet(folder / "features/part.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "Unexpected feature schema"):
                load_training_data(config, 60)
            features.to_parquet(folder / "features/part.parquet", index=False)
            labels = pd.read_parquet(folder / "labels")
            labels.loc[0, "target"] = None
            labels.to_parquet(folder / "labels/part.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "known target classes"):
                load_training_data(config, 60)


if __name__ == "__main__":
    unittest.main()
