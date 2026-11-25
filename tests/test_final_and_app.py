import contextlib
import importlib.util
import io
import json
import os
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import patch

from src.final import evaluate_final, freeze_predictions, read_json, verify_frozen
from src.ingest import atomic_json
from src.temporal import date_for_day, snapshot_windows
from src.train import train


DEPENDENCIES = all(importlib.util.find_spec(name) is not None for name in ("streamlit", "pandas", "sklearn", "xgboost"))


@unittest.skipUnless(DEPENDENCIES, "UI and final-cohort tests require Docker dependencies")
class FinalAndAppTests(unittest.TestCase):
    def fixture(self, root):
        from test_models import ModelTests
        config = ModelTests().fixture(root)
        with contextlib.redirect_stdout(io.StringIO()):
            run = train(config, 60)
        path = root / "config.json"
        atomic_json(path, config)
        return config, path, run

    def test_frozen_predictions_immutable_and_final_evaluation_never_fits(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, _, run = self.fixture(root)
            config["training_replay_days"] = [60]
            state = Path(config["state_dir"])
            ingest = read_json(state / "ingest.json")
            snapshot_state = read_json(state / "snapshots.json")
            source = Path(snapshot_state["output_dir"])
            original = source / f"snapshot_date={run['split']['validation']['snapshot_date']}"
            features = pd.read_parquet(original / "features")
            windows = snapshot_windows(date(2025, 10, 1), 76, config)
            features["snapshot_date"] = windows["snapshot_date"]
            folder = source / f"snapshot_date={windows['snapshot_date']}"
            (folder / "features").mkdir(parents=True)
            (folder / "labels").mkdir()
            features.to_parquet(folder / "features/part.parquet", index=False)
            metadata = {**windows, "label_status": "pending", "missing_outcome_dates": [windows["outcome_end"]]}
            snapshot_state["snapshots"] = [metadata]
            ingest["through_day"] = 76
            atomic_json(state / "ingest.json", ingest)
            atomic_json(state / "snapshots.json", snapshot_state)
            original_read = pd.read_parquet

            def no_future_labels(path, *args, **kwargs):
                if str(path).endswith("labels"):
                    raise AssertionError("Freeze must not read outcomes")
                return original_read(path, *args, **kwargs)

            with patch("pandas.read_parquet", side_effect=no_future_labels), contextlib.redirect_stdout(io.StringIO()):
                frozen = freeze_predictions(config)
                self.assertEqual(freeze_predictions(config), frozen)
            frozen_dir = Path(config["artifacts_dir"]) / "final_predictions"
            self.assertEqual(verify_frozen(frozen_dir), frozen)
            with self.assertRaises(ValueError):
                evaluate_final(config)
            labels = features[["subscriber_key", "snapshot_date"]].copy()
            labels["target"] = [index % 2 for index in range(len(labels))]
            labels.to_parquet(folder / "labels/part.parquet", index=False)
            ingest.update(through_day=91, available_through=str(date_for_day(date(2025, 10, 1), 91)))
            metadata.update(label_status="mature", missing_outcome_dates=[])
            snapshot_state["snapshots"] = [metadata]
            atomic_json(state / "ingest.json", ingest)
            atomic_json(state / "snapshots.json", snapshot_state)
            before_active = (state / "active_model.json").read_bytes()
            with patch("sklearn.pipeline.Pipeline.fit", side_effect=AssertionError("Final evaluation must not refit")):
                result = evaluate_final(config)
            self.assertEqual(len(result["models"]), 2)
            self.assertEqual(result["rows"], len(features))
            self.assertEqual((state / "active_model.json").read_bytes(), before_active)
            self.assertEqual(verify_frozen(frozen_dir), frozen)
            file = frozen_dir / f"{next(iter(frozen['thresholds']))}.parquet"
            file.write_bytes(file.read_bytes() + b"modified")
            with self.assertRaisesRegex(ValueError, "changed"):
                verify_frozen(frozen_dir)

    def test_three_views_prediction_and_pending_outcome_state(self):
        from streamlit.testing.v1 import AppTest
        with tempfile.TemporaryDirectory() as temporary:
            config, path, run = self.fixture(Path(temporary))
            state = Path(config["state_dir"])
            ingest = read_json(state / "ingest.json")
            ingest.update(loaded_event_end=ingest["available_through"], complete_cdr_dates=[], complete_ticket_dates=[],
                          quality={source: {} for source in ("cdr", "tickets")})
            atomic_json(state / "ingest.json", ingest)
            snapshots = read_json(state / "snapshots.json")
            root = Path(snapshots["output_dir"])
            metadata = read_json(root / f"snapshot_date={run['split']['validation']['snapshot_date']}" / "manifest.json")
            metadata["output_dir"] = str(root / f"snapshot_date={metadata['snapshot_date']}")
            snapshots["snapshots"] = [metadata]
            atomic_json(state / "snapshots.json", snapshots)
            with patch.dict(os.environ, {"BDA_CONFIG": str(path)}):
                app = AppTest.from_file("src/app.py", default_timeout=20).run()
                self.assertFalse(app.exception)
                app.radio(key="section").set_value("История моделей").run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Исходы ожидаются" in element.value for element in app.info))
                app.radio(key="section").set_value("Прогноз абонента").run()
                app.button(key="predict").click().run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Наблюдаемый исход" in element.value for element in app.info))
                metadata.update(label_status="pending", missing_outcome_dates=[metadata["outcome_end"]])
                snapshots["snapshots"] = [metadata]
                atomic_json(state / "snapshots.json", snapshots)
                app.run()
                self.assertFalse(app.exception)
                self.assertTrue(any("Исход пока неизвестен" in element.value for element in app.info))


if __name__ == "__main__":
    unittest.main()
