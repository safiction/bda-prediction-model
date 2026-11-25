"""HW_5_1/HW_5_2: two fitted pipelines, chronological validation, versioned artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path

from .evaluate import choose_model, evaluate_predictions, write_comparison
from .features import KEY_COLUMNS, validate_feature_columns
from .ingest import atomic_json
from .predict import load_model, predict_probabilities
from .temporal import date_for_day, temporal_split_manifest


def create_pipelines(config):
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from xgboost import XGBClassifier
    settings = config["modeling"]
    # HW_5: медианная импутация; масштабирование нужно логистической регрессии.
    logistic = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("scaler", StandardScaler()),
                         ("model", LogisticRegression(random_state=settings["seed"], **settings["logistic_regression"]))])
    xgboost = Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                        ("model", XGBClassifier(random_state=settings["seed"], **settings["xgboost"]))])
    return {"logistic": logistic, "xgboost": xgboost}


def read_snapshot(root, windows, config, fingerprint, available):
    import numpy as np
    import pandas as pd
    folder = root / f"snapshot_date={windows['snapshot_date']}"
    if not folder.resolve().is_relative_to(root):
        raise ValueError("Snapshot path escapes the configured directory")
    metadata = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if any(metadata[name] != value for name, value in windows.items()):
        raise ValueError("Snapshot boundaries disagree with the configured temporal split")
    if (metadata["input_fingerprint"] != fingerprint or metadata["available_through"] != available
            or metadata["label_status"] != "mature" or metadata["missing_outcome_dates"]
            or metadata["outcome_end"] > available):
        raise ValueError("Training requires mature snapshots from the current ingest batch")
    if metadata["feature_names"] != config["feature_allowlist"]:
        raise ValueError("Snapshot feature order disagrees with configuration")
    features, labels = pd.read_parquet(folder / "features"), pd.read_parquet(folder / "labels")
    validate_feature_columns(features.columns.tolist(), config)
    if set(labels.columns) != set(KEY_COLUMNS + ["target"]):
        raise ValueError("Unexpected label columns")
    for frame in (features, labels):
        if frame[KEY_COLUMNS].isna().any().any() or frame.duplicated(KEY_COLUMNS).any():
            raise ValueError("Example keys must be usable and unique")
        frame["snapshot_date"] = frame["snapshot_date"].astype(str)
        if not frame["snapshot_date"].eq(windows["snapshot_date"]).all():
            raise ValueError("Rows belong to a different snapshot")
    data = features.merge(labels, on=KEY_COLUMNS, how="outer", validate="one_to_one", indicator=True)
    if not data["_merge"].eq("both").all() or len(data) != metadata["feature_rows"]:
        raise ValueError("Feature and label cohorts disagree")
    if data["target"].isna().any() or set(data["target"]) != {0, 1}:
        raise ValueError("Every training/validation snapshot requires both known target classes")
    counts = {str(int(key)): int(value) for key, value in data["target"].value_counts().items()}
    if counts != metadata["class_counts"]:
        raise ValueError("Saved class counts disagree with labels")
    for name in config["feature_allowlist"]:
        if not pd.api.types.is_numeric_dtype(data[name]) or np.isinf(data[name].to_numpy()).any():
            raise ValueError(f"Feature is not finite numeric data (missing values are allowed): {name}")
    return data.drop(columns="_merge"), metadata


def load_training_data(config, milestone):
    import pandas as pd
    state_dir = Path(config["state_dir"])
    ingest = json.loads((state_dir / "ingest.json").read_text(encoding="utf-8"))
    snapshots = json.loads((state_dir / "snapshots.json").read_text(encoding="utf-8"))
    if milestone not in config["training_replay_days"]:
        raise ValueError("Training is allowed only at the configured training milestones")
    expected_day = milestone + config["label_delivery_lag_days"]
    if expected_day >= config["final_snapshot_day"]:
        raise ValueError("Training must finish before the reserved final prediction snapshot")
    if ingest["through_day"] != expected_day:
        raise ValueError(f"Milestone {milestone} requires delivery clock day {expected_day}; later replay data cannot be used")
    if (snapshots["input_fingerprint"] != ingest["fingerprint"] or
            snapshots["available_through"] != ingest["available_through"]):
        raise ValueError("Rebuild snapshots from the current ingest batch before training")
    batch = Path(ingest["output_dir"]).resolve()
    root = Path(snapshots["output_dir"]).resolve()
    if not batch.is_relative_to(Path(config["processed_dir"]).resolve()) or not root.is_relative_to(Path(config["snapshots_dir"]).resolve()):
        raise ValueError("Only configured processed data and snapshot paths may be read")
    source = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
    if {item["source"] for item in source["files"]} != {"cdr", "tickets"}:
        raise ValueError("Only CDR and tickets may be modeling sources")
    start = min(date.fromisoformat(item["partition"]) for item in source["files"])
    if ingest["available_through"] != str(date_for_day(start, expected_day)):
        raise ValueError("Replay day and availability date disagree")
    split = next(run for run in temporal_split_manifest(start, config)["runs"] if run["milestone_day"] == milestone)
    train_rows = sum(json.loads((root / f"snapshot_date={row['snapshot_date']}" / "manifest.json").read_text())["feature_rows"]
                     for row in split["training"])
    validation_rows = json.loads((root / f"snapshot_date={split['validation']['snapshot_date']}" / "manifest.json").read_text())["feature_rows"]
    if train_rows > config["modeling"]["max_training_rows"] or validation_rows > config["modeling"]["max_validation_rows"]:
        raise ValueError("Subscriber-level modeling tables exceed the configured row limit")
    training, manifests = [], []
    for windows in split["training"]:
        data, manifest = read_snapshot(root, windows, config, ingest["fingerprint"], ingest["available_through"])
        training.append(data)
        manifests.append(manifest)
    validation, manifest = read_snapshot(root, split["validation"], config, ingest["fingerprint"], ingest["available_through"])
    manifests.append(manifest)
    return pd.concat(training, ignore_index=True), validation, {
        "split": split, "snapshot_manifests": manifests, "input_manifest": source,
        "input_fingerprint": ingest["fingerprint"], "available_through": ingest["available_through"]}


def save_validation(path, cohort, probabilities, version, threshold):
    predictions = cohort[KEY_COLUMNS + ["target"]].copy()
    predictions["probability"] = probabilities
    predictions["threshold"] = threshold
    predictions["model_version"] = version
    predictions.to_parquet(path, index=False)


def train(config, milestone):
    import joblib
    import numpy as np
    state_dir, artifacts = Path(config["state_dir"]), Path(config["artifacts_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    with (state_dir / "ingest.lock").open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    run = None
    try:
        run_id = f"m{milestone}_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}_{uuid.uuid4().hex[:6]}"
        run_dir, models = artifacts / "runs" / run_id, artifacts / "models"
        run_dir.mkdir(parents=True)
        models.mkdir(parents=True, exist_ok=True)
        active_path = state_dir / "active_model.json"
        parent = json.loads(active_path.read_text())["version"] if active_path.exists() else None
        run = {"run_id": run_id, "milestone": milestone, "status": "running", "stage": "load",
               "parent_version": parent, "candidates": {}, "incumbent": None, "config": config}
        atomic_json(run_dir / "run.json", run)
        started = time.perf_counter()
        print(f"Load chronological train/validation tables for milestone {milestone}", flush=True)
        training, validation, inputs = load_training_data(config, milestone)
        run.update(inputs)
        run["rows"] = {"training": len(training), "validation": len(validation)}
        run["class_counts"] = {name: {str(int(key)): int(value) for key, value in table.target.value_counts().items()}
                               for name, table in (("training", training), ("validation", validation))}
        run["code_sha256"] = hashlib.sha256(b"".join(path.read_bytes() for path in sorted(Path(__file__).parent.glob("*.py")))).hexdigest()
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True) if shutil.which("git") else None
        run["code_commit"] = commit.stdout.strip() if commit and commit.returncode == 0 else None
        packages = ("numpy", "pandas", "scipy", "scikit-learn", "xgboost-cpu", "joblib", "pyarrow", "pyspark")
        run["package_versions"] = {name: importlib.metadata.version(name) for name in packages}
        run["python_version"] = platform.python_version()
        names, threshold = config["feature_allowlist"], config["modeling"]["threshold"]
        x_train, x_validation = training[names].astype(float), validation[names].astype(float)
        y_train, y_validation = training.target.astype(int), validation.target.astype(int)
        with tempfile.TemporaryDirectory(prefix=".train-", dir=models) as temporary:
            work = Path(temporary)
            for name, pipeline in create_pipelines(config).items():
                version = f"{run_id}_{name}"
                folder = work / version
                folder.mkdir()
                run["stage"] = f"train_{name}"
                atomic_json(run_dir / "run.json", run)
                print(f"Fit {name}: {len(training)} rows, validate on {len(validation)} rows", flush=True)
                fit_started = time.perf_counter()
                pipeline.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - fit_started
                probability = predict_probabilities(pipeline, x_validation, names)
                metrics = evaluate_predictions(y_validation, probability, threshold)
                metadata = {"model_version": version, "run_id": run_id, "model_name": name,
                    "parent_version": parent, "milestone": milestone, "threshold": threshold,
                    "feature_names": names, "feature_types": inputs["snapshot_manifests"][0]["feature_types"],
                    "split": inputs["split"], "input_fingerprint": inputs["input_fingerprint"],
                    "metrics": metrics, "fit_seconds": fit_seconds, "rows": run["rows"],
                    "class_counts": run["class_counts"], "seed": config["modeling"]["seed"],
                    "package_versions": run["package_versions"], "code_sha256": run["code_sha256"],
                    "target_description": "14-day inactivity proxy for churn"}
                joblib.dump(pipeline, folder / "pipeline.joblib")
                atomic_json(folder / "metadata.json", metadata)
                restored, _ = load_model(folder)
                np.testing.assert_allclose(predict_probabilities(restored, x_validation, names), probability, rtol=1e-10, atol=1e-12)
                save_validation(folder / "validation.parquet", validation, probability, version, threshold)
                run["candidates"][version] = metadata
                print(f"{name}: PR-AUC={metrics['pr_auc']:.6f}, log loss={metrics['log_loss']:.6f}", flush=True)
            if parent is not None:
                parent_dir = (models / parent).resolve()
                if not parent_dir.is_relative_to(models.resolve()):
                    raise ValueError("Incumbent must be in the configured model directory")
                incumbent, metadata = load_model(parent_dir)
                if metadata["milestone"] > milestone:
                    raise ValueError("Incumbent was fitted at a later historical milestone")
                if metadata["feature_names"] != names:
                    raise ValueError("Incumbent feature schema differs; it cannot be compared on this cohort")
                probability = predict_probabilities(incumbent, x_validation, names)
                run["incumbent"] = {"metrics": evaluate_predictions(y_validation, probability, metadata["threshold"]),
                                    "validation_rows": len(validation), "threshold": metadata["threshold"]}
                save_validation(run_dir / "incumbent_validation.parquet", validation, probability, parent, metadata["threshold"])
            selected, reason = choose_model({version: info["metrics"] for version, info in run["candidates"].items()},
                                             parent, run["incumbent"]["metrics"] if parent else None, config)
            run.update(selected_version=selected, selection_reason=reason, status="completed", stage="completed",
                       duration_seconds=time.perf_counter() - started)
            write_comparison(run_dir / "comparison.md", run)
            for version in run["candidates"]:
                (work / version).replace(models / version)
            atomic_json(run_dir / "run.json", run)
            # Активный указатель меняем последним, только после успешной проверки всех артефактов.
        if selected != parent:
            atomic_json(active_path, {"version": selected, "run_id": run_id})
        return run
    except Exception as error:
        if run is not None:
            run.update(status="failed", error=f"{type(error).__name__}: {error}")
            atomic_json(run_dir / "run.json", run)
        raise
    finally:
        (state_dir / "ingest.lock").unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--milestone", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("data_dir", "audit_dir", "processed_dir", "snapshots_dir", "state_dir", "artifacts_dir"):
        config[name] = str((args.config.resolve().parent / config[name]).resolve())
    result = train(config, args.milestone)
    print(f"Completed run {result['run_id']}; active model: {result['selected_version']}")


if __name__ == "__main__":
    main()
