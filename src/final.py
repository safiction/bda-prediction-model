"""Freeze day-76 predictions and evaluate them later, without refitting or selection."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from .evaluate import evaluate_predictions
from .features import KEY_COLUMNS, validate_feature_columns
from .ingest import atomic_json
from .predict import load_model, predict_probabilities


def read_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def completed_models(config):
    versions = {}
    for path in sorted((Path(config["artifacts_dir"]) / "runs").glob("*/run.json")):
        run = read_json(path)
        if run["status"] == "completed":
            for version, metadata in run["candidates"].items():
                versions[version] = metadata
    return versions


def snapshot_details(config, day):
    snapshots = read_json(Path(config["state_dir"]) / "snapshots.json", {})
    ingest = read_json(Path(config["state_dir"]) / "ingest.json", {})
    if not snapshots or snapshots["input_fingerprint"] != ingest.get("fingerprint"):
        raise ValueError("Сначала подготовьте снимки текущей партии")
    root = Path(snapshots["output_dir"]).resolve()
    if not root.is_relative_to(Path(config["snapshots_dir"]).resolve()):
        raise ValueError("Snapshot must be inside the configured directory")
    matches = [item for item in snapshots["snapshots"] if item["snapshot_day"] == day]
    if len(matches) != 1:
        raise ValueError(f"Снимок дня {day} ещё не подготовлен")
    metadata = matches[0]
    return root / f"snapshot_date={metadata['snapshot_date']}", metadata


def verify_frozen(folder):
    folder = Path(folder)
    manifest = read_json(folder / "manifest.json")
    if manifest is None:
        raise ValueError("Frozen predictions are unavailable")
    for name, checksum in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != checksum:
            raise ValueError(f"Frozen file changed: {name}")
    return manifest


def freeze_predictions(config):
    import pandas as pd
    state = Path(config["state_dir"])
    ingest = read_json(state / "ingest.json")
    if ingest["through_day"] != config["final_snapshot_day"]:
        raise ValueError("Predictions must be frozen at day 76 before final outcomes arrive")
    destination = Path(config["artifacts_dir"]) / "final_predictions"
    if destination.exists():
        return verify_frozen(destination)
    folder, snapshot = snapshot_details(config, config["final_snapshot_day"])
    features = pd.read_parquet(folder / "features")
    validate_feature_columns(features.columns.tolist(), config)
    if features.empty or features.duplicated(KEY_COLUMNS).any():
        raise ValueError("Final cohort must be nonempty and unique")
    versions = completed_models(config)
    if not set(config["training_replay_days"]) <= {item["milestone"] for item in versions.values()}:
        raise ValueError("Complete all scheduled training milestones before freezing")
    active = read_json(state / "active_model.json")["version"]
    if active not in versions or not versions:
        raise ValueError("A completed active model is required")
    artifacts = Path(config["artifacts_dir"])
    artifacts.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".freeze-", dir=artifacts) as temporary:
        publish = Path(temporary) / "result"
        publish.mkdir()
        features.to_parquet(publish / "features.parquet", index=False)
        thresholds = {}
        for version in versions:
            pipeline, metadata = load_model(artifacts / "models" / version)
            probability = predict_probabilities(pipeline, features.drop(columns=KEY_COLUMNS), metadata["feature_names"])
            predictions = features[KEY_COLUMNS].copy()
            predictions["probability"], predictions["threshold"], predictions["model_version"] = probability, metadata["threshold"], version
            predictions.to_parquet(publish / f"{version}.parquet", index=False)
            thresholds[version] = metadata["threshold"]
            print(f"Frozen {version}: {len(features)} subscribers", flush=True)
        manifest = {"snapshot_date": snapshot["snapshot_date"], "snapshot_day": config["final_snapshot_day"],
                    "outcome_start": snapshot["outcome_start"], "outcome_end": snapshot["outcome_end"],
                    "input_fingerprint": ingest["fingerprint"], "rows": len(features),
                    "active_version": active, "thresholds": thresholds,
                    "files": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in publish.glob("*.parquet")}}
        atomic_json(publish / "manifest.json", manifest)
        publish.replace(destination)
    return manifest


def evaluate_final(config):
    import pandas as pd
    state = Path(config["state_dir"])
    ingest = read_json(state / "ingest.json")
    if ingest["through_day"] != config["final_replay_day"] + config["label_delivery_lag_days"]:
        raise ValueError("Final evaluation requires the complete final outcome window and delivery allowance")
    artifacts = Path(config["artifacts_dir"])
    frozen = verify_frozen(artifacts / "final_predictions")
    if read_json(state / "active_model.json")["version"] != frozen["active_version"]:
        raise ValueError("Active model changed after the final predictions were frozen")
    destination = artifacts / "final_evaluation"
    if destination.exists():
        return read_json(destination / "comparison.json")
    folder, snapshot = snapshot_details(config, config["final_snapshot_day"])
    if (snapshot["label_status"] != "mature" or snapshot["missing_outcome_dates"] or
            snapshot["outcome_end"] > ingest["available_through"] or snapshot["outcome_end"] != frozen["outcome_end"]):
        raise ValueError("Final labels are still pending or have different boundaries")
    labels = pd.read_parquet(folder / "labels")
    if set(labels.columns) != set(KEY_COLUMNS + ["target"]) or set(labels.target) != {0, 1} or labels.duplicated(KEY_COLUMNS).any():
        raise ValueError("Final labels must be unique, known, and contain both classes")
    labels["snapshot_date"] = labels.snapshot_date.astype(str)
    results = []
    for version, threshold in frozen["thresholds"].items():
        predictions = pd.read_parquet(artifacts / "final_predictions" / f"{version}.parquet")
        predictions["snapshot_date"] = predictions.snapshot_date.astype(str)
        joined = predictions.merge(labels, on=KEY_COLUMNS, how="outer", validate="one_to_one", indicator=True)
        if len(joined) != frozen["rows"] or not joined["_merge"].eq("both").all():
            raise ValueError("Final label cohort differs from the frozen cohort")
        results.append({"version": version, "active_at_freeze": version == frozen["active_version"],
                        "threshold": threshold, **evaluate_predictions(joined.target, joined.probability, threshold)})
    result = {"snapshot_date": frozen["snapshot_date"], "outcome_start": frozen["outcome_start"],
              "outcome_end": frozen["outcome_end"], "available_through": ingest["available_through"],
              "rows": frozen["rows"], "class_counts": {str(value): int(count) for value, count in labels.target.value_counts().sort_index().items()},
              "active_version": frozen["active_version"], "models": results}
    with tempfile.TemporaryDirectory(prefix=".final-", dir=artifacts) as temporary:
        publish = Path(temporary) / "result"
        publish.mkdir()
        labels.to_parquet(publish / "outcomes.parquet", index=False)
        atomic_json(publish / "comparison.json", result)
        lines = ["# Shared final cohort", "", "Frozen day-76 probabilities; outcomes 77–90. No refitting or model selection.", "",
                 f"Snapshot: {frozen['snapshot_date']}; outcomes: {frozen['outcome_start']} to {frozen['outcome_end']}.",
                 f"Subscribers: {frozen['rows']:,}; class counts: {result['class_counts']} (1 = inactive).",
                 f"Active version selected before freezing: `{frozen['active_version']}`.", "",
                 "PR-AUC means average precision. All thresholds remain 0.5.", "",
                 "| Version | PR-AUC | Log loss | ROC-AUC | Precision | Recall | F1 |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for row in results:
            values = " | ".join(f"{row[name]:.6f}" for name in ("pr_auc", "log_loss", "roc_auc", "precision", "recall", "f1"))
            lines.append(f"| {row['version']} | {values} |")
        lines += ["", "| Version | TN | FP | FN | TP |", "| --- | ---: | ---: | ---: | ---: |"]
        for row in results:
            counts = " | ".join(str(row["confusion_matrix"][name]) for name in ("tn", "fp", "fn", "tp"))
            lines.append(f"| {row['version']} | {counts} |")
        lines += ["", "## Limitations", "",
                  "The target is 14-day CDR inactivity, an indirect churn proxy rather than confirmed cancellation.",
                  "Rolling validation dates and repeated subscribers prevent interpreting the history as steady improvement or unseen-subscriber performance.",
                  "A fixed threshold can give very low recall for XGBoost; this final cohort is not used to tune it.",
                  "The result describes one cohort and one outcome window in this dataset."]
        (publish / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        publish.replace(destination)
    return result


def main():
    from .worker import load_config
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "evaluate"])
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    config = load_config(args.config)
    lock = Path(config["state_dir"]) / "ingest.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        result = freeze_predictions(config) if args.action == "freeze" else evaluate_final(config)
        print(json.dumps(result, ensure_ascii=False))
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
