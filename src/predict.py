"""Reload the saved preprocessing/model pipeline and predict from historical features."""

import argparse
import json
from datetime import date
from pathlib import Path

from .features import KEY_COLUMNS


def load_model(model_dir):
    import joblib
    model_dir = Path(model_dir)
    metadata = json.loads((model_dir / "metadata.json").read_text(encoding="utf-8"))
    return joblib.load(model_dir / "pipeline.joblib"), metadata


def predict_probabilities(pipeline, features, feature_names):
    import numpy as np
    from pandas.api.types import is_numeric_dtype
    if len(features.columns) != len(set(features.columns)) or set(features.columns) != set(feature_names):
        raise ValueError("Prediction input must contain exactly the saved feature allowlist")
    if any(not is_numeric_dtype(features[name]) for name in feature_names):
        raise ValueError("Prediction features must be numeric")
    ordered = features.loc[:, feature_names].astype(float)
    if np.isinf(ordered.to_numpy()).any():
        raise ValueError("Infinite feature values are not supported")
    return pipeline.predict_proba(ordered)[:, 1]


def main():
    import pandas as pd
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--snapshot-day", type=int, required=True)
    parser.add_argument("--subscriber", required=True)
    parser.add_argument("--version")
    args = parser.parse_args()
    base = args.config.resolve().parent
    config = json.loads(args.config.read_text(encoding="utf-8"))
    state_dir = base / config["state_dir"]
    snapshots = json.loads((state_dir / "snapshots.json").read_text(encoding="utf-8"))
    matches = [item for item in snapshots["snapshots"] if item["snapshot_day"] == args.snapshot_day]
    if len(matches) != 1:
        raise ValueError("Build the requested snapshot before prediction")
    snapshot = matches[0]
    root = Path(snapshots["output_dir"]).resolve()
    if not root.is_relative_to((base / config["snapshots_dir"]).resolve()):
        raise ValueError("Snapshot is outside the configured output directory")
    folder = root / f"snapshot_date={date.fromisoformat(snapshot['snapshot_date'])}"
    features = pd.read_parquet(folder / "features", filters=[("subscriber_key", "==", args.subscriber)])
    if len(features) != 1:
        raise ValueError("Subscriber must be eligible and unique at this snapshot")
    version = args.version or json.loads((state_dir / "active_model.json").read_text(encoding="utf-8"))["version"]
    models = (base / config["artifacts_dir"] / "models").resolve()
    model_dir = (models / version).resolve()
    if model_dir.parent != models:
        raise ValueError("Model must be inside the configured model directory")
    pipeline, metadata = load_model(model_dir)
    probability = float(predict_probabilities(pipeline, features.drop(columns=KEY_COLUMNS), metadata["feature_names"])[0])
    print(json.dumps({"subscriber_key": args.subscriber, "snapshot_date": snapshot["snapshot_date"],
        "model_version": version, "inactivity_probability": probability,
        "threshold": metadata["threshold"], "predicted_inactive": probability >= metadata["threshold"],
        "target_description": "14-day inactivity proxy for churn"}, indent=2))


if __name__ == "__main__":
    main()
