"""Build historical feature/label Parquets from the latest successful ingest batch."""

import argparse
import hashlib
import json
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path

from .features import KEY_COLUMNS, build_features, validate_feature_columns
from .ingest import atomic_json, start_spark
from .labels import build_labels
from .temporal import date_for_day, snapshot_schedule, snapshot_windows, temporal_split_manifest


def build_snapshots(config, days=None, spark=None):
    state_dir = Path(config["state_dir"])
    lock_path = state_dir / "ingest.lock"
    if not (state_dir / "ingest.json").exists():
        raise ValueError("Run ingestion before building snapshots")
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    own_spark = spark is None
    try:
        state = json.loads((state_dir / "ingest.json").read_text(encoding="utf-8"))
        batch = Path(state["output_dir"]).resolve()
        if not batch.is_relative_to(Path(config["processed_dir"]).resolve()):
            raise ValueError("Ingest batch must be inside the configured processed directory")
        source_manifest = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        files = source_manifest["files"]
        if {item["source"] for item in files} != {"cdr", "tickets"}:
            raise ValueError("Only CDR and tickets are allowed")
        start = min(date.fromisoformat(item["partition"]) for item in files)
        available = date.fromisoformat(state["available_through"])
        loaded_end = date.fromisoformat(state["loaded_event_end"])
        schedule = snapshot_schedule(config)
        if days is None:
            days = [day for day in schedule if date_for_day(start, day) <= loaded_end]
        days = sorted(set(days))
        if not days:
            raise ValueError("No scheduled snapshot is available; ingest at least the first 14 days")
        for day in days:
            snapshot = date_for_day(start, day)
            if day < config["feature_window_days"] or snapshot > min(loaded_end, available):
                raise ValueError(f"Snapshot day {day} lacks a full historical feature window")
            required = {snapshot - timedelta(days=i) for i in range(config["feature_window_days"])}
            for source in ("cdr", "tickets"):
                partitions = {date.fromisoformat(item["partition"]) for item in files if item["source"] == source}
                if not required <= partitions:
                    raise ValueError(f"Snapshot day {day}: missing {source} feature partitions")
        code = b"".join(Path(__file__).with_name(name).read_bytes() for name in
                        ("snapshots.py", "features.py", "labels.py", "preprocess.py", "temporal.py"))
        version = hashlib.sha256(code + json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
        output = Path(config["snapshots_dir"]) / f"{state['fingerprint'][:12]}_{version}"
        output.mkdir(parents=True, exist_ok=True)
        if spark is None:
            spark = start_spark(config)
        from pyspark.sql.types import NumericType
        daily = spark.read.parquet(str(batch / "daily/cdr"))
        if "traffic_valid_count" not in daily.columns:
            raise ValueError("Daily aggregates use an older schema; rerun ingestion with the current code")
        tickets = spark.read.parquet(str(batch / "clean/tickets"))
        cdr = spark.read.parquet(str(batch / "clean/cdr"))
        complete = {date.fromisoformat(value) for value in state["complete_cdr_dates"]}
        results = []
        for day in days:
            snapshot = date_for_day(start, day)
            destination = output / f"snapshot_date={snapshot}"
            if destination.exists():
                results.append(json.loads((destination / "manifest.json").read_text(encoding="utf-8")))
                print(f"Unchanged snapshot: {snapshot}", flush=True)
                continue
            print(f"Build features: day {day} ({snapshot})", flush=True)
            features = build_features(daily, tickets, snapshot, config).cache()
            labels = None
            try:
                validate_feature_columns(features.columns, config)
                for field in features.schema:
                    if field.name not in KEY_COLUMNS and not isinstance(field.dataType, NumericType):
                        raise ValueError(f"Feature must be numeric: {field.name}")
                rows = features.count()
                if not rows or features.groupBy(*KEY_COLUMNS).count().filter("count > 1").limit(1).count():
                    raise ValueError("Feature cohort must be nonempty and its example key unique")
                print(f"Build labels: day {day}, eligible={rows}", flush=True)
                labels = build_labels(cdr, features, snapshot, available, complete,
                                      config["outcome_window_days"]).cache()
                counts = {"unknown" if row.target is None else str(row.target): row['count']
                          for row in labels.groupBy("target").count().collect()}
                if sum(counts.values()) != rows:
                    raise ValueError("Label rows do not match the eligible cohort")
                windows = snapshot_windows(start, day, config)
                needed = {snapshot + timedelta(days=i) for i in range(1, config["outcome_window_days"] + 1)}
                result = {**windows, "input_fingerprint": state["fingerprint"], "available_through": str(available),
                          "feature_rows": rows, "class_counts": counts,
                          "label_status": "pending" if counts.get("unknown", 0) else "mature",
                          "missing_outcome_dates": sorted(str(value) for value in needed - complete),
                          "feature_names": config["feature_allowlist"],
                          "feature_types": {field.name: field.dataType.simpleString() for field in features.schema},
                          "uses": [role for role, _ in schedule.get(day, [])],
                          "output_dir": str(destination.resolve())}
                with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=output) as temporary:
                    publish = Path(temporary) / "result"
                    features.write.parquet(str(publish / "features"))
                    labels.write.parquet(str(publish / "labels"))
                    atomic_json(publish / "manifest.json", result)
                    publish.replace(destination)
                results.append(result)
                print(f"Completed snapshot {day}: {result['label_status']}, {counts}", flush=True)
            finally:
                features.unpersist()
                if labels is not None:
                    labels.unpersist()
        atomic_json(output / "split_manifest.json", temporal_split_manifest(start, config))
        summary = {"input_fingerprint": state["fingerprint"], "output_dir": str(output.resolve()),
                   "available_through": str(available), "snapshots": results}
        atomic_json(state_dir / "snapshots.json", summary)
        return summary
    finally:
        try:
            if own_spark and spark is not None:
                spark.stop()
        finally:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--snapshot-day", type=int, nargs="+")
    selection.add_argument("--scheduled", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("state_dir", "processed_dir", "snapshots_dir"):
        config[name] = str((args.config.resolve().parent / config[name]).resolve())
    build_snapshots(config, args.snapshot_day)


if __name__ == "__main__":
    main()
