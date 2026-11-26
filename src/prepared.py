"""Reuse a previously prepared replay cutoff without cleaning CSV or starting Spark."""

import hashlib
import json
from datetime import date
from pathlib import Path

from .data import discover_files
from .ingest import atomic_json
from .temporal import date_for_day, replay_dates, snapshot_windows


# Настройки моделей и каталоги результатов не меняют подготовленные признаки.
DATA_SETTINGS = ("expected_days", "feature_window_days", "outcome_window_days",
                 "label_delivery_lag_days", "malformed_csv_policy", "snapshot_stride_days",
                 "encodings", "timestamp_assumption", "allowed_sources", "excluded_inputs", "feature_allowlist")


def reuse_prepared(config, through_day, days):
    if not days:
        return False
    processed, snapshots = Path(config["processed_dir"]), Path(config["snapshots_dir"])
    for batch in sorted(processed.glob(f"day={through_day:03}_*"), key=lambda path: path.stat().st_mtime, reverse=True):
        if not (batch / "quality.json").exists() or not (batch / "manifest.json").exists():
            continue
        quality = json.loads((batch / "quality.json").read_text(encoding="utf-8"))
        source = json.loads((batch / "manifest.json").read_text(encoding="utf-8"))
        if (any(source["config"].get(key) != config.get(key) for key in DATA_SETTINGS)
                or Path(source["config"]["data_dir"]).resolve() != Path(config["data_dir"]).resolve()):
            continue
        if not source["files"] or {item["source"] for item in source["files"]} != {"cdr", "tickets"}:
            continue
        start = min(date.fromisoformat(item["partition"]) for item in source["files"])
        event_end, available = replay_dates(start, through_day, config)
        if (quality["through_day"] != through_day or quality["available_through"] != str(available)
                or quality["loaded_event_end"] != str(event_end)):
            continue
        actual = {(str(day), kind, path.as_posix()) for day, kind, path in discover_files(config["data_dir"]) if day <= event_end}
        recorded = {(item["partition"], item["source"], item["path"]) for item in source["files"]}
        if actual != recorded or not all((batch / part).is_dir() for part in ("clean/cdr", "clean/tickets", "daily/cdr")):
            continue
        expected = {str(date_for_day(start, day)) for day in range(1, min(through_day, config["expected_days"]) + 1)}
        if any({item["partition"] for item in source["files"] if item["source"] == kind} != expected for kind in ("cdr", "tickets")):
            continue
        for root in sorted(snapshots.glob(f"{quality['fingerprint'][:12]}_*"), key=lambda path: path.stat().st_mtime, reverse=True):
            rows = []
            for day in days:
                folder = root / f"snapshot_date={date_for_day(start, day)}"
                manifest = folder / "manifest.json"
                if not manifest.exists() or not all((folder / name).is_dir() for name in ("features", "labels")):
                    break
                row = json.loads(manifest.read_text(encoding="utf-8"))
                if (row["input_fingerprint"] != quality["fingerprint"] or row["available_through"] != str(available)
                        or row["feature_names"] != config["feature_allowlist"]
                        or any(row.get(key) != value for key, value in snapshot_windows(start, day, config).items())):
                    break
                rows.append({**row, "output_dir": str(folder.resolve())})
            if len(rows) != len(days):
                continue
            # Проверяем байты исходников, но не разбираем и не очищаем CSV повторно.
            unchanged = True
            for item in source["files"]:
                path = Path(item["path"])
                if not path.is_file():
                    unchanged = False
                    break
                with path.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != item["sha256"]:
                        unchanged = False
                        break
            if not unchanged:
                break
            state = Path(config["state_dir"])
            restored = {**quality, "output_dir": str(batch.resolve())}
            atomic_json(state / "ingest.json", restored)
            atomic_json(state / "snapshots.json", {"input_fingerprint": quality["fingerprint"],
                        "output_dir": str(root.resolve()), "available_through": str(available), "snapshots": rows})
            return True
    return False
