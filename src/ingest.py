"""Build a cutoff-limited clean Parquet batch: python -m src.ingest --through-day 60."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

from .data import discover_files, inspect_file, read_rows
from .preprocess import CDR_COLUMNS, TICKET_COLUMNS, PROVENANCE, clean_row, daily_cdr, deduplicate, read_staged
from .temporal import replay_dates


def start_spark(config):
    from pyspark.sql import SparkSession
    # Не используем случайный Python из PATH для дочерних процессов Spark.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    java = shutil.which("java")
    if java and not Path(os.environ.get("JAVA_HOME", ""), "bin").is_dir():
        os.environ["JAVA_HOME"] = str(Path(java).resolve().parent.parent)
    settings = config["spark"]
    spark = (SparkSession.builder.master(settings["master"]).appName("bda-ingest")
             .config("spark.driver.memory", settings["driver_memory"])
             .config("spark.sql.shuffle.partitions", settings["shuffle_partitions"])
             .config("spark.sql.session.timeZone", "UTC")
             .config("spark.ui.enabled", "false").getOrCreate())
    spark.sparkContext.setLogLevel("WARN")
    return spark


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def ingest(config, through_day, spark=None):
    if config["allowed_sources"] != ["cdr", "tickets"] or config["malformed_csv_policy"] != "drop":
        raise ValueError("This stage supports only CDR/tickets and the agreed malformed-row drop policy")
    files = discover_files(config["data_dir"])
    if not files:
        raise ValueError("No allowed source files found")
    start = min(partition for partition, _, _ in files)
    event_end, cutoff = replay_dates(start, through_day, config)
    files = [(partition, source, path) for partition, source, path in files if partition <= event_end]
    expected = {start + timedelta(days=i) for i in range((event_end - start).days + 1)}
    for source in ("cdr", "tickets"):
        missing = expected - {day for day, kind, _ in files if kind == source}
        if missing:
            raise ValueError(f"Missing {source} partitions: {sorted(missing)}")
    state_dir, processed = Path(config["state_dir"]), Path(config["processed_dir"])
    state_dir.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "ingest.json"
    lock_path = state_dir / "ingest.lock"
    # Один процесс меняет результаты; прежнее состояние сохраняется при ошибке.
    with lock_path.open("x", encoding="utf-8") as lock:
        lock.write(str(os.getpid()))
    own_spark = spark is None
    try:
        if state_path.exists():
            previous = json.loads(state_path.read_text(encoding="utf-8"))
            if through_day < previous["through_day"]:
                raise ValueError("Replay cannot move backwards; use a separate output/state directory for experiments")
        manifest = []
        for partition, source, path in files:
            encoding, sha256 = inspect_file(path, config["encodings"])
            manifest.append({"partition": str(partition), "source": source, "path": path.as_posix(),
                             "encoding": encoding, "sha256": sha256})
        # При изменении кода очистки повторный запуск тоже должен пересчитать результат.
        code_hash = hashlib.sha256(b"".join(Path(__file__).with_name(name).read_bytes()
                                     for name in ("ingest.py", "preprocess.py", "data.py", "temporal.py"))).hexdigest()
        fingerprint = hashlib.sha256(json.dumps({"files": manifest, "config": config,
                             "through_day": through_day, "code": code_hash}, sort_keys=True).encode()).hexdigest()
        destination = processed / f"day={through_day:03}_{fingerprint[:12]}"
        if destination.exists():
            result = json.loads((destination / "quality.json").read_text(encoding="utf-8"))
            atomic_json(state_path, result)
            print(f"Unchanged batch: {destination}", flush=True)
            return result
        quality = {}
        with tempfile.TemporaryDirectory(prefix=".ingest-", dir=processed) as temporary:
            work = Path(temporary)
            publish = work / "result"
            publish.mkdir()
            for source in ("cdr", "tickets"):
                stats = Counter()
                staged = work / f"{source}.csv"
                columns = (CDR_COLUMNS if source == "cdr" else TICKET_COLUMNS) + PROVENANCE
                with staged.open("w", encoding="utf-8", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=columns)
                    writer.writeheader()
                    for item in manifest:
                        if item["source"] != source:
                            continue
                        path = Path(item["path"])
                        print(f"Clean {source}: {item['partition']}", flush=True)
                        for number, row in enumerate(read_rows(path, item["encoding"], source), 1):
                            clean = clean_row(row, source, date.fromisoformat(item["partition"]), path, number, cutoff, stats)
                            if clean is not None:
                                writer.writerow(clean)
                if spark is None:
                    spark = start_spark(config)
                from pyspark import StorageLevel
                from pyspark.sql import functions as F
                print(f"Stage: clean {source} — deduplicate and check quality", flush=True)
                # HW_3: большой пакет не помещается в heap; переиспользуем его с диска.
                frame = read_staged(spark, staged, source).persist(StorageLevel.DISK_ONLY)
                clean = deduplicate(frame, source).persist(StorageLevel.DISK_ONLY)
                try:
                    stats["output_rows"] = clean.count()
                    identifier = "record_id" if source == "cdr" else "ticket_id"
                    stats["unique_ids"] = clean.select(identifier).distinct().count()
                    stats["duplicate_rows_removed"] = stats["accepted_rows"] - stats["output_rows"]
                    stats["conflicting_ids"] = (frame.groupBy(identifier).agg(
                        F.countDistinct("row_hash").alias("versions")
                    ).filter("versions > 1").count())
                    print(f"Stage: save {source} — write clean Parquet", flush=True)
                    clean.write.partitionBy("event_date").parquet(str(publish / "clean" / source))
                    if source == "cdr":
                        print("Stage: aggregate cdr — write daily aggregates", flush=True)
                        daily_cdr(clean).write.partitionBy("event_date").parquet(str(publish / "daily" / "cdr"))
                finally:
                    clean.unpersist()
                    frame.unpersist()
                quality[source] = dict(stats)
            pending = {key.removeprefix("pending_date:") for key in quality["cdr"] if key.startswith("pending_date:")}
            result = {"through_day": through_day, "available_through": str(cutoff),
                      "loaded_event_end": str(event_end), "output_dir": str(destination.resolve()),
                      "fingerprint": fingerprint, "quality": quality,
                      "complete_cdr_dates": sorted(str(day) for day in expected if str(day) not in pending),
                      "complete_ticket_dates": sorted(str(day) for day in expected)}
            atomic_json(publish / "quality.json", result)
            atomic_json(publish / "manifest.json", {"files": manifest, "config": config, "code_sha256": code_hash})
            publish.replace(destination)
            atomic_json(state_path, result)
            print(f"Completed: {destination}", flush=True)
            return result
    finally:
        try:
            if own_spark and spark is not None:
                spark.stop()
        finally:
            lock_path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--through-day", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for name in ("data_dir", "processed_dir", "state_dir"):
        config[name] = str((args.config.resolve().parent / config[name]).resolve())
    ingest(config, args.through_day)


if __name__ == "__main__":
    main()
