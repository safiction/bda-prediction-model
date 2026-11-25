"""Milestone 1: source inventory, calendar, and inactivity-target feasibility.

Run from the repository root: python -m src.audit
Only aggregate diagnostics are saved; this does not advance replay or train models.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import sqlite3
import sys
import tempfile
from collections import Counter
from contextlib import closing
from datetime import timedelta
from pathlib import Path

from .data import discover_files, inspect_file, parse_timestamp, read_rows
from .temporal import date_for_day, outcome_is_complete, snapshot_schedule


KEY_PATTERN = re.compile(r"SUB[0-9]{9}")


def create_database(connection):
    # SQLite нужен только для аудита: все идентификаторы не храним в памяти.
    connection.execute("PRAGMA cache_size = -65536")  # Кеш SQLite: не более 64 МБ.
    connection.executescript("""
        CREATE TABLE cdr (
            id TEXT PRIMARY KEY, signature BLOB, subscriber TEXT,
            event_day TEXT, available_day TEXT
        );
        CREATE TABLE tickets (id TEXT PRIMARY KEY, signature BLOB);
    """)


def scan_file(connection, partition, source, path, config):
    encoding, fingerprint = inspect_file(path, config["encodings"])
    stats = Counter()
    columns = None
    timestamp_column = "event_ts" if source == "cdr" else "created_ts"
    id_column = "record_id" if source == "cdr" else "ticket_id"
    min_time, max_time, max_available = None, None, None
    for row in read_rows(path, encoding, source):
        stats["rows"] += 1
        if None in row or any(value is None for value in row.values()):
            stats["malformed_csv_rows"] += 1
            continue
        columns = list(row)
        event = parse_timestamp(row[timestamp_column])
        key = row["subscriber_key"].strip()
        record_id = row[id_column].strip()
        valid_key = bool(KEY_PATTERN.fullmatch(key))
        stats["invalid_key"] += not valid_key
        stats["invalid_timestamp"] += event is None
        stats["missing_id"] += not record_id
        if event:
            min_time = min(min_time or event, event)
            max_time = max(max_time or event, event)
            stats["partition_date_mismatch"] += event.date() != partition
        ingest = parse_timestamp(row["ingest_ts"]) if source == "cdr" else None
        if source == "cdr":
            stats["invalid_ingest_timestamp"] += ingest is None
            stats["ingest_before_event"] += bool(ingest and event and ingest < event)
            stats["ingest_after_partition_day"] += bool(ingest and ingest.date() > partition)
            for name in ("duration_sec", "bytes_up", "bytes_down", "charge"):
                try:
                    number = float(row[name])
                    bad = not math.isfinite(number) or number < 0
                except ValueError:
                    bad = True
                stats[f"invalid_{name}"] += bad
        else:
            closed = parse_timestamp(row["closed_ts"])
            stats["invalid_closed_timestamp"] += bool(row["closed_ts"].strip() and not closed)
            stats["closure_before_creation"] += bool(closed and event and closed < event)
        if not (event and valid_key and record_id):
            continue
        # Невалидный ingest_ts не заменяем выдуманной датой доступности.
        if source == "cdr" and (not ingest or ingest < event):
            stats["availability_unusable"] += 1
            continue
        signature = hashlib.blake2b(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode(), digest_size=16
        ).digest()
        if source == "cdr":
            available = max(partition, event.date(), ingest.date()).isoformat()
            max_available = max(max_available or available, available)
            inserted = connection.execute(
                "INSERT OR IGNORE INTO cdr VALUES (?, ?, ?, ?, ?)",
                (record_id, signature, key, event.date().isoformat(), available),
            ).rowcount
        else:
            inserted = connection.execute(
                "INSERT OR IGNORE INTO tickets VALUES (?, ?)", (record_id, signature)
            ).rowcount
        if not inserted:
            stats["duplicate_id_rows"] += 1
            previous = connection.execute(
                f"SELECT signature FROM {source} WHERE id = ?", (record_id,)
            ).fetchone()[0]
            stats["conflicting_duplicate_rows"] += previous != signature
            if source == "cdr":
                # Самая ранняя доступная версия; при равенстве — порядок файлов/строк.
                connection.execute(
                    "UPDATE cdr SET signature=?, subscriber=?, event_day=?, available_day=? "
                    "WHERE id=? AND available_day > ?",
                    (signature, key, event.date().isoformat(), available, record_id, available),
                )
        if stats["rows"] % 10000 == 0:
            connection.commit()
    connection.commit()
    return {
        "path": path.as_posix(), "source": source, "partition_date": partition.isoformat(),
        "bytes": path.stat().st_size, "encoding": encoding, "sha256": fingerprint,
        "columns": columns, "min_timestamp": str(min_time) if min_time else None,
        "max_timestamp": str(max_time) if max_time else None,
        "max_available_date": max_available, "quality": dict(stats),
    }


def measure_snapshot(connection, start, day, config, complete_dates):
    snapshot = date_for_day(start, day)
    feature_start = snapshot - timedelta(days=config["feature_window_days"] - 1)
    outcome_end = snapshot + timedelta(days=config["outcome_window_days"])
    # Обе границы полуоткрытые. Когорта использует только доступные к D записи.
    eligible = {row[0] for row in connection.execute(
        "SELECT DISTINCT subscriber FROM cdr WHERE event_day >= ? AND event_day < ? "
        "AND available_day <= ?",
        (feature_start.isoformat(), (snapshot + timedelta(days=1)).isoformat(), snapshot.isoformat()),
    )}
    active = {row[0] for row in connection.execute(
        "SELECT DISTINCT subscriber FROM cdr WHERE event_day >= ? AND event_day < ?",
        ((snapshot + timedelta(days=1)).isoformat(), (outcome_end + timedelta(days=1)).isoformat()),
    )}
    ready_on = connection.execute(
        "SELECT MAX(available_day) FROM cdr WHERE event_day > ? AND event_day <= ?",
        (snapshot.isoformat(), outcome_end.isoformat()),
    ).fetchone()[0]
    coverage = outcome_is_complete(snapshot, config["outcome_window_days"], outcome_end, complete_dates)
    feature_coverage = all(feature_start + timedelta(days=i) in complete_dates
                           for i in range(config["feature_window_days"]))
    positive = len(eligible - active) if coverage and feature_coverage else None
    negative = len(eligible & active) if coverage and feature_coverage else None
    mature = coverage and ready_on is not None and ready_on <= outcome_end.isoformat()
    return {
        "snapshot_day": day, "snapshot_date": snapshot.isoformat(),
        "feature_start": feature_start.isoformat(), "outcome_start": (snapshot + timedelta(days=1)).isoformat(),
        "outcome_end": outcome_end.isoformat(), "eligible": len(eligible),
        "feature_coverage_complete": feature_coverage, "outcome_partitions_complete": coverage,
        "diagnostic_inactive": positive, "diagnostic_active": negative,
        "diagnostic_inactive_share": positive / len(eligible) if eligible and positive is not None else None,
        "outcome_records_available_on": ready_on,
        "labels_ready_at_horizon": mature,
        "unknown_at_horizon": 0 if mature else len(eligible),
    }


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    for name, content in {
        "audit.json": json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        "calendar.json": json.dumps(report["calendar"], indent=2) + "\n",
    }.items():
        temporary = output / (name + ".tmp")
        temporary.write_text(content, encoding="utf-8")
        temporary.replace(output / name)
    with (output / "snapshots.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(report["snapshots"][0]))
        writer.writeheader()
        writer.writerows(report["snapshots"])
    lines = ["# Allowed-data audit", "", f"Gate: **{report['gate']}**", "",
             f"Calendar: {report['calendar']['start']} through {report['calendar']['end']}.",
             "", "Only CDR and tickets were read. Raw files were unchanged.", ""]
    for source in ("cdr", "tickets"):
        items = [item for item in report["inventory"] if item["source"] == source]
        quality = Counter()
        for item in items:
            quality.update(item["quality"])
        size = sum(item["bytes"] for item in items) / 1024 ** 2
        encodings = dict(Counter(item["encoding"] for item in items))
        late = sum("/late_delivery/" in item["path"] for item in items)
        lines += [f"## {source}", "",
                  f"Files: {len(items)} ({late} late_delivery); size: {size:.1f} MiB; encodings: {encodings}.",
                  f"Raw rows: {quality['rows']:,}; unique usable IDs: {report['unique_valid_ids'][source]:,}.",
                  "", "Reported defects (counts can overlap):", ""]
        lines += [f"- {name}: {value:,}" for name, value in sorted(quality.items()) if value and name != "rows"]
        lines.append("")
    lines += ["## Target feasibility", "",
             "Diagnostic class counts use full observed outcome records, including records delivered later.",
             "They are feasibility statistics, not replay-ready training labels or final model scores.", "",
             "| Day | Eligible | Inactive (diagnostic) | Active (diagnostic) | Outcome records available on |",
             "| --- | ---: | ---: | ---: | --- |"]
    for row in report["snapshots"]:
        lines.append(f"| {row['snapshot_day']} | {row['eligible']} | {row['diagnostic_inactive']} | "
                     f"{row['diagnostic_active']} | {row['outcome_records_available_on']} |")
    lines += ["", "Issues:", ""] + [f"- {issue}" for issue in report["issues"]]
    if not report["issues"]:
        lines.append("- None.")
    lines += ["", "Assumptions:", "",
              "- Complete-day file exposure includes raw and late_delivery partitions together.",
              "- CDR ingest_ts is treated as record availability; missing or reversed ingest timestamps are flagged and excluded from the diagnostic cohort.",
              "- Eligibility uses only records available by the snapshot's end of day.",
              "- A horizon is conservatively pending if any valid outcome record is still unavailable.",
              "- First valid CDR version by availability wins; ties follow sorted partition/path and row order.",
              "- Tickets have no arrival timestamp. File exposure does not prove when ticket revisions or closures became observable.",
              "- Numeric defects do not erase CDR activity; zero duration/charge are valid.",
              f"- Malformed CSV policy: {report['config'].get('malformed_csv_policy', 'review')}; raw files stay unchanged.",
              f"- Evaluation allows {report['config'].get('label_delivery_lag_days', 0)} extra delivery day(s); outcome event windows stay unchanged.",
              "- No ticket text or retention_contact is used to define the target.",
              "- Repeated subscribers across snapshots are correlated observations.",
              f"- {report['config']['timestamp_assumption']}", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_audit(config):
    if config["allowed_sources"] != ["cdr", "tickets"]:
        raise ValueError("Only cdr and tickets are supported")
    for name in ("expected_days", "feature_window_days", "outcome_window_days", "snapshot_stride_days"):
        if not isinstance(config[name], int) or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    files = discover_files(config["data_dir"])
    if not files:
        raise ValueError("No allowed source files found")
    start = min(partition for partition, _, _ in files)
    expected = {start + timedelta(days=i) for i in range(config["expected_days"])}
    calendar = {"start": start.isoformat(), "end": max(expected).isoformat(), "sources": {}}
    issues, inventory = [], []
    for source in ("cdr", "tickets"):
        dates = {partition for partition, kind, _ in files if kind == source}
        calendar["sources"][source] = {
            "dates": sorted(d.isoformat() for d in dates),
            "missing": sorted(d.isoformat() for d in expected - dates),
            "unexpected": sorted(d.isoformat() for d in dates - expected),
        }
        if dates != expected:
            issues.append(f"{source}: calendar does not match {config['expected_days']} consecutive days.")
    output = Path(config["audit_dir"])
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="scan-", dir=output) as temporary:
        with closing(sqlite3.connect(Path(temporary) / "audit.sqlite")) as connection:
            create_database(connection)
            for index, (partition, source, path) in enumerate(files, 1):
                print(f"[{index}/{len(files)}] {source} {partition}", flush=True)
                inventory.append(scan_file(connection, partition, source, path, config))
            connection.execute("CREATE INDEX cdr_event_day ON cdr(event_day)")
            complete = {partition for partition, kind, _ in files if kind == "cdr"}
            schedule = snapshot_schedule(config)
            snapshots = [measure_snapshot(connection, start, day, config, complete) for day in schedule]
            unique = {source: connection.execute(f"SELECT COUNT(*) FROM {source}").fetchone()[0]
                      for source in ("cdr", "tickets")}
    for row in snapshots:
        if not row["diagnostic_inactive"] or not row["diagnostic_active"]:
            issues.append(f"Snapshot {row['snapshot_day']}: both target classes are not demonstrated.")
        ready = row["outcome_records_available_on"]
        for role, replay in schedule[row["snapshot_day"]]:
            delivery_day = replay + config.get("label_delivery_lag_days", 0)
            if not ready or ready > date_for_day(start, delivery_day).isoformat():
                issues.append(f"Snapshot {row['snapshot_day']} ({role}): outcome records are not all available by replay day {replay}.")
    for source in ("cdr", "tickets"):
        bad_availability = sum(item["quality"].get("availability_unusable", 0)
                               for item in inventory if item["source"] == source)
        if bad_availability:
            issues.append(f"{source}: {bad_availability} otherwise valid records have unusable availability; resolve before training.")
        missing_ids = sum(item["quality"].get("missing_id", 0)
                          for item in inventory if item["source"] == source)
        if missing_ids:
            issues.append(f"{source}: {missing_ids} rows lack record IDs; decide deduplication handling before training.")
        malformed = sum(item["quality"].get("malformed_csv_rows", 0)
                        for item in inventory if item["source"] == source)
        if malformed and config.get("malformed_csv_policy") != "drop":
            issues.append(f"{source}: {malformed} malformed CSV rows were excluded; review their handling before training.")
    report = {"gate": "needs_review" if issues else "passed", "issues": issues,
              "config": config, "calendar": calendar, "inventory": inventory,
              "unique_valid_ids": unique, "snapshots": snapshots, "python_version": sys.version}
    write_report(output, report)
    print(f"Gate: {report['gate']}. Report: {output / 'report.md'}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    # Пути конфигурации считаются относительно самого config.json.
    for name in ("data_dir", "audit_dir"):
        config[name] = str((args.config.resolve().parent / config[name]).resolve())
    report = run_audit(config)
    raise SystemExit(0 if report["gate"] == "passed" else 2)


if __name__ == "__main__":
    main()
