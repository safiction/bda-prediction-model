"""Small cleaning functions and Spark aggregations for CDR and tickets."""

import hashlib
import json
import math
import re
from datetime import datetime, time, timedelta

from .data import parse_timestamp


CDR_COLUMNS = ["record_id", "subscriber_key", "event_ts", "ingest_ts", "service_type",
               "direction", "duration_sec", "bytes_up", "bytes_down", "charge", "roaming_flag"]
TICKET_COLUMNS = ["ticket_id", "subscriber_key", "created_ts", "closed_ts", "channel",
                  "category", "resolution_code", "bad_resolution_duration"]
PROVENANCE = ["available_date", "source_partition", "source_file", "source_row", "row_hash"]


def clean_row(row, source, partition, path, row_number, cutoff, stats):
    """Return a clean allowed-column row, or None with a recorded reason."""
    stats["input_rows"] += 1
    if None in row or any(value is None for value in row.values()):
        stats["dropped_malformed"] += 1
        return None
    timestamp_name = "event_ts" if source == "cdr" else "created_ts"
    id_name = "record_id" if source == "cdr" else "ticket_id"
    event = parse_timestamp(row[timestamp_name])
    key, record_id = row["subscriber_key"].strip(), row[id_name].strip()
    if not event or not re.fullmatch(r"SUB[0-9]{9}", key) or not record_id:
        stats["dropped_invalid_identity_or_timestamp"] += 1
        return None
    available = max(partition, event.date())
    result = {id_name: record_id, "subscriber_key": key, timestamp_name: event.isoformat(" ")}
    if source == "cdr":
        ingest = parse_timestamp(row["ingest_ts"])
        if not ingest or ingest < event:
            stats["dropped_invalid_availability"] += 1
            return None
        available = max(available, ingest.date())
        result["ingest_ts"] = ingest.isoformat(" ")
        for name in ("duration_sec", "bytes_up", "bytes_down", "charge"):
            try:
                value = float(row[name])
                if not math.isfinite(value) or value < 0:
                    raise ValueError
                result[name] = value
            except ValueError:
                result[name] = None
                stats[f"invalid_{name}"] += 1
        for name in ("service_type", "direction"):
            result[name] = row.get(name, "").strip().lower() or "unknown"
        flag = row.get("roaming_flag", "").strip().lower()
        result["roaming_flag"] = {"true": "true", "1": "true", "false": "false", "0": "false"}.get(flag)
        if result["roaming_flag"] is None:
            stats["unknown_roaming_flag"] += 1
    else:
        closed = parse_timestamp(row["closed_ts"])
        bad_duration = bool(closed and closed < event)
        if bad_duration:
            stats["invalid_resolution_duration"] += 1
        if row["closed_ts"].strip() and not closed:
            stats["invalid_closed_timestamp"] += 1
        result["closed_ts"] = closed.isoformat(" ") if closed and not bad_duration else None
        result["bad_resolution_duration"] = "true" if bad_duration else "false"
        for name in ("category", "channel", "resolution_code"):
            result[name] = row.get(name, "").strip().lower() or None
    if available > cutoff:
        stats["pending_availability"] += 1
        stats[f"pending_date:{event.date()}"] += 1
        return None
    if event >= datetime.combine(cutoff + timedelta(days=1), time.min):
        stats["pending_event_date"] += 1
        return None
    # Хешируем только разрешённые поля; текст и retention_contact не сохраняем.
    result["row_hash"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result.update(available_date=available.isoformat(), source_partition=partition.isoformat(),
                  source_file=path.as_posix(), source_row=row_number)
    stats["accepted_rows"] += 1
    return result


def read_staged(spark, path, source):
    from pyspark.sql import functions as F
    columns = (CDR_COLUMNS if source == "cdr" else TICKET_COLUMNS) + PROVENANCE
    frame = spark.read.schema(",".join(f"{name} STRING" for name in columns)).option("header", True).csv(str(path))
    timestamps = ("event_ts", "ingest_ts") if source == "cdr" else ("created_ts", "closed_ts")
    for name in timestamps:
        frame = frame.withColumn(name, F.to_timestamp(name, "yyyy-MM-dd HH:mm:ss"))
    for name in ("available_date", "source_partition"):
        frame = frame.withColumn(name, F.to_date(name))
    frame = frame.withColumn("source_row", F.col("source_row").cast("long"))
    if source == "cdr":
        for name in ("duration_sec", "bytes_up", "bytes_down", "charge"):
            frame = frame.withColumn(name, F.col(name).cast("double"))
        frame = frame.withColumn("roaming_flag", F.col("roaming_flag").cast("boolean"))
        frame = frame.withColumn("event_date", F.to_date("event_ts"))
    else:
        frame = frame.withColumn("bad_resolution_duration", F.col("bad_resolution_duration").cast("boolean"))
        frame = frame.withColumn("event_date", F.to_date("created_ts"))
    return frame


def deduplicate(frame, source):
    from pyspark.sql import Window, functions as F
    keys = ["record_id"] if source == "cdr" else ["ticket_id", "row_hash"]
    # Версии tickets сохраняем: исторический снимок сам выбирает доступную ему версию.
    order = Window.partitionBy(*keys).orderBy("available_date", "source_partition", "source_file", "source_row")
    return frame.withColumn("_rank", F.row_number().over(order)).filter("_rank = 1").drop("_rank")


def tickets_as_of(frame, snapshot):
    from pyspark.sql import Window, functions as F
    end = str(snapshot + timedelta(days=1))
    frame = frame.filter((F.col("available_date") <= F.lit(str(snapshot))) & (F.col("created_ts") < F.lit(end)))
    order = Window.partitionBy("ticket_id").orderBy(F.col("available_date").desc(),
             F.col("source_partition").desc(), F.col("source_file").desc(), F.col("source_row").desc())
    frame = frame.withColumn("_rank", F.row_number().over(order)).filter("_rank = 1").drop("_rank")
    frame = frame.withColumn("closed_ts", F.when(F.col("closed_ts") < F.lit(end), F.col("closed_ts")))
    return frame.withColumn("resolution_code", F.when(F.col("closed_ts").isNotNull(), F.col("resolution_code")))


def daily_cdr(frame):
    from pyspark.sql import functions as F
    voice = F.col("service_type") == "voice"
    # Дату доступности сохраняем, чтобы поздняя доставка не меняла прошлые признаки.
    expressions = [F.count("*").alias("event_count")]
    for service in ("voice", "sms", "data"):
        expressions.append(F.sum(F.when(F.col("service_type") == service, 1).otherwise(0)).alias(f"{service}_count"))
    for name in ("bytes_up", "bytes_down", "charge"):
        expressions += [F.sum(name).alias(f"{name}_sum"), F.count(name).alias(f"{name}_valid_count")]
    expressions += [F.sum(F.when(voice, F.col("duration_sec"))).alias("voice_duration_sum"),
                    F.count(F.when(voice, F.col("duration_sec"))).alias("voice_duration_valid_count"),
                    F.sum(F.when(F.col("roaming_flag"), 1).otherwise(0)).alias("roaming_count"),
                    F.count("roaming_flag").alias("roaming_known_count")]
    valid_traffic = F.col("bytes_up").isNotNull() & F.col("bytes_down").isNotNull()
    expressions += [F.sum(F.when(valid_traffic, F.col("bytes_up") + F.col("bytes_down"))).alias("traffic_bytes_sum"),
                    F.count(F.when(valid_traffic, 1)).alias("traffic_valid_count")]
    return frame.groupBy("subscriber_key", "event_date", "available_date").agg(*expressions)
