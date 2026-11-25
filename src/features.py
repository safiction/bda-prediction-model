"""HW_4_1/HW_4_2, tasks 3–4: aggregate each source, then left join.

Adaptations: explicit historical dates, exact active days, valid-value denominators,
and observable ticket closures. No preprocessing is fitted at this stage.
"""

from datetime import timedelta

from .preprocess import tickets_as_of


KEY_COLUMNS = ["subscriber_key", "snapshot_date"]


def validate_feature_columns(columns, config):
    expected = set(KEY_COLUMNS + config["feature_allowlist"])
    if len(columns) != len(set(columns)) or set(columns) != expected:
        raise ValueError(f"Unexpected feature schema: extra={set(columns) - expected}, missing={expected - set(columns)}")
    if set(columns) & set(config["excluded_inputs"]):
        raise ValueError("Forbidden columns cannot be features")


def build_features(daily, tickets, snapshot_date, config):
    from pyspark.sql import functions as F
    if config["feature_window_days"] != 14:
        raise ValueError("The initial seven-day trend features require a 14-day feature window")
    for frame in (daily, tickets):
        if set(frame.columns) & set(config["excluded_inputs"]):
            raise ValueError("Forbidden source columns cannot enter feature construction")
    start = snapshot_date - timedelta(days=13)
    end = snapshot_date + timedelta(days=1)
    middle = snapshot_date - timedelta(days=6)
    history = daily.filter((F.col("event_date") >= F.lit(str(start))) &
                           (F.col("event_date") < F.lit(str(end))) &
                           (F.col("available_date") <= F.lit(str(snapshot_date))))
    # HW_4: сначала агрегируем CDR до одной строки на абонента.
    cdr = history.groupBy("subscriber_key").agg(
        F.sum("event_count").alias("event_count"),
        F.countDistinct("event_date").alias("active_days"),
        F.max("event_date").alias("last_activity_date"),
        *[F.sum(name).alias(name) for name in ("voice_count", "sms_count", "data_count",
            "voice_duration_sum", "voice_duration_valid_count", "bytes_up_sum", "bytes_down_sum",
            "traffic_bytes_sum", "traffic_valid_count", "charge_sum", "charge_valid_count",
            "roaming_count", "roaming_known_count")],
        F.sum(F.when(F.col("event_date") >= F.lit(str(middle)), F.col("event_count")).otherwise(0)).alias("events_last_7d"),
        F.sum(F.when(F.col("event_date") < F.lit(str(middle)), F.col("event_count")).otherwise(0)).alias("events_previous_7d"),
    ).withColumn("days_since_last_activity", F.datediff(F.lit(str(snapshot_date)), "last_activity_date"))

    # Средние и доли считаем по корректным наблюдениям; нулевой знаменатель даёт пропуск.
    for service in ("voice", "sms", "data"):
        cdr = cdr.withColumn(f"{service}_share", F.col(f"{service}_count") / F.col("event_count"))
    cdr = (cdr.withColumn("total_voice_duration_sec", F.when(F.col("voice_count") == 0, 0).otherwise(F.col("voice_duration_sum")))
           .withColumn("total_bytes_up", F.col("bytes_up_sum"))
           .withColumn("total_bytes_down", F.col("bytes_down_sum"))
           .withColumn("total_charge", F.col("charge_sum")))
    ratios = {
        "avg_voice_duration_sec": ("voice_duration_sum", "voice_duration_valid_count"),
        "voice_duration_valid_share": ("voice_duration_valid_count", "voice_count"),
        "avg_traffic_bytes_per_event": ("traffic_bytes_sum", "traffic_valid_count"),
        "traffic_valid_share": ("traffic_valid_count", "event_count"),
        "avg_charge": ("charge_sum", "charge_valid_count"),
        "charge_valid_share": ("charge_valid_count", "event_count"),
        "roaming_share": ("roaming_count", "roaming_known_count"),
        "roaming_known_share": ("roaming_known_count", "event_count"),
    }
    for name, (numerator, denominator) in ratios.items():
        cdr = cdr.withColumn(name, F.when(F.col(denominator) > 0, F.col(numerator) / F.col(denominator)))
    cdr = cdr.withColumn("activity_change_ratio", F.when(F.col("events_previous_7d") > 0,
        (F.col("events_last_7d") - F.col("events_previous_7d")) / F.col("events_previous_7d")))

    # HW_4: отдельно агрегируем tickets и только затем выполняем LEFT JOIN.
    ticket_history = tickets_as_of(tickets, snapshot_date).filter(
        (F.col("created_ts") >= F.lit(str(start))) & (F.col("created_ts") < F.lit(str(end))))
    category = F.when(F.col("category").isin("quality", "connection", "billing"), F.col("category")).otherwise("other")
    ticket_counts = ["ticket_count", "unresolved_ticket_count", "closed_ticket_count"]
    category_counts = [f"ticket_{value}_count" for value in ("quality", "connection", "billing", "other")]
    ticket_agg = ticket_history.groupBy("subscriber_key").agg(
        F.count("*").alias("ticket_count"),
        F.sum(F.when(F.col("closed_ts").isNull(), 1).otherwise(0)).alias("unresolved_ticket_count"),
        F.count("closed_ts").alias("closed_ticket_count"),
        F.avg((F.col("closed_ts").cast("long") - F.col("created_ts").cast("long")) / 3600).alias("avg_resolution_hours"),
        *[F.sum(F.when(category == value, 1).otherwise(0)).alias(f"ticket_{value}_count")
          for value in ("quality", "connection", "billing", "other")],
    )
    features = cdr.join(ticket_agg, "subscriber_key", "left").fillna(0, subset=ticket_counts + category_counts)
    features = (features.withColumn("resolution_missing", (F.col("closed_ticket_count") == 0).cast("int"))
                .withColumn("snapshot_date", F.to_date(F.lit(str(snapshot_date)))))
    # Промежуточные поля и служебные даты не входят в модельную таблицу.
    features = features.select(*KEY_COLUMNS, *config["feature_allowlist"])
    validate_feature_columns(features.columns, config)
    return features
