"""Future activity is used only here, independently of historical features."""

from datetime import timedelta

from .temporal import outcome_is_complete


def build_labels(cdr, cohort, snapshot_date, available_through, complete_dates, horizon=14):
    from pyspark.sql import functions as F
    keys = cohort.select("subscriber_key", "snapshot_date")
    if not outcome_is_complete(snapshot_date, horizon, available_through, complete_dates):
        # Неизвестную метку никогда не заменяем нулём или единицей.
        return keys.withColumn("target", F.lit(None).cast("int"))
    start = snapshot_date + timedelta(days=1)
    end = snapshot_date + timedelta(days=horizon + 1)
    active = (cdr.filter((F.col("event_ts") >= F.lit(str(start))) &
                        (F.col("event_ts") < F.lit(str(end))) &
                        (F.col("event_date") >= F.lit(str(start))) &
                        (F.col("event_date") < F.lit(str(end))) &
                        (F.col("available_date") <= F.lit(str(available_through))))
              .select("subscriber_key").distinct().withColumn("target", F.lit(0)))
    return keys.join(active, "subscriber_key", "left").fillna(1, subset=["target"]).select(
        "subscriber_key", "snapshot_date", "target")
