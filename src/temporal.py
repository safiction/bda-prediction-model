"""Calendar rules shared by the audit and the later pipeline."""

from datetime import timedelta


def date_for_day(start, day):
    if day < 1:
        raise ValueError("Dataset day numbers start at 1")
    return start + timedelta(days=day - 1)


def snapshot_schedule(config):
    """Return the role of each snapshot and its planned label cutoff."""
    window = config["feature_window_days"]
    horizon = config["outcome_window_days"]
    stride = config["snapshot_stride_days"]
    uses = {}
    for replay in config["training_replay_days"]:
        for day in range(window, replay - 2 * horizon + 1, stride):
            uses.setdefault(day, []).append((f"train at {replay}", replay))
        day = replay - horizon
        uses.setdefault(day, []).append((f"validation at {replay}", replay))
    uses.setdefault(config["final_snapshot_day"], []).append(
        ("final evaluation", config["final_replay_day"])
    )
    return dict(sorted(uses.items()))


def outcome_is_complete(snapshot, horizon, available_through, complete_dates):
    end = snapshot + timedelta(days=horizon)
    required = {snapshot + timedelta(days=i) for i in range(1, horizon + 1)}
    return end <= available_through and required <= set(complete_dates)


def replay_dates(start, through_day, config):
    """Delivery time can extend past the last event date without a new partition."""
    last = config["expected_days"]
    if not 1 <= through_day <= last + config.get("label_delivery_lag_days", 0):
        raise ValueError("Replay day is outside the configured calendar and delivery allowance")
    return date_for_day(start, min(through_day, last)), date_for_day(start, through_day)


def snapshot_windows(start, day, config):
    snapshot = date_for_day(start, day)
    return {"snapshot_day": day, "snapshot_date": str(snapshot),
            "feature_start": str(snapshot - timedelta(days=config["feature_window_days"] - 1)),
            "feature_end": str(snapshot), "outcome_start": str(snapshot + timedelta(days=1)),
            "outcome_end": str(snapshot + timedelta(days=config["outcome_window_days"]))}


def temporal_split_manifest(start, config):
    schedule = snapshot_schedule(config)
    runs = []
    for replay in config["training_replay_days"]:
        training = [snapshot_windows(start, day, config) for day, uses in schedule.items()
                    if (f"train at {replay}", replay) in uses]
        validation = snapshot_windows(start, replay - config["outcome_window_days"], config)
        if not training or max(row["outcome_end"] for row in training) >= validation["outcome_start"]:
            raise ValueError("Training outcomes must precede the validation outcome period")
        runs.append({"milestone_day": replay,
                     "delivery_clock_day": replay + config.get("label_delivery_lag_days", 0),
                     "training": training, "validation": validation})
    return {"runs": runs, "final": snapshot_windows(start, config["final_snapshot_day"], config),
            "final_delivery_clock_day": config["final_replay_day"] + config.get("label_delivery_lag_days", 0),
            "note": "Subscribers may repeat across chronological snapshots; observations are correlated."}
