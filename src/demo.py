"""Run or resume the agreed demonstration through the shared background worker."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from .final import read_json
from .worker import actions, current_job, load_config, replay_day, submit_job


def next_action(options):
    for action in ("train", "prepare", "freeze", "evaluate"):
        if options[action]["enabled"]:
            return action
    if options["snapshots"]["enabled"] and not options["train"]["ready"]:
        return "snapshots"
    raise ValueError("No valid next step. Check job logs and replay state.")


def run_demo(config_path="config.json", poll_seconds=5):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    audit = read_json(Path(config["audit_dir"]) / "audit.json", {})
    rules = ("feature_window_days", "outcome_window_days", "expected_days", "label_delivery_lag_days", "malformed_csv_policy")
    if audit.get("gate") != "passed" or any(audit.get("config", {}).get(name) != config[name] for name in rules):
        subprocess.run([sys.executable, "-m", "src.audit", "--config", str(config_path)], check=True)
    watched = None
    last_message = None
    while True:
        if watched:
            job = read_json(Path(config["state_dir"]) / "jobs" / f"{watched}.json")
            if job["status"] == "failed":
                raise RuntimeError(f"Job {watched} failed: {job.get('error')}. Fix the cause and rerun this command.")
        result = read_json(Path(config["artifacts_dir"]) / "final_evaluation/comparison.json")
        if result is not None and current_job(config) is None:
            print(f"Demo completed: {result['rows']} subscribers; results in {config['artifacts_dir']}/final_evaluation", flush=True)
            return result
        job = current_job(config)
        if job is None:
            action = next_action(actions(config))
            try:
                job = submit_job(action, config_path)
                print(f"Submitted {action}: {job['job_id']}", flush=True)
            except (ValueError, FileExistsError):
                # Другая вкладка могла отправить то же действие первой.
                time.sleep(poll_seconds)
                continue
        watched = job["job_id"]
        message = (watched, job["status"], job["stage"], job.get("files_processed", 0))
        if message != last_message:
            print(f"Replay {replay_day(config)} | {job['action']} | {job['stage']} | files {job.get('files_processed', 0)}", flush=True)
            last_message = message
        time.sleep(poll_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    result = run_demo(args.config)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
