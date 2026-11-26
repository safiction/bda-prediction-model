"""One local background worker; jobs and logs survive browser refresh and restart."""

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .final import read_json
from .ingest import atomic_json
from .prepared import reuse_prepared


def load_config(path="config.json"):
    path = Path(path).resolve()
    config = read_json(path)
    for name in ("data_dir", "audit_dir", "processed_dir", "snapshots_dir", "state_dir", "artifacts_dir"):
        config[name] = str((path.parent / config[name]).resolve())
    return config


def now():
    return datetime.now(timezone.utc).isoformat()


def replay_day(config):
    state = Path(config["state_dir"])
    return read_json(state / "replay.json", read_json(state / "ingest.json", {"through_day": 0}))["through_day"]


def completed_training(config, milestone, fingerprint=None):
    for path in (Path(config["artifacts_dir"]) / "runs").glob("*/run.json"):
        run = read_json(path)
        if run["status"] == "completed" and run["milestone"] == milestone and (fingerprint is None or run["input_fingerprint"] == fingerprint):
            return True
    return False


def actions(config):
    state = Path(config["state_dir"])
    ingest = read_json(state / "ingest.json", {})
    clock = replay_day(config)
    committed = read_json(state / "replay.json", ingest)
    lag = config["label_delivery_lag_days"]
    milestones = config["training_replay_days"]
    current = next((day for day in milestones if day + lag == clock), None)
    snapshots = read_json(state / "snapshots.json", {})
    required = list(range(config["feature_window_days"], current - 2 * config["outcome_window_days"] + 1,
                          config["snapshot_stride_days"])) + [current - config["outcome_window_days"]] if current else []
    ready = bool(current and ingest.get("through_day") == clock and snapshots.get("input_fingerprint") == ingest.get("fingerprint") and
                 all(any(row["snapshot_day"] == day and row["label_status"] == "mature" for row in snapshots.get("snapshots", [])) for day in required))
    trained = bool(current and completed_training(config, current, ingest.get("fingerprint")))
    next_cutoff = next((day + lag for day in milestones if day + lag > clock), None)
    previous = [day for day in milestones if day + lag <= clock]
    committed_fingerprint = committed.get("input_fingerprint", committed.get("fingerprint"))
    previous_done = not previous or completed_training(config, previous[-1], committed_fingerprint)
    frozen = (Path(config["artifacts_dir"]) / "final_predictions/manifest.json").exists()
    evaluated = (Path(config["artifacts_dir"]) / "final_evaluation/comparison.json").exists()
    return {
        "prepare": {"enabled": next_cutoff is not None and previous_done, "cutoff": next_cutoff,
                    "training_number": milestones.index(next_cutoff - lag) + 1 if next_cutoff is not None else None,
                    "label": "Открыть данные"},
        "train": {"enabled": ready and not trained, "milestone": current,
                  "training_number": milestones.index(current) + 1 if current is not None else None,
                  "label": "Обучить модели"},
        "freeze": {"enabled": clock in (milestones[-1] + lag, config["final_snapshot_day"]) and not frozen and
                   all(completed_training(config, day) for day in milestones), "cutoff": config["final_snapshot_day"], "label": "Зафиксировать прогнозы"},
        "evaluate": {"enabled": frozen and not evaluated and clock in (config["final_snapshot_day"], config["final_replay_day"] + lag),
                     "cutoff": config["final_replay_day"] + lag, "label": "Проверить прогнозы"},
    }


def process_alive(pid, token):
    # Приложение работает в Linux-контейнере; UUID защищает от повторного использования PID.
    if not pid:
        return False
    path = Path(f"/proc/{pid}/cmdline")
    try:
        return token.encode() in path.read_bytes()
    except (FileNotFoundError, PermissionError):
        return False


def current_job(config):
    jobs = Path(config["state_dir"]) / "jobs"
    claim = read_json(jobs / "active.json")
    if claim is None:
        return None
    job = read_json(jobs / f"{claim['job_id']}.json", claim)
    if job.get("status") in ("completed", "failed"):
        if process_alive(job.get("pid"), job["job_id"]):
            return job
        (jobs / "active.json").unlink(missing_ok=True)
        return None
    alive = process_alive(job.get("pid"), job["job_id"]) or process_alive(job.get("stage_pid"), "src.")
    if alive or time.time() - job["created_timestamp"] < 30:
        return job
    job.update(status="failed", error="Выполнение прервано. Последний успешный результат сохранён; повторите действие.", finished_at=now())
    atomic_json(jobs / f"{job['job_id']}.json", job)
    lock = Path(config["state_dir"]) / "ingest.lock"
    if lock.exists() and lock.read_text().strip() == str(job.get("stage_pid")):
        lock.unlink()
    (jobs / "active.json").unlink(missing_ok=True)
    return None


def submit_job(action, config_path="config.json"):
    config_path = Path(config_path).resolve()
    config = load_config(config_path)
    if current_job(config) is not None:
        raise ValueError("Уже выполняется задание")
    options = actions(config)
    if action not in options or not options[action]["enabled"]:
        raise ValueError("Для этого действия ещё не выполнены условия")
    if (Path(config["state_dir"]) / "ingest.lock").exists():
        raise ValueError("Сейчас работает командный обработчик. Дождитесь завершения.")
    jobs = Path(config["state_dir"]) / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    job_id = uuid.uuid4().hex
    job = {"job_id": job_id, "action": action, "status": "queued",
           "created_at": now(), "created_timestamp": time.time(), "pid": None, "stage_pid": None,
           "config_path": str(config_path)}
    # O_EXCL — общий барьер для двух вкладок или двух одновременных нажатий.
    with (jobs / "active.json").open("x", encoding="utf-8") as stream:
        json.dump(job, stream)
    try:
        atomic_json(jobs / f"{job_id}.json", job)
        kwargs = {"start_new_session": True} if os.name == "posix" else {"creationflags": subprocess.CREATE_NO_WINDOW}
        with (jobs / f"{job_id}.log").open("a", encoding="utf-8") as log:
            subprocess.Popen([sys.executable, "-u", "-m", "src.worker", "--config", str(config_path), "--job-id", job_id],
                             stdout=log, stderr=subprocess.STDOUT, **kwargs)
        return job
    except Exception as error:
        # Освобождаем свою очередь до публикации ошибки запуска: новый submit
        # не должен потерять уже созданную им active.json.
        (jobs / "active.json").unlink(missing_ok=True)
        job.update(status="failed", error=str(error), finished_at=now())
        atomic_json(jobs / f"{job_id}.json", job)
        raise


def run_command(config, job, module, arguments):
    jobs = Path(config["state_dir"]) / "jobs"
    command = [sys.executable, "-u", "-m", module, "--config", job["config_path"], *arguments]
    process = subprocess.Popen(command, stderr=subprocess.STDOUT)
    job["stage_pid"] = process.pid
    atomic_json(jobs / f"{job['job_id']}.json", job)
    returncode = process.wait()
    lock = Path(config["state_dir"]) / "ingest.lock"
    if lock.exists() and lock.read_text().strip() == str(process.pid) and not process_alive(process.pid, "src."):
        lock.unlink()
    job["stage_pid"] = None
    if returncode:
        raise RuntimeError("Не удалось выполнить задание. Проверьте подготовленные данные и повторите попытку.")


def run_job(job_id, config_path):
    config = load_config(config_path)
    jobs = Path(config["state_dir"]) / "jobs"
    job = read_json(jobs / f"{job_id}.json")
    claim = read_json(jobs / "active.json", {})
    if not job or claim.get("job_id") != job_id:
        raise ValueError("Задание не владеет очередью")
    try:
        job.update(status="running", pid=os.getpid(), started_at=now())
        atomic_json(jobs / f"{job_id}.json", job)
        option = actions(config)[job["action"]]
        if not option["enabled"]:
            raise ValueError("Условия задания изменились; обновите страницу")
        replay = Path(config["state_dir"]) / "replay.json"
        if not replay.exists():
            atomic_json(replay, {"through_day": replay_day(config),
                                 "input_fingerprint": read_json(Path(config["state_dir"]) / "ingest.json", {}).get("fingerprint")})
        action = job["action"]
        if action in ("prepare", "freeze", "evaluate"):
            cutoff = option["cutoff"]
            job["cutoff"] = cutoff
            if action == "prepare":
                milestone = cutoff - config["label_delivery_lag_days"]
                days = list(range(config["feature_window_days"], milestone - 2 * config["outcome_window_days"] + 1, config["snapshot_stride_days"]))
                days.append(milestone - config["outcome_window_days"])
            else:
                days = [config["final_snapshot_day"]]
            if not reuse_prepared(config, cutoff, days):
                raise ValueError(f"Подготовленные данные для дня {cutoff} не найдены или не соответствуют исходникам. Сначала подготовьте набор данных отдельно.")
            if action != "prepare":
                run_command(config, job, "src.final", [action])
            atomic_json(replay, {"through_day": cutoff, "job_id": job_id,
                                 "input_fingerprint": read_json(Path(config["state_dir"]) / "ingest.json")["fingerprint"]})
        elif action == "train":
            run_command(config, job, "src.train", ["--milestone", str(option["milestone"])])
        job.update(status="completed", finished_at=now())
    except Exception as error:
        job.update(status="failed", error=str(error), finished_at=now())
    finally:
        atomic_json(jobs / f"{job_id}.json", job)
        if read_json(jobs / "active.json", {}).get("job_id") == job_id:
            (jobs / "active.json").unlink(missing_ok=True)
    return job


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    job = run_job(args.job_id, args.config.resolve())
    raise SystemExit(0 if job["status"] == "completed" else 1)


if __name__ == "__main__":
    main()
