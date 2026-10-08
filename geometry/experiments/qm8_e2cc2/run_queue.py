"""Bounded QM8 E2-CC2 gate queue for two geometries and three fixed seeds.

This supervisor launches only its own child processes. It never powers on,
deletes an instance, evaluates test data before all training completes, or
overwrites a completed run. Start with --dry-run to inspect the command list.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


GEOMETRIES = ("dft", "mmff")
SEEDS = (42, 2025, 3407)
ROOT = Path(__file__).resolve().parent


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def jobs_for(args):
    jobs = []
    # Interleave geometries within each seed; every choice is fixed before results.
    for seed in SEEDS:
        for geometry in GEOMETRIES:
            run = Path(args.runs).resolve() / geometry / f"seed_{seed}"
            jobs.append({"name": f"{geometry}_seed_{seed}", "fusion": "gate",
                         "geometry_condition": geometry,
                         "data": str(Path(args.data).resolve() / geometry),
                         "seed": seed, "out": str(run)})
    return jobs


def command_for(args, job, mode):
    command = [
        sys.executable, "-u", str(ROOT / "train_fusion.py"),
        "--mode", mode, "--data", job["data"],
        "--out", job["out"], "--fusion", job["fusion"],
        "--seed", str(job["seed"]), "--split-seed", "20260927",
        "--epochs", "200", "--min-epochs", "50", "--patience", "30",
        "--max-minutes", "420", "--device", args.device
    ]
    last_checkpoint = Path(job["out"]) / "last.pt"
    if mode == "train" and last_checkpoint.exists():
        command.extend(["--resume", str(last_checkpoint)])
    return command


def stop_children(active):
    # Each child owns a newly created process group; never use broad pkill.
    for item in active:
        process = item["process"]
        if process.poll() is None:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            else:
                process.terminate()
    deadline = time.monotonic() + 15
    for item in active:
        process = item["process"]
        try:
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
        item["log"].close()


def run_queue(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "queue.lock"
    # An interrupted supervisor requires explicit inspection before lock removal.
    lock_fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(lock_fd, str(os.getpid()).encode("ascii"))
    os.close(lock_fd)
    started = time.monotonic()
    active = []
    jobs = jobs_for(args)
    state = {
        "status": "running", "pid": os.getpid(), "phase": "train",
        "dataset": "QM8", "target": "E2-CC2", "unit": "hartree", "original_cache_rows": 21747,
        "geometries": GEOMETRIES, "fusion_type": "gate",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "max_hours": args.max_hours, "concurrency": args.concurrency,
        "gpu_hourly_price_reference_cny": 1.88,
        "gpu_reference_cost_at_limit_cny": round(args.max_hours * 1.88, 2),
        "jobs": jobs, "completed_training": [], "completed_finalization": []
    }

    def persist():
        state["elapsed_seconds"] = time.monotonic() - started
        state["active"] = [{"job": item["job"]["name"], "pid": item["process"].pid}
                           for item in active]
        save_json(output / "queue_status.json", state)

    def interrupted(signum, frame):
        raise InterruptedError(f"Supervisor received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    env = dict(os.environ, OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="0")
    try:
        # Both phases are barriers. No test scores exist until all six train jobs
        # have succeeded and their best-validation checkpoints are fixed.
        for phase in ("train", "finalize"):
            state["phase"] = phase
            pending = list(jobs)
            while pending or active:
                if time.monotonic() - started >= args.max_hours * 3600:
                    raise TimeoutError("Queue wall-time limit reached; checkpoints retained")
                capacity = args.concurrency if phase == "train" else 1
                while pending and len(active) < capacity:
                    job = pending.pop(0)
                    run = Path(job["out"])
                    status = read_json(run / "status.json").get("status")
                    if phase == "train" and status in ("trained", "completed"):
                        state["completed_training"].append(job["name"])
                        continue
                    if phase == "finalize" and status == "completed":
                        state["completed_finalization"].append(job["name"])
                        continue
                    run.mkdir(parents=True, exist_ok=True)
                    log_path = output / f"{job['name']}.{phase}.log"
                    log = log_path.open("a", encoding="utf-8")
                    command = command_for(args, job, phase)
                    log.write(json.dumps({"command": command,
                                          "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
                    log.flush()
                    process = subprocess.Popen(
                        command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                        stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                    )
                    active.append({"process": process, "log": log, "job": job})
                persist()
                for item in list(active):
                    code = item["process"].poll()
                    if code is None:
                        continue
                    item["log"].close()
                    active.remove(item)
                    job = item["job"]
                    status = read_json(Path(job["out"]) / "status.json").get("status")
                    required = ("trained", "completed") if phase == "train" else ("completed",)
                    if code != 0 or status not in required:
                        raise RuntimeError(
                            f"{job['name']} {phase} failed: exit={code}, status={status}; see job log"
                        )
                    key = "completed_training" if phase == "train" else "completed_finalization"
                    state[key].append(job["name"])
                    persist()
                if pending or active:
                    time.sleep(5)
        state["status"] = "completed"
        state["phase"] = "complete"
        persist()
        print(json.dumps(state, ensure_ascii=False), flush=True)
    except BaseException as exc:
        stop_children(active)
        active.clear()
        state.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        persist()
        raise
    finally:
        # Only this process's lock, created above, is removed.
        lock.unlink()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=str(ROOT / "data"), help="Parent of matched dft/ and mmff/ caches")
    parser.add_argument("--runs", default=str(ROOT / "runs"))
    parser.add_argument("--output", default=str(ROOT / "queue"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--max-hours", type=float, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.concurrency <= 6 or not 0 < args.max_hours <= 12:
        parser.error("concurrency must be 1..6 and max-hours must be in (0,12]")
    return args


if __name__ == "__main__":
    options = parse_args()
    if options.dry_run:
        print(json.dumps([{"job": job, "command": command_for(options, job, "train")}
                          for job in jobs_for(options)], indent=2))
    elif options.detach:
        output = Path(options.output).resolve()
        output.mkdir(parents=True, exist_ok=True)
        if (output / "queue.lock").exists():
            raise RuntimeError("An active or uninspected previous queue.lock exists")
        arguments = [argument for argument in sys.argv[1:] if argument != "--detach"]
        log_path = output / "supervisor.log"
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), *arguments],
                cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True
            )
        print(json.dumps({"pid": process.pid, "log": str(log_path)}))
    else:
        run_queue(options)
