"""One diagnostic train/resume/finalize cycle on a supplied real matched cache.

No formal queue is started. All outputs are explicitly marked diagnostic.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", default="qm9_mu")
    parser.add_argument("--data", type=Path, required=True, help="One real matched condition cache, e.g. data/dft")
    parser.add_argument("--out", type=Path, required=True, help="A new temporary diagnostic directory")
    args = parser.parse_args()
    entry = read(ROOT / "experiments.json")[args.experiment]
    args.out.mkdir(parents=True, exist_ok=False)
    for name, source in entry["files"].items():
        shutil.copyfile(ROOT / source, args.out / name)
    run = args.out / "diagnostic_run"
    command = [sys.executable, "-B", str(args.out.resolve() / "train_fusion.py"),
               "--data", str(args.data.resolve()), "--out", str(run.resolve()),
               "--diagnostic", "--device", "cpu", "--seed", "42", "--epochs", "1",
               "--min-epochs", "1", "--patience", "1", "--subset", "64", "--max-steps", "1",
               "--batch-size", "4", "--hidden-dim", "16", "--heads", "4", "--smiles-layers", "1",
               "--graph-layers", "1", "--point-dim", "32", "--cpu-threads", "2", "--max-minutes", "10"]
    subprocess.run(command + ["--mode", "train"], check=True)
    before = {name: sha(run / name) for name in ("best.pt", "last.pt")}
    subprocess.run(command + ["--mode", "train", "--resume", str((run / "last.pt").resolve())], check=True)
    assert before == {name: sha(run / name) for name in before}, "Completed training resume changed weights"
    subprocess.run(command + ["--mode", "finalize"], check=True)
    summary = read(run / "summary.json")
    assert summary["status"] == "completed" and summary["best_checkpoint_reload_verified"]
    config = read(run / "config.json")
    assert config["diagnostic"] is True
    print(json.dumps({"status": "PASS", "experiment": args.experiment,
                      "checked": ["one-step training", "completed-run resume", "checkpoint reload", "held-out diagnostic evaluation"],
                      "formal_training_started": False}))


if __name__ == "__main__":
    main()
