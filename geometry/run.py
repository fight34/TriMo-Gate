"""Materialize and run the archived geometry experiments without site-specific paths."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def prepare(args, entry):
    work = args.work_dir.resolve()
    if work.exists() and any(work.iterdir()):
        raise ValueError("--work-dir must be empty; existing results are never overwritten")
    data = args.data.resolve()
    expected_split = read(HERE / entry["split"])
    # Audit all inputs before creating the runtime directory.
    metadata = {}
    for condition in ("dft", "mmff"):
        folder = data / condition
        meta = read(folder / "metadata.json")
        if read(folder / "fixed_split_ids.json") != expected_split:
            raise ValueError("Prepared cache does not have the published matched ID split")
        if digest(folder / "cache.npz") != meta["cache_sha256"]:
            raise ValueError("Prepared cache checksum mismatch")
        if digest(folder / "fixed_split_ids.json") != meta["fixed_split_sha256"]:
            raise ValueError("Prepared split checksum mismatch")
        if meta["dataset"] != entry["dataset"] or meta["geometry_condition"] != condition:
            raise ValueError("Wrong dataset or coordinate condition")
        if entry["target"] not in meta["target_names"]:
            raise ValueError("Target missing from the supplied cache")
        # Unit declarations are property-specific in the historical metadata.
        units = meta.setdefault("target_units", {})
        if entry["target"] in units and units[entry["target"]] != entry["unit"]:
            raise ValueError("Existing target unit disagrees with the frozen experiment")
        units[entry["target"]] = entry["unit"]
        metadata[condition] = meta
    etkdg = data / "etkdg_test.npz"
    if not etkdg.is_file():
        raise FileNotFoundError("Prepared data require etkdg_test.npz with ids and points")
    if entry["dataset"] == "QM9" and not (data / "identity_categories.json").is_file():
        raise FileNotFoundError("QM9 requires identity_categories.json from source verification")
    work.mkdir(parents=True, exist_ok=True)
    for runtime_name, source in entry["files"].items():
        shutil.copyfile(HERE / source, work / runtime_name)
    for condition, meta in metadata.items():
        target = work / "data" / condition
        target.mkdir(parents=True)
        for name in ("cache.npz", "fixed_split_ids.json"):
            original = data / condition / name
            if args.copy_data:
                shutil.copyfile(original, target / name)
            else:
                try:
                    os.link(original, target / name)
                except OSError as exc:
                    raise OSError("Data hardlink failed; use an empty directory on the same volume, "
                                  "or --copy-data if sufficient disk space is available") from exc
        (target / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    shutil.copyfile(etkdg, work / "data/etkdg_test.npz")
    if entry["dataset"] == "QM9":
        shutil.copyfile(data / "identity_categories.json", work / "identity_categories.json")
    protocol = read(work / "experiment_protocol.json")
    protocol["etkdg_test_sha256"] = digest(work / "data/etkdg_test.npz")
    protocol["release_runtime_preparation"] = {
        "experiment": args.experiment, "source_status": entry["source_status"],
        "note": "Fresh reproduction runtime; this is not the historical server directory.",
        "data_sha256": {condition: digest(work / "data" / condition / "cache.npz")
                        for condition in ("dft", "mmff")},
    }
    (work / "experiment_protocol.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    (work / "release_source_manifest.json").write_text(json.dumps({
        name: digest(work / name) for name in entry["files"]
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"prepared": str(work), "experiment": args.experiment,
                      "note": "No training or evaluation was started."}))


def main():
    catalog = read(HERE / "experiments.json")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "prepare", "train", "evaluate"))
    parser.add_argument("--experiment", choices=sorted(catalog))
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--data", type=Path, help="Prepared dft/, mmff/, etkdg_test.npz and QM9 identity categories")
    parser.add_argument("--copy-data", action="store_true", help="Copy large caches instead of same-volume hardlinks")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--concurrency", type=int, choices=range(1, 7),
                        help="Default/max: four for historical mu/E1-CC2, six for later experiments")
    parser.add_argument("--dry-run", action="store_true", help="Print the frozen six training commands")
    parser.add_argument("--coordinates", type=Path, help="Full generation output required by historical CPU ABCD")
    args = parser.parse_args()
    if args.action == "list":
        for key, entry in catalog.items():
            print(f"{key}: {entry['target']} [{entry['unit']}], {entry['source_status']}")
        return
    if not args.experiment or args.work_dir is None:
        parser.error("--experiment and --work-dir are required")
    entry = catalog[args.experiment]
    max_concurrency = 4 if entry["evaluation_backend"] == "cpu" else 6
    concurrency = args.concurrency if args.concurrency is not None else max_concurrency
    if concurrency > max_concurrency:
        parser.error(f"{args.experiment} retains the historical maximum concurrency {max_concurrency}")
    if args.action == "prepare":
        if args.data is None:
            parser.error("prepare requires --data")
        prepare(args, entry)
        return
    work = args.work_dir.resolve()
    manifest = read(work / "release_source_manifest.json")
    if any(digest(work / name) != expected for name, expected in manifest.items()):
        raise ValueError("Prepared scientific sources changed; use a new runtime directory")
    if args.action == "train":
        command = [sys.executable, "-B", str(work / "run_queue.py"), "--device", args.device,
                   "--concurrency", str(concurrency)]
        if args.dry_run:
            command.append("--dry-run")
    elif entry["evaluation_backend"] == "cpu":
        if args.coordinates is None:
            parser.error("Historical mu/E1-CC2 ABCD evaluation requires --coordinates")
        if args.device != "cuda:0":
            parser.error("Historical CPU ABCD includes the original CUDA cross-inference prerequisite")
        # The historical audit compares C against its earlier coordinate-substitution pass.
        subprocess.run([sys.executable, "-B", str(work / "cross_infer.py"),
                        "--device", args.device], cwd=work, check=True)
        command = [sys.executable, "-B", str(HERE / "common/evaluate_historical_abcd.py"),
                   entry["dataset"].lower(), "--source", str(work), "--coordinates",
                   str(args.coordinates.resolve()), "--out", str(work / "results")]
    else:
        if args.device != "cuda:0":
            parser.error("The archived 18-property evaluator requires CUDA; CPU would be a changed protocol")
        command = [sys.executable, "-B", str(work / "evaluate_abcd.py"), "--device", args.device]
    subprocess.run(command, cwd=work, check=True)


if __name__ == "__main__":
    main()
