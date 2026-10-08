"""Portable entry point for the supplied historical benchmark implementations."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["qm8", "qm9"], required=True)
    parser.add_argument("--table", type=Path, required=True, help="CSV or XLSX label table")
    parser.add_argument("--smiles-column", required=True)
    parser.add_argument("--target", required=True, help="Exact target column name; no unit conversion")
    parser.add_argument("--unit", required=True, help="Unit of the supplied target values")
    parser.add_argument("--xyz-dir", type=Path, required=True)
    parser.add_argument("--xyz-map", type=Path, required=True, help="CSV with row_index,smiles,xyz_file")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modalities", nargs="+", choices=["S", "G", "P"], default=["S", "G", "P"])
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--cmcr", action="store_true", help="Full three-branch CMCR and gate/alignment export")
    group.add_argument("--bayes-opt", action="store_true", help="Optional 15-trial-per-seed Optuna implementation")
    parser.add_argument("--check-inputs", action="store_true", help="Validate all modalities without training")
    args = parser.parse_args()
    modalities = [name for name in ["S", "G", "P"] if name in args.modalities]
    if len(modalities) != len(args.modalities):
        parser.error("Do not repeat modality names.")
    if args.cmcr and modalities != ["S", "G", "P"]:
        parser.error("The historical CMCR export entry point requires all three modalities.")
    tokens = {"S": "1D", "G": "2D", "P": "3D"}
    suffix = "+CMCR" if args.cmcr else "_BayesianOptimization" if args.bayes_opt else ""
    result_dir = args.output.resolve() / ("RandomSplit_3_Seeds_" + "+".join(tokens[m] for m in modalities) + suffix + "_Results")
    config = vars(args).copy()
    config.update(modalities=modalities, result_dir=str(result_dir))
    for name in ["table", "xyz_dir", "xyz_map", "output"]:
        config[name] = str(config[name].resolve())
    os.environ["TRIMOGATE_BENCHMARK_CONFIG"] = json.dumps(config)

    from data_config import load_inputs
    frame, _, _, _ = load_inputs()
    if args.check_inputs:
        import Graph_dataloader
        import SMILES_dataloader_atom
        import importlib
        points = importlib.import_module(args.dataset.upper() + "_xyz_to_3d_coords")
        lengths = [len(frame), len(Graph_dataloader.graphs), len(SMILES_dataloader_atom.indices), len(points.point_cloud)]
        if len(set(lengths)) != 1:
            raise ValueError(f"Modality length mismatch: {lengths}")
        print(json.dumps({"rows": len(frame), "point_features": points.d_input, "status": "input_checks_passed"}))
        return
    if result_dir.exists() and any(result_dir.iterdir()):
        parser.error("The run directory is not empty; select another --output to preserve existing results.")
    result_dir.mkdir(parents=True, exist_ok=True)
    config["input_sha256"] = {name: hashlib.sha256(Path(config[name]).read_bytes()).hexdigest() for name in ["table", "xyz_map"]}
    config["seeds"] = [42, 44, 46]
    (result_dir / "run_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    script = "Train_and_evaluation.py" if args.bayes_opt else "random_3_seeds_cmcr_with_gate_export.py" if args.cmcr else "random_3_seeds.py"
    runpy.run_path(str(Path(__file__).with_name(script)), run_name="__main__")


if __name__ == "__main__":
    main()
