"""Small CPU checks; does not train a benchmark or download any data."""

import ast
import csv
import itertools
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from data_config import validate_xyz_map
from Model_1 import GatedFusion, MultimodalModel

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(1)


def source_functions(filename, names, namespace):
    """Load pure helpers without importing a top-level training loop."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(nodes) != len(names):
        raise AssertionError(f"Missing helpers in {filename}")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), namespace)
    return namespace


class ReleaseChecks(unittest.TestCase):
    def test_all_python_sources_compile(self):
        for path in ROOT.glob("*.py"):
            compile(path.read_text(encoding="utf-8-sig"), path.name, "exec")

    def test_full_cli_input_checks_for_both_datasets(self):
        from rdkit import Chem
        with tempfile.TemporaryDirectory(prefix="benchmark-input-check-") as directory:
            root = Path(directory)
            table = root / "labels.csv"
            mapping = root / "mapping.csv"
            smiles = ["C", "N", "O", "CF", "CCO", "C(C)CO"]
            with table.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["smiles", "target"])
                writer.writerows((smi, index + 1) for index, smi in enumerate(smiles))
            with mapping.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["row_index", "smiles", "xyz_file"])
                for index, smi in enumerate(smiles):
                    molecule = Chem.AddHs(Chem.MolFromSmiles(smi))
                    filename = f"molecule_{index}.xyz"
                    atoms = [f"{atom.GetSymbol()} {i} {i % 2} 0" for i, atom in enumerate(molecule.GetAtoms())]
                    (root / filename).write_text(f"{len(atoms)}\nsynthetic input-validation fixture\n" + "\n".join(atoms) + "\n", encoding="utf-8")
                    writer.writerow([index, smi, filename])
            for dataset in ["qm8", "qm9"]:
                process = subprocess.run([
                    sys.executable, "-B", str(ROOT / "run.py"), "--dataset", dataset,
                    "--table", str(table), "--smiles-column", "smiles", "--target", "target",
                    "--unit", "synthetic", "--xyz-dir", str(root), "--xyz-map", str(mapping),
                    "--output", str(root / "output"), "--check-inputs",
                ], capture_output=True, text=True, encoding="utf-8", errors="replace")
                self.assertEqual(process.returncode, 0, process.stdout + process.stderr)
                self.assertIn('"point_features": 9', process.stdout)
                self.assertIn('"status": "input_checks_passed"', process.stdout)

    def test_mapping_rejects_row_identity_and_reused_file(self):
        with tempfile.TemporaryDirectory(prefix="mapping-check-") as directory:
            root = Path(directory)
            (root / "a.xyz").write_text("fixture", encoding="utf-8")
            (root / "b.xyz").write_text("fixture", encoding="utf-8")
            rows = [dict(row_index="0", smiles="C", xyz_file="a.xyz"),
                    dict(row_index="1", smiles="N", xyz_file="b.xyz")]
            self.assertEqual(len(validate_xyz_map(["C", "N"], rows, root)), 2)
            with self.assertRaises(ValueError):
                validate_xyz_map(["N", "C"], rows, root)
            with self.assertRaises(ValueError):
                validate_xyz_map(["C", "N"], rows[::-1], root)
            rows[1]["xyz_file"] = "a.xyz"
            with self.assertRaises(ValueError):
                validate_xyz_map(["C", "N"], rows, root)
            rows[1]["xyz_file"] = "../outside.xyz"
            with self.assertRaises(ValueError):
                validate_xyz_map(["C", "N"], rows, root)
            rows[1]["xyz_file"] = "missing.xyz"
            with self.assertRaises(FileNotFoundError):
                validate_xyz_map(["C", "N"], rows, root)

    def test_xyz_centering_fortran_and_invalid_data(self):
        for dataset in ["qm8", "qm9"]:
            parser_name = f"parse_{dataset}_xyz_and_center"
            namespace = source_functions(dataset.upper() + "_xyz_to_3d_coords.py",
                                         ["fortran_float", parser_name], {"np": np, "re": re})
            with tempfile.TemporaryDirectory(prefix="xyz-check-") as directory:
                path = Path(directory) / "molecule.xyz"
                path.write_text("2\nfixture\nH 1.0*^0 0 0\nH 3.0 0 0\n", encoding="utf-8")
                coords, symbols = namespace[parser_name](path)
                np.testing.assert_array_equal(coords, np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float32))
                self.assertEqual(symbols, ["H", "H"])
                path.write_text("2\nfixture\nH nan 0 0\nH 3.0 0 0\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    namespace[parser_name](path)
                path.write_text("2\nfixture\nH 1 0 0\n", encoding="utf-8")
                with self.assertRaises(ValueError):
                    namespace[parser_name](path)

    def test_seven_modality_forward_backward_and_gate_mask(self):
        torch.manual_seed(42)
        graph = Batch.from_data_list([
            Data(x=torch.randn(2, 5), edge_index=torch.tensor([[0, 1], [1, 0]]), edge_attr=torch.randn(2, 3)),
            Data(x=torch.randn(1, 5), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 3))),
        ])
        smiles = torch.tensor([[1, 2, 3, 4, 5, 6, 0, 0], [1, 2, 3, 4, 5, 0, 0, 0]])
        points = torch.randn(2, 7, 9)
        lengths = torch.tensor([7, 5])
        for flags in itertools.product([False, True], repeat=3):
            if not any(flags):
                continue
            with self.subTest(flags=flags):
                model = MultimodalModel(10, 8, 32, 4, 1, 8, 5, 3, 32, 1, 9, 64, 32, 1, *flags)
                prediction = model(smiles, graph, points, lengths)
                self.assertEqual(tuple(prediction.shape), (2, 1))
                self.assertTrue(torch.isfinite(prediction).all())
                prediction.square().mean().backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                if sum(flags) > 1:
                    self.assertIsInstance(model.fusion_model, GatedFusion)
                    gates = torch.stack(model.fusion_model.gates, dim=1)
                    torch.testing.assert_close(gates.sum(dim=1), torch.ones(2, 32))
                    for index, enabled in enumerate(flags):
                        if not enabled:
                            self.assertEqual(torch.count_nonzero(gates[:, index]).item(), 0)

    def test_split_and_full_pool_retrieval(self):
        namespace = source_functions("random_3_seeds.py", ["random_split_indices"], {"np": np})
        for seed in [42, 44, 46]:
            splits = namespace["random_split_indices"](101, seed)
            self.assertEqual([len(part) for part in splits], [80, 10, 11])
            self.assertEqual(len(np.unique(np.concatenate(splits))), 101)
            self.assertTrue(all(np.all(part[:-1] < part[1:]) for part in splits))
        namespace = source_functions("random_3_seeds_cmcr_with_gate_export.py",
                                     ["l2_normalize_np", "top1_cross_modal_retrieval_np"], {"np": np})
        identity = np.eye(5)
        self.assertEqual(namespace["top1_cross_modal_retrieval_np"](identity, identity, chunk_size=2), 1.0)
        tied = np.ones((5, 4))
        self.assertEqual(namespace["top1_cross_modal_retrieval_np"](tied, tied, chunk_size=2), 0.2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
