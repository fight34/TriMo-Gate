"""Frozen bounded conformer-generation function; see SOURCE_PROVENANCE.json."""
from collections import Counter
import time
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import primitives as geometry

def generate(row, max_atoms):
    started = time.perf_counter()
    result = {"id": row["id"], "test_index": row["test_index"], "input_smiles": row["input_smiles"],
              "etkdg_available": False, "mmff_available": False, "mmff_status": "not_attempted",
              "embedding_attempts": [], "mmff_codes": [], "error": None}
    points = {}
    try:
        if not row["input_header_identity_equal"] or not row["composition_matches_dft"]:
            result["mmff_status"] = "source_identity_warning"
            return result, points
        mol = Chem.AddHs(Chem.MolFromSmiles(row["canonical_isomeric_smiles"]))
        result["length"] = mol.GetNumAtoms()
        assert result["length"] == row["dft_length"]
        assert dict(sorted(Counter(a.GetSymbol() for a in mol.GetAtoms()).items())) == row["dft_element_counts"]
        original_seed = geometry.seed_for("QM8", row["id"])
        for attempt in range(2):
            seed = (original_seed + attempt) % 2147483647
            params = geometry.embed_parameters(seed)
            params.useRandomCoords = bool(attempt)
            mol.RemoveAllConformers()
            begin = time.perf_counter()
            code = int(AllChem.EmbedMolecule(mol, params))
            result["embedding_attempts"].append({"attempt": attempt+1, "seed": seed,
                "useRandomCoords": bool(attempt), "code": code,
                "failure_counts": list(params.GetFailureCounts()), "wall_s": time.perf_counter()-begin})
            if code >= 0:
                break
        if code < 0:
            result["mmff_status"] = "embedding_failed"
            return result, points
        assert geometry.canonical(mol) == row["canonical_isomeric_smiles"]
        points["etkdg"] = geometry.point_features(mol, max_atoms)
        result["etkdg_available"] = True
        optimized = Chem.Mol(mol)
        if not AllChem.MMFFHasAllMoleculeParams(optimized):
            result["mmff_status"] = "unsupported_parameters"
            return result, points
        begin = time.perf_counter()
        props = AllChem.MMFFGetMoleculeProperties(optimized, mmffVariant="MMFF94")
        ff = AllChem.MMFFGetMoleculeForceField(optimized, props, nonBondedThresh=100.0)
        result["mmff_energy_before_kcal_mol"] = float(ff.CalcEnergy())
        for budget in (300, 700):
            code = int(ff.Minimize(maxIts=budget, forceTol=1e-4, energyTol=1e-6))
            result["mmff_codes"].append(code)
            if code != 1:
                break
        result["mmff_wall_s"] = time.perf_counter()-begin
        result["mmff_energy_after_kcal_mol"] = float(ff.CalcEnergy())
        assert np.isfinite([result["mmff_energy_before_kcal_mol"], result["mmff_energy_after_kcal_mol"]]).all()
        assert result["mmff_energy_after_kcal_mol"] <= result["mmff_energy_before_kcal_mol"] + 1e-6
        assert geometry.canonical(optimized) == row["canonical_isomeric_smiles"]
        optimized_points = geometry.point_features(optimized, max_atoms)
        assert np.array_equal(points["etkdg"][:, 4:], optimized_points[:, 4:])
        result["mmff_status"] = "converged" if code == 0 else "not_converged" if code == 1 else "optimization_failed"
        if code == 0:
            points["mmff"] = optimized_points
            result["mmff_available"] = True
        else:
            points["mmff_not_converged"] = optimized_points
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["pipeline_wall_s"] = time.perf_counter()-started
    return result, points
