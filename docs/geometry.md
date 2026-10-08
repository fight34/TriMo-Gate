# Geometry source comparison: QM9 and QM8

This directory releases the scientific code for eight QM9 and twelve QM8
geometry comparisons. It is separate from both the original modality ablations
and six-fusion comparisons. Each property has six training runs: DFT and MMFF
coordinates, each with seeds 42, 2025, and 3407. A/B/C use the **same selected
DFT checkpoint** per seed; D is trained independently with MMFF coordinates.

| Configuration | Training coordinates | Test coordinates |
|---|---|---|
| A (historical label A_prime) | DFT | DFT |
| B (B_prime) | DFT | ETKDGv3 without subsequent force-field optimization |
| C (C_prime) | DFT | ETKDGv3 + MMFF94 |
| D | ETKDGv3 + MMFF94 | ETKDGv3 + MMFF94 |

All reference quantum-chemical labels are retained. These experiments predict
the original properties from alternative geometric inputs; they do not
recompute quantum-chemical properties on generated conformers. They use no
CMCR. The original nine point channels are centered x/y/z in angstrom, radius,
and one-hot C/F/H/N/O, including hydrogens. Padding is 29 atoms for QM9 and 26
for QM8. SMILES, graphs, labels, and retained molecular IDs remain fixed.

## Data inputs and coordinate preparation

First construct or obtain the **original audited DFT cache** described in
[the fusion cache documentation](fusion.md#cache-contract-and-preparation).
It must contain `cache.npz`, `metadata.json`, and `manifest.csv`. QM9 also
requires the original individual XYZ files. QM8 requires the author-provided
multi-record `XYZ_B3LYP_631G2dfp.xyz`, with the exact verified record mapping.
The historical original and matched split-ID JSON files are included here.
Large caches, raw data, generated coordinates, predictions and weights are
not committed. In particular, the historical QM8 audit inputs described in
the fusion documentation are required; this repository alone does not supply
missing dataset records or reconstruct an unknown mapping.

Use the separate CPU environment in
`environments/geometry-generation-requirements.txt`. For example, from the
repository root:

```sh
python geometry/generation/prepare.py --dataset QM9 --cache /data/qm9_cache --xyz /data/QM9 --out /data/qm9_generated
python geometry/generation/prepare.py --dataset QM8 --cache /data/qm8_cache --xyz /data/qm8_cache/official_source/XYZ_B3LYP_631G2dfp.xyz --out /data/qm8_generated
```

Run once per dataset, not once per property. Both output directories must be
new. The source checks verify XYZ identity, elemental composition, cached
graphs and point features; QM9 also checks SMILES tokens. The generated
`data/` directory contains paired DFT/MMFF caches, filtered fixed splits,
`etkdg_test.npz`, and QM9 identity categories. Full coordinates and failure
records remain outside `data/` for the historical B evaluation. This command
is a portable replacement for the original staged local orchestration; the
per-molecule numerical generation functions are extracted unchanged.

One ETKDGv3 conformer is attempted per molecule with one thread,
`maxIterations=1000`, and chirality enforcement. The deterministic seed is
the first four big-endian bytes of SHA256 of
`geometry-pilot-v1|20260928|DATASET|ID`, modulo 2147483647. Only embedding
failure triggers one `useRandomCoords=True` retry with seed+1. MMFF94 starts
from a copy of that conformer, uses 300 iterations plus 700 more only if not
converged, `forceTol=1e-4`, `energyTol=1e-6`, and `nonBondedThresh=100`.
Accepted conformers must converge, retain molecular identity/composition,
have finite valid features, and not increase force-field energy by more than
1e-6 kcal/mol. Failed geometries remain unavailable; no DFT/UFF replacement,
DFT alignment, or conformer energy-selection ensemble is used. There is no
claimed DFT-to-generated atom-index correspondence.

The retained QM9 population is 132486 of 133885 records, with train/validation/
test counts 105992/13251/13243. The original fixed random split is filtered;
it is **not** a structural-group split, and the official QM9 geometry-exclusion
list was not applied. QM8 retains 21689 of the 21747 audited source records,
with counts 17356/2164/2169 and the original structural groups preserved.
The 39 earlier QM8 omissions precede this geometry comparison. All properties
reuse the same generated coordinates and successful cohort. If regeneration
changes the success set, preparation stops rather than silently changing the
published cohort. RDKit version/platform differences must then be investigated.

## Prepare a property, train, and evaluate

The archived scientific files are materialized into an empty directory before
running. This keeps the shared model byte-identical while allowing each
property's original target-specific loader and trainer to remain intact.
The common model is stored once in `fusion/Model.py`.

```sh
python geometry/run.py prepare --experiment qm9_alpha --data /data/qm9_generated/data --work-dir /results/qm9_alpha
python geometry/run.py train --experiment qm9_alpha --work-dir /results/qm9_alpha --dry-run
python geometry/run.py train --experiment qm9_alpha --work-dir /results/qm9_alpha
python geometry/run.py evaluate --experiment qm9_alpha --work-dir /results/qm9_alpha
```

`prepare` checks cache hashes and the exact matched split before creating a
runtime. It hardlinks the two large cache files by default (same filesystem
required); use `--copy-data` explicitly when copying is necessary and space
is available. It declares an absent property unit from the frozen protocol
but rejects a conflicting existing unit. Fresh-runtime metadata/protocol
hashes are recorded separately from the original archive. Do not mutate
linked caches. `train` runs the bounded original queue and finalizes all six
models only after all six trainings finish. It does not connect to a server,
rent resources or shut down a machine. The queue resumes its own valid
checkpoints; never mix historical and new-release run directories.

The historical mu and E1-CC2 queues allow at most four concurrent runs;
later queues allow six. Defaults retain those limits. `--concurrency` may
reduce concurrency for smaller hardware without changing batch size. Keep
FP32, batch 128, AdamW 2e-4, weight decay 1e-4, gradient clipping 0.5,
200-epoch maximum, 50-epoch minimum, patience 30, and train-only target
standardization. A and D use the same prescribed settings without D-specific
retuning. Validation MAE selects each best checkpoint. A cumulative 420-minute
limit applies per run; a time-limit failure is not a completed experiment.

The **historical complete mu and E1-CC2 ABCD tables were CPU reevaluations**
after B was added. Their released workflow preserves the earlier CUDA
cross-inference prerequisite, then runs the archived CPU A/B/C/D evaluation:

```sh
python geometry/run.py evaluate --experiment qm9_mu --work-dir /results/qm9_mu --coordinates /data/qm9_generated
python geometry/run.py evaluate --experiment qm8_e1cc2 --work-dir /results/qm8_e1cc2 --coordinates /data/qm8_generated
```

Thus these two commands still require CUDA for the prerequisite. The other
18 properties evaluate all four conditions on `cuda:0`. CPU-only diagnostic
checks are available below, but do not reproduce the formal CUDA evaluation.
The historical and newer result tables should not be described as sharing
exactly the same evaluation backend. Re-running an existing evaluation refuses
to overwrite the committed output. Read `results/comparison.json` for
per-seed MAE/RMSE, means, and sample SD; three seeds alone do not establish
statistical significance. Paired per-seed percentage changes and a percentage
computed from aggregate mean errors are distinct summaries.

## Targets and units

`python geometry/run.py list` prints the complete registry. QM9 names are
`qm9_mu`, `qm9_alpha`, `qm9_homo`, `qm9_gap`, `qm9_r2`, `qm9_zpve`,
`qm9_u0atom`, and `qm9_cv`. QM8 names follow `qm8_e1cc2`, `qm8_f2pbe0`, etc.

| QM9 target | Original reporting unit |
|---|---|
| mu | Debye |
| alpha | a0^3 |
| HOMO and gap | hartree |
| electronic spatial extent R2 | a0^2 |
| ZPVE | hartree |
| U0_atom = E_molecule − sum of atomic reference energies | kcal mol^-1 |
| Cv | cal mol^-1 K^-1 |

Gap uses the dataset's original `gap` column, not a recomputed LUMO−HOMO.
U0_atom uses zero-based target column 22, not raw molecular U0. For optional
eV reporting, multiply both mean and SD of hartree metrics by 27.211386246.
QM8 excitation energies are hartree; oscillator strengths are dimensionless.
PBE0 uses the original def2SVP columns and CAM-B3LYP the original def2TZVP
columns. Do not substitute similarly named labels.

## Provenance and validation

`geometry/SOURCE_PROVENANCE.json` records original paths relative to the
research workspace, original SHA256, release SHA256, and explicit adaptations.
Training, model, queue and loader code are preserved byte-for-byte. Identical
files are stored once and copied to their expected runtime names. The three
mu/alpha/E1-CC2 source directories come from local completed-runtime backups;
the other 17 source directories are local deployment/preparation snapshots.
Their registry/manifest excerpts say so explicitly: a preparation manifest
alone is not proof that every byte equals a final remote runtime. Historical
`prepared` fields describe when a protocol was frozen, not current experiment
completion. Server orchestration, SSH credentials, shutdown scripts, and
machine management are excluded.

```sh
python -B -m unittest discover -s geometry/tests -p test_release.py
python -B geometry/tests/smoke_training.py --experiment qm9_mu --data /data/qm9_generated/data/dft --out /tmp/qm9_diagnostic
```

The first command checks all twenty target/unit configurations, exact source
fingerprints, fixed-split consistency, compilation and default queue dry-runs.
The optional CPU smoke performs one tiny diagnostic training step, resumes its
completed checkpoint, and verifies checkpoint reload and held-out diagnostic
evaluation. It uses reduced model dimensions and never starts a formal run.
Full GPU training and cross-platform numerical replication are not implied
by these local release checks.
