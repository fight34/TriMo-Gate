# TriMo-Gate

Research code accompanying **TriMo-Gate: Multimodal Fusion and Sensitivity to Input Geometry in Quantum-Chemical Property Prediction** by Mengxin Yang.

TriMo-Gate combines SMILES, molecular graph, and molecular point-cloud representations for property prediction. This repository collects the relevant QM9 and QM8 implementations, fusion comparisons, cross-modal contrastive regularization (CMCR), and geometry-quality experiments. It excludes unrelated classification projects, cloud administration scripts, private credentials, datasets, and trained weights.

## Choose the appropriate experiment

| Directory | Purpose | Protocol |
| --- | --- | --- |
| `benchmark/` | Original modality ablations and CMCR, including gate and alignment exports | Historical seeds 42, 44, 46; see [benchmark guide](docs/benchmark.md) |
| `fusion/` | Six fusion implementations and archived controlled comparisons | Seeds 42, 2025, 3407; see [fusion guide](docs/fusion.md) |
| `geometry/` | Conformer generation and A/B/C/D evaluation for eight QM9 and twelve QM8 targets | Seeds 42, 2025, 3407; see [geometry guide](docs/geometry.md) |

These experiment families have different data and evaluation protocols. Their scores must not be substituted for one another. The guides describe the available source snapshots, publication adaptations, required inputs, and verification limits.

## Getting started

1. Create a separate Python environment for the experiment family you need. Read its guide before installing dependencies. PyTorch, CUDA and compiled PyG extensions must be mutually compatible; the coordinate-generation RDKit environment is recorded separately.
2. Obtain QM9 or QM8 from the original dataset sources and prepare the inputs described in the corresponding guide. No commercial quantum-chemistry software is needed for the alternative-conformer generation step.
3. Inspect the command line with `--help`, run the documented small checks, and then launch the required training or evaluation command. Training may require a CUDA GPU; conformer generation runs on the CPU.

**Input availability:** this is a code release, not a self-contained numerical reproduction bundle. Exact historical runs require the specified prepared tables, molecule mappings and caches. In particular, replacing the author's original QM8 SMILES strings with another canonically equivalent serialization changes the token input. The guides state which prepared inputs are not distributed here; downloading an arbitrary QM8/QM9 CSV is not sufficient to reproduce an archived run unchanged.

There is deliberately no single command that launches every experiment on import. Outputs are written to explicitly selected run directories. Never load an untrusted PyTorch checkpoint: checkpoint deserialization may execute code.

## Geometry configurations

| Configuration | Training coordinates | Test coordinates |
| --- | --- | --- |
| A | Dataset-provided DFT | Dataset-provided DFT |
| B | Same trained checkpoint as A | ETKDGv3 without subsequent force-field optimization |
| C | Same trained checkpoint as A | ETKDGv3 followed by MMFF94 |
| D | ETKDGv3 followed by MMFF94 | ETKDGv3 followed by MMFF94 |

A–C share the selected DFT-trained checkpoint for each property and seed. D is trained independently. Comparisons use the common valid molecule subset and retain the original quantum-chemical labels; those labels are **not** recomputed for generated conformers. The nine point features and the SMILES/graph inputs are held consistent within each comparison. QM9 uses the retained fixed random split, whereas the geometry QM8 experiments use the retained structural-group split.

Report native units and the mean and **sample** standard deviation over the three seeds. The original QM9 geometry-exclusion list was not applied in these geometry experiments. Historical evaluation-backend differences are documented in the geometry guide. Three seeds alone do not establish statistical significance or equivalence.

## Data and attribution

- QM9: Ramakrishnan et al., *Scientific Data* **1**, 140022 (2014), [doi:10.1038/sdata.2014.22](https://doi.org/10.1038/sdata.2014.22).
- QM8: Ramakrishnan et al., *The Journal of Chemical Physics* **143**, 084111 (2015), [doi:10.1063/1.4928757](https://doi.org/10.1063/1.4928757).
- Dataset overview: [MoleculeNet](https://moleculenet.org/datasets-1).

Please cite the datasets and relevant methods/software as well as the associated manuscript. `CITATION.cff` identifies this code project; no publication DOI or acceptance status is implied.

## Reproducibility and licensing

Source hashes and the documented release adaptations distinguish the collected experiment code from portable publication copies. Small tests check implementation behavior and packaging; they are not a rerun of the full published experiments. See [reproducibility notes](docs/reproducibility.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

No project-wide reuse license has yet been selected by the author. Public availability alone does not grant an open-source license. Third-party dependencies retain their own licenses.
