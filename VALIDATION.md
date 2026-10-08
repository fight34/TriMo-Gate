# Publication validation

Checks performed on 2026-10-08 during preparation of this code release:

| Component | Completed checks |
| --- | --- |
| Benchmark | Six CPU checks covering all seven modality combinations, forward/backward computation, input-map failures, QM8/QM9 table and XYZ loading, split handling and retrieval ties |
| Fusion | Five release tests including the existing protocol tests; both trainers' six-fusion CPU self-tests; both 18-job queue dry-runs; six cache-tool help commands |
| Geometry | Twenty target/unit/source/split configurations and default queue dry-runs; retained conformer functions compared against original outputs on three real molecules per dataset; a real-cache one-step diagnostic training/resume/finalization; runtime preparation on an existing matched cache |
| Packaging | Python syntax, JSON parsing, a file-type/size inventory, and scans for machine-specific addresses and common credential patterns |

The model and original scientific function definitions were compared with their source snapshots. Portable paths, explicit input validation and the documented benchmark default-fusion selection are release adaptations. File-level provenance and component validation records provide the detailed scope.

The local CPU checks used Python 3.11.4, PyTorch 2.6.0+cu124, PyG 2.5.3, torch-scatter 2.1.2+pt26cu124, NumPy 2.1.2 and RDKit 2026.03.3. This is the **packaging-test environment**, not a replacement claim about the historical server environment. Historical stacks are recorded separately.

These checks did not rerun formal training, the full geometry-generation population, the optional Bayesian search, or all manuscript figures and Excel exports. They do not establish full numerical reproduction, historical table provenance, or license ownership. Test-generated caches, logs and checkpoints are excluded from the repository.
