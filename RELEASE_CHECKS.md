# Publication checks

## Dataset core

- Public Hub: ZhangQing123/cigAMT, CC BY-SA 4.0.
- 123 selected files, 29,365,062,865 bytes, remotely checked against local upload receipts.
- Large-file SHA-256 and small-file Git blob hashes match; original source sizes and modification times remain unchanged.
- Public file inventory: metadata/release_checksums.json on the dataset Hub.
- Original HDF5 waveform datasets remain unchanged. Two publication copies remove only a machine-path root attribute, with all arrays and other attributes compared.
- Thirty-six bounded replay comparisons passed against saved offline data.
- The published preview Parquet was downloaded again; all 32 embedded images decoded successfully.
- All four Dataset Viewer configurations are ready with no failed or pending splits; the 32-image preview endpoint was checked successfully.

## Benchmark code and weights

- All 28,138 sferic input rows match the historical benchmark arrays exactly.
- All 33 selected sferic neural checkpoints strictly match their architectures and passed bounded forward checks.
- The exported sferic TCN seed-42 asset was evaluated on all 2,488 test windows; primary metrics match the original run report.
- All 80 selected AMT checkpoints strictly match their architectures; nine model types passed bounded forward checks.
- Seven-task reader parity passed for 21 selected examples against the existing materialized offline views.
- Numerical impedance checks passed on known synthetic tensor fixtures. This is not a clean-reference validation of unlabeled field cases.
- Nine checkpoint archives passed ZIP integrity and SHA-256 checks. Model files are inference exports; original checkpoint files were not modified.
- These checks establish storage/interface compatibility. They do not claim fresh retraining, complete repetition of every historical experiment, or statistical significance.

## Website and source preparation

- Static link, eight-task/66-row/238-metric metadata, and all 14 copied image checksum checks pass.
- JavaScript syntax and 121 Python source parse checks pass; main documentation local links resolve.
- No Git commit, Git push, or GitHub Release publication is performed by preparation.

## Generation modules

GeoAMT-DiT architecture, training, sampling and evaluation source is included. Its 13 source files, CLI entry points, full input-size forward and bounded two-step DDIM checks passed. The historical pretrained reference-generator weight file was not found and is not included. Two run-matched Noise-DiT inference checkpoints are packaged with model/EMA tensors; exact historical NPZ sampling-checkpoint linkage is not recorded.

Both Noise-DiT generator checkpoints (step 200000) passed strict model and EMA state loading and four finite full-length forward checks; generation/training/evaluation CLI checks passed. Website visual inspection could not run because the browser interface could not reliably determine the current URL; static website checks passed.
