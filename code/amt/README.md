# cigAMT: 150 Hz benchmark code

This directory contains the actual seven-task benchmark implementation used for
the AMT time-series experiments, adapted to read the released reference/noise
core. It includes training, validation-only checkpoint/threshold selection,
evaluation, classical comparisons, model definitions, and Welch–Huber impedance
estimation. The 24 kHz classification benchmark is in `../sferic/`.

## Installation and data

Use Python 3.10 or 3.11 and install a suitable PyTorch build first. Then, from
this directory:

```bash
python -m pip install -r requirements.txt
```

Download `amt_150hz/` from
[ZhangQing123/cigAMT](https://huggingface.co/datasets/ZhangQing123/cigAMT).
Pass its local path as `--data-root` in every command. It must contain
`manifest.json`, `clean/`, `noise/`, `metadata/` and `splits/`.

The publication adapter replays fixed epoch-0 task rows from shared reference
and noise banks. It does not require the historical approximately 890 GB
`offline/` exports. When complete offline shards are supplied, the original
direct-shard reader remains available. Both paths retain the same normalization,
fixed task schedules and exact global stratified selection. Fingerprints record
which reader was used; a replay run does not claim identical offline file hashes.
The adapter keeps training at epoch 0 to reproduce the materialized benchmark.

## Tasks and architectures

| Implementation | Models / tasks |
|---|---|
| `amtbench/models.py` | TCN, U-Net1D, BiGRU, Patch Transformer, AMTMaskFormer, Spectral FNO; seven tasks |
| `amtbench/models.py` | DLinear; forecasting |
| `specialist_baselines/vendor/patchtst/` | PatchTST; forecasting adapter |
| `specialist_baselines/vendor/saits/` | SAITS; interpolation and imputation adapters |
| `amtbench/classical.py` | Robust threshold, identity, wavelet, persistence, ridge AR, linear interpolation, mean fill, where applicable |

PatchTST and SAITS include their original license files and upstream/adaptation
records. The included RevIN implementation is additionally covered by
`specialist_baselines/vendor/patchtst/LICENSE-RevIN` (MIT, ETRI 2022), from
[ts-kim/RevIN](https://github.com/ts-kim/RevIN).
The SAITS adapter uses the benchmark's missing-position MSE, rather than
claiming reproduction of the original SAITS training objective. Patch Transformer
and PatchTST are distinct implementations. Consult the upstream records before
reusing vendored code; the dataset's CC BY-SA 4.0 license does not replace third-party
software licenses.

## Inspect, train and evaluate

Commands below are run from `code/amt/`. Substitute your own data and output
paths. Output paths should be outside the downloaded dataset.

```bash
python run.py inspect --data-root /path/to/amt_150hz --task denoising --split test --scheme A

python run.py train --data-root /path/to/amt_150hz --config configs/main_AC_bs256_budget_s42.json --task denoising --model tcn --scheme AC --seed 42 --output /path/to/new-run --eval-after

python run_specialists.py train --data-root /path/to/amt_150hz --config configs/supplement_patchtst_saits_s42.json --task forecasting --model patchtst --scheme none --seed 42 --output /path/to/new-patchtst-run --eval-after

python run_specialists.py train --data-root /path/to/amt_150hz --config configs/supplement_patchtst_saits_s42.json --task imputation --model saits --scheme none --seed 42 --output /path/to/new-saits-run --eval-after

python run.py evaluate --data-root /path/to/amt_150hz --run /path/to/pretrained/amt/denoising/tcn/seed_42 --schemes A C --device cuda:0 --tag publication_replay

python run_specialists.py evaluate --data-root /path/to/amt_150hz --run /path/to/pretrained/amt/forecasting/patchtst/seed_42 --schemes none --device cuda:0 --tag publication_replay
```

For CPU, add `--device cpu --precision fp32` when training. For evaluation use
`--device cpu`. `--max-samples` on evaluation is a bounded diagnostic, not a full
benchmark result. Full test evaluation is the default.

Historical runs used different budgets across groups and seeds. The checked-in
`configs/checkpoints/<task>/<model>/seed_N/training.json` records each available
checkpoint's actual resolved configuration; do not substitute one generic config
when reproducing a specific released checkpoint. For retraining, create a config
with this object's `config` as its top-level `training` value, or pass the matching
original suite config. `configs/main_AC_bs256_budget_s42.json` is an explicit
example budget and does not assert that every released model used that budget.

## Pretrained checkpoints

`weights_inventory.json` lists the selected 80 AMT checkpoints by model, task and
seed. Inference archives are prepared separately from the Git source: obtain them
through the project's pretrained-model links when published. Extracted runs contain
`best.pt`, `run_config.json`, and `calibration.json`. They retain model weights and
architecture/configuration metadata, while optimizer/RNG state is removed.
The export is inference-only; it is not a training-resume checkpoint.

Original machine paths are not needed. `publication_export: true` explicitly
identifies portable evaluation copies; evaluation still strictly checks model
state keys and dataset identity. It does not silently equate changed publication
code fingerprints with the historical package. All 80 selected original weights
were checked with restricted deserialization and strict architecture loading.

## Metrics and output

Evaluation preserves the original implementation in `metrics.py` and `engine.py`:
classwise classification, point/event localization, overall and bandwise denoising,
and errors computed only at hidden positions for missing-data tasks. Thresholds are
selected on validation and frozen for testing. A/C are evaluated separately; shared
clean controls are included only once when training with AC. Aggregation writes
per-case rows and summaries. A seed standard deviation is descriptive; the code
does not invent missing repetitions.

## Impedance estimation

```bash
python impedance/run_impedance.py --input /path/to/physical_components.npy --output /path/to/impedance/case --frequencies 5 10 20 30
```

Input is a finite NumPy array `[4,T]` ordered `Ex,Ey,Hx,Hy`, sampled at 150 Hz.
Use original physical-amplitude coordinates, undoing stored per-component
normalization before calculating impedance. Default `--unit si` expects E in V/m
and H in A/m. `--unit field` expects E in mV/km and magnetic values in nT and uses
the corresponding apparent-resistivity factor. Merely normalized arrays cannot
establish calibrated apparent resistivity.

The implementation retains Hann-window Welch coefficients, elementwise complex
Huber spectral averaging, tensor conditioning checks and multiple coherence.
Short records may reject requested frequencies with insufficient Welch segments.
Requested and realized FFT frequencies and rejection reasons are saved. Long
records use weighted delete-group jackknife diagnostics. This CLI is the 150 Hz
Welch–Huber extension; it does not replace the separate 24 kHz sferic-selection
case estimator.

## Publication adaptations

`source_fingerprints.json` records the source modules before portability edits.
The changes are confined to the shared-core reader adapter, portable entry points,
restricted checkpoint loading, and removal of development-drive fallbacks.
Model architectures, task losses, numerical metrics and impedance numerical
functions are retained. No raw recordings, task arrays, training logs or large
weights are bundled in this source directory.
