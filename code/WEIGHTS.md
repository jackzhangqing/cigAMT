# Pretrained models

The nine checkpoint archives are available in the [weights-v1 GitHub Release](https://github.com/jackzhangqing/cigAMT/releases/tag/weights-v1): 128 downstream inference assets and two Noise-DiT generation checkpoints, totaling 1,514,617,640 bytes. All nine uploaded archives were verified against the manifest by size and SHA-256. Release attachments are distributed separately from Git source.

## Archive inventory

| Archive | Inference assets | Size (MiB) |
|---|---:|---:|
| `cigamt-sferic-checkpoints.zip` | 48 | 247.6 |
| `cigamt-amt-forecasting.zip` | 16 | 152.0 |
| `cigamt-amt-imputation.zip` | 14 | 117.7 |
| `cigamt-amt-interpolation.zip` | 14 | 117.6 |
| `cigamt-amt-denoising.zip` | 12 | 115.5 |
| `cigamt-amt-frequency-denoising.zip` | 12 | 115.6 |
| `cigamt-amt-noise-classification.zip` | 6 | 57.8 |
| `cigamt-amt-noise-localization.zip` | 6 | 57.8 |

Each ZIP contains model tensors/estimators with the configuration and validation calibration needed for inference. Training-resume optimizer/RNG state and machine-local paths are excluded from neural exports. The manifest records every member and SHA-256 hash.

- Sferic: 11 neural architectures x 3 seeds (42, 2022, 2026), plus four feature classifiers and the prior estimator x 3 seeds.
- AMT: denoising, frequency-component denoising, forecasting, interpolation and imputation have selected checkpoints for seeds 42/43. Noise classification and localization currently have selected main-run checkpoints for seed 42 only; missing repetitions are not invented.

## Generation checkpoints

`cigamt-noise-generators.zip` contains two additional Noise-DiT checkpoints (multilength and fixed-2048), each with model and EMA tensors plus portable configuration. Its size is 462.9 MiB. These checkpoints match the generating runs/configurations, but the saved NPZ metadata does not identify the exact historical sampling checkpoint step or EMA choice. See [noise_generation/README.md](noise_generation/README.md) for sampling and training commands.

The GeoAMT-DiT reference-generator checkpoint referenced by the two released 10k outputs was not found in the reviewed directories. Its code and architecture are included in [reference_generation/](reference_generation/README.md); its pretrained weights are not included or claimed available. DDPM and DDIM reference outputs use two samplers of one generator.

## Download and verify

List the available archives or download and verify a selected archive:

```bash
python tools/download_weights.py --list
python tools/download_weights.py --asset cigamt-sferic-checkpoints.zip --output ./downloaded_weights
```

The helper checks the ZIP SHA-256 and extracts paths safely. The extracted structure is `sferic/<model>/seed_N/{model.pt,release_run.json}` or `amt/<task>/<model>/seed_N/{best.pt,run_config.json,calibration.json}`. Keep these companion files together.

## Evaluate

Run from `code/sferic/`:

```bash
python evaluate_pretrained.py --data-root ../../data/sferic_24khz --run-dir ../../downloaded_weights/sferic/tcn/seed_42 --device cpu --output ../../outputs/tcn_seed42_test.json
```

Run from `code/amt/`:

```bash
python run.py evaluate --data-root ../../data/amt_150hz --run ../../downloaded_weights/amt/denoising/tcn/seed_42 --schemes A C --device cpu --tag publication_replay
python run_specialists.py evaluate --data-root ../../data/amt_150hz --run ../../downloaded_weights/amt/forecasting/patchtst/seed_42 --schemes none --device cpu --tag publication_replay
```

Neural files use restricted PyTorch deserialization and strict architecture loading. The sklearn `estimator.joblib` assets are pickle-based; use the documented compatible environment and explicit `--trust-joblib` only for trusted official assets. See [sferic instructions](sferic/README.md) and [AMT instructions](amt/README.md) for architecture and data preprocessing details.

Checkpoints are inference exports, not resume checkpoints. Original source hashes are retained in the preparation reports, and originals remain unchanged. The publication-port validation does not claim fresh retraining or that every historical metric has been independently reproduced.
