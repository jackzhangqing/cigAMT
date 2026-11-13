# GeoAMT-DiT reference-signal generation

This is the supplied implementation for **four-component reference-signal
generation**, separate from structured-noise generation and the downstream
benchmark models. It includes the actual model, diffusion scheduler, EMA,
source-data reader, training script, sampling script, and generation-quality
evaluation pipeline. Publication copies remove machine-specific path defaults
and use `weights_only=True` for PyTorch checkpoint loading.

## What produced the released 20,000 windows

The two reviewed output files each contain 10,000 `float32 [4,2048]` windows:

| Output | Sampler | Steps | Sampling seed | Weight set |
|---|---|---:|---:|---|
| `geoamt_seed42_DDIM_10000.h5` | DDIM, eta=0 | 100 | 42 (filename) | EMA |
| `geoamt_seed44_DDPM_10000.h5` | DDPM | 1,000 | 44 (filename) | EMA |

Both HDF5 attributes reference **the same** `exp_v1` generator checkpoint and
architecture configuration. DDPM and DDIM are two sampling procedures, not
two independently trained generators. The sampling seeds above come from the
saved filenames; they are not dedicated seed attributes in those HDF5 files.

**Pretrained generator checkpoint status:** the checkpoint named by those
outputs was not found in the reviewed local generation directories. The
architecture and code are available here; the original generator weights are
not bundled or claimed to be downloadable. The existing generated data can be
used without this checkpoint. Recovering the exact EMA checkpoint is necessary
to regenerate those samples; architecture code alone cannot reproduce them.

## Architecture and diffusion

`model.py:GeoAMT_DiT` consumes a noisy `[B,4,2048]` sequence and diffusion step
and predicts noise of the same shape. The recorded configuration uses patch
size 8, hidden dimension 384, 12 DiT blocks, six heads, and MLP ratio 4.
`diffusion.py:GaussianDiffusion` implements the 1,000-step linear schedule,
forward perturbation, noise-prediction objective, DDPM and DDIM reverse paths.
`ema.py` maintains the inference EMA state. The reverse scheduler clamps its
estimated normalized clean signal to `[-5,5]`, as in the supplied implementation.

`configs/generator_linear.json` preserves the recorded hyperparameters with
portable path placeholders. The original training seed is 42. This JSON is a
configuration record; `train.py` accepts the explicit CLI flags shown below.

## Install and sample when weights are available

```bash
python -m pip install -r requirements.txt
python sample.py --help
```

The original sampling checkpoint schema is `config` plus tensor state dict
`ema` (preferred) or `model`. `export_generator.py` safely prepares a new
inference-only EMA copy, removing optimizer/scaler state and private paths:

```bash
python export_generator.py --source /path/to/recovered/latest.pt --destination /path/to/new/geoamt_dit_ema.pt
python sample.py --ckpt /path/to/new/geoamt_dit_ema.pt --n 10000 --batch_size 64 --num_steps 100 --seed 42 --device cuda --out_path /path/to/output/geoamt_ddim.h5
python sample.py --ckpt /path/to/new/geoamt_dit_ema.pt --n 10000 --batch_size 64 --num_steps 1000 --seed 44 --device cuda --out_path /path/to/output/geoamt_ddpm.h5
```

These show the recorded sampler settings and input contract. They do not claim
bitwise reproduction: the actual original checkpoint, sampling batch size,
environment, and hardware are also relevant. Use `--device cpu` for CPU
sampling, which is substantially slower. Outputs use HDF5 dataset `data` in
`Ex,Ey,Hx,Hy` order; they remain normalized samples, not calibrated field
measurements. Generated reference windows are used only in benchmark training.

## Training input and provenance boundary

The supplied training reader expects source-specific prepared files:

```text
generator_sources/
  real/clean_normalized.h5
  syn_amt_A/clean_normalized.h5
  syn_amt_B/clean_normalized.h5
  syn_mt/clean_normalized.h5
```

Each file must contain finite normalized `data [N,4,2048]` at 150 Hz. Optional
attributes and arrays preserve source/model identifiers. The training sampler
balances real against combined physical sources at ratio 1:1 and applies the
supplied time reversal and x/y exchange augmentation. Prepared sources must
use the intended normalization convention; do not independently standardize
components or substitute arbitrary files without reviewing physical effects.

To fit a new generator, supply an explicit NPZ split manifest containing the
zero-based source-local arrays `real_train`, `syn_amt_A_train`,
`syn_amt_B_train`, and `syn_mt_train`:

```bash
python train.py --data_root /path/to/generator_sources --split_indices /path/to/verified/split_indices.npz --use_sources real syn_amt_A syn_amt_B syn_mt --out_dir /path/to/new/training_output --beta_schedule linear --mixed_precision fp16 --batch_size 128 --num_steps 200000 --seed 42
```

This is an input contract for **new training**. The compact benchmark release
does not by itself reconstruct the original generator fitting pool, its
normalization provenance, or the exact historical fitting split. The stored
historical configuration does not establish train/test source isolation.
Do not infer such isolation merely because generated outputs are restricted
to downstream training. Exact retraining requires the original prepared
source pools and verified fitting manifests. No training was performed during
publication preparation.

## Generation-quality evaluation

The actual evaluation entry is `evaluate_samples.py`, with its supporting
`evaluation/` package. Its inputs are HDF5 dataset `data` or NPY arrays in
`[N,C,T]` or `[N,T,C]` layout. Specify an independent reference bank and
generated banks explicitly:

```bash
python -m pip install -r requirements-evaluation.txt
python evaluate_samples.py --real /path/to/independent/reference.h5 --generated DDIM=/path/to/geoamt_ddim.h5 --generated DDPM=/path/to/geoamt_ddpm.h5 --sampling-rate 150 --max-samples 5000 --skip-neural --skip-tsne --output /path/to/new/evaluation
```

This runs handcrafted feature/distribution, spectral, coherence, channel
correlation, coverage, and nearest-neighbour diagnostics. Remove
`--skip-neural` to fit the LDS/LPS evaluation probes with an appropriate
PyTorch environment; those probes are evaluation models, not pretrained
GeoAMT-DiT weights. Context-FID additionally needs the separately obtained
official TS2Vec repository and the specified encoder checkpoint/settings
(`--ts2vec-repo`, `--ts2vec-checkpoint`); it is skipped when absent and must not
be confused with AMT-feature FID. The TS2Vec implementation is not vendored.
Primary method attribution and metric distinctions are retained in
`PAPER_METRIC_NOTES` and the evaluation implementation.

Results from different reference banks, encoder fits, sample counts, or
preprocessing settings are different protocols. The evaluator does not make
historical screenshot values into a unified common-protocol ranking. Similarity,
coverage, and waveform plausibility alone do not establish physical validity or
rule out training-set memorization.
