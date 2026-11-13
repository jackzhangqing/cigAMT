# Conditional Noise-DiT

Actual Noise-DiT architecture, single-process/DDP training, DDPM/DDIM sampling,
procedural synthesis and generated-noise evaluation code, copied from the authors'
noise-generation implementation. Original experiment directories are unchanged.

## Files and variants

| File | Purpose |
|---|---|
| `noise_model.py`, `model.py` | Conditional 1D DiT and its transformer blocks |
| `diffusion.py`, `ema.py` | Gaussian diffusion and EMA weights |
| `noise_dataset.py` | NPY template loading, RMS normalization, support embedding and augmentation |
| `train_noise.py`, `train_noise_ddp.py` | Original training loops |
| `train_from_config.py` | Portable saved-config entry point |
| `sample_noise.py` | Class/length-conditioned DDPM or DDIM sampling |
| `generate_synthetic_noise.py` | Procedural noise construction |
| `evaluate_noise.py`, `noise_eval_*.py` | Descriptor-space distribution and similarity diagnostics, figures |

The multi-length configuration uses hidden width 384, 12 blocks, 6 heads,
8-sample patches and length-ratio conditioning. The fixed-2048 configuration uses
the same architecture without length conditioning. Both produce `[N,1,2048]`
waveforms with ten class labels. Source conditioning is disabled in the supplied
configs. A requested support length is a conditioning label; the sampler saves
the complete 2,048-sample generated waveform.

The released noise bank contains 600,000 generated multi-length templates and
50,000 generated fixed-length templates from the corresponding runs. The original
NPZ files identify the run/configuration but do not record the exact checkpoint
step or EMA choice used for sampling. The available step-200,000 final checkpoints
are therefore supplied as run-matched generation checkpoints, not as a claim of
bitwise regeneration of those saved templates.

## Setup and training inputs

Use Python 3.10/3.11, install an appropriate PyTorch build and then:

```bash
python -m pip install -r requirements.txt
```

Training consumes one-dimensional `.npy` clips in the original layout:

```text
noise_data/
  Real-Len100/Pulse/*.npy
  Real-Len150/Quasi-Sinusoidal/*.npy
  Real-2048/Composite/*.npy
  Synthetic-Norm/
    Synthetic-Len100/Pulse/*.npy
    Synthetic-Len2048/Composite/*.npy
```

Ten folder labels are `Pulse`, `Quasi-Sinusoidal`, `Square-wave`, `Step-wave`,
`Triangular-wave`, `Power-frequency`, `White-noise`, `Color-noise`, `Harmonic`,
and `Composite`. The trainer normalizes and embeds clips according to its config;
their valid support must be no longer than 2,048 samples.

To derive this layout from the Hugging Face core without changing it:

```bash
python prepare_noise_npy.py --data-root /path/to/amt_150hz --output /path/to/new/noise_data
```

This conversion includes only field-extracted and procedural templates assigned
to the released **training** partition. It excludes generated templates and
validation/test templates. It provides a portable training subset, not an exact
reconstruction of the historical generators' fitting membership. The historical
fixed-length run referenced a separate `Synthetic-Norm3` fitting collection;
that collection is not bundled here.

## Train

Run each command from `code/noise_generation/` with a new output directory:

```bash
python train_from_config.py --config configs/multilength.json --data-root /path/to/noise_data --synthetic-root /path/to/noise_data/Synthetic-Norm --output /path/to/new/multilength_run

python train_from_config.py --config configs/fixed2048.json --data-root /path/to/noise_data --synthetic-root /path/to/noise_data/Synthetic-Norm --output /path/to/new/fixed_run
```

For multiple GPUs, use the DDP wrapper, whose batch size is per process:

```bash
torchrun --standalone --nproc_per_node=2 train_from_config.py --ddp --config configs/multilength.json --data-root /path/to/noise_data --synthetic-root /path/to/noise_data/Synthetic-Norm --output /path/to/new/ddp_run
```

## Sample from an inference checkpoint

Extract a generation checkpoint archive separately, then:

```bash
python sample_noise.py --ckpt /path/to/noise_generation/multilength/inference.pt --n 120 --lengths 100,150,200,300,400,450,500,600,750,1000,1500,2048 --num_steps 50 --seed 42 --out_path /path/to/outputs/multilength.npz

python sample_noise.py --ckpt /path/to/noise_generation/fixed2048/inference.pt --n 100 --length 2048 --num_steps 50 --seed 42 --out_path /path/to/outputs/fixed2048.npz
```

EMA is selected by default; `--no_ema` selects the model state. With the supplied
1,000-step diffusion config, `--num_steps 50` uses DDIM and `--num_steps 1000`
uses DDPM. Sampling batches are accumulated in RAM, so generate large collections
in manageable batches or separate output files. New NPZ metadata uses Unicode
arrays instead of pickle-backed object arrays.

## Evaluate generated noise

```bash
python evaluate_noise.py --generated /path/to/outputs/multilength.npz --data_root /path/to/noise_data --synthetic_root /path/to/noise_data/Synthetic-Norm --config configs/multilength.json --output_dir /path/to/noise_evaluation
python noise_eval_report.py /path/to/noise_evaluation
```

These are the original handcrafted-descriptor diagnostics: conditional coverage,
integrated beta-recall, authenticity and candidate waveform similarity. Descriptor
adaptations are not the original learned one-class metric implementation.
Near-copy candidates do not by themselves establish or exclude memorization.

## Portability and verification

`source_fingerprints.json` records the original source hashes. Publication edits
remove local drive/home paths, use restricted checkpoint loading, add config and
NPY-conversion wrappers, and save non-pickled NPZ metadata. Numerical model,
diffusion, normalization and evaluation implementations are retained. Inference
exports preserve model and EMA states plus portable config and step, while
excluding optimizer/scaler state. They are not training-resume checkpoints.
