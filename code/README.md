# Benchmark-code release preparation

The project website source is available in this repository. The **original benchmark training, inference, and evaluation code has not yet been packaged for public release here**. This directory documents its scope; it is not a runnable copy of the four original modules.

## Modules

| Module | Purpose | Verified source entry points |
| --- | --- | --- |
| Reference-signal generation | Joint four-component AMT generation using GeoAMT-DiT and DDPM/DDIM sampling. | `train.py`, `sample.py`, `model.py`, `diffusion.py`, `geoamt_clean_dataset.py` in the reference-generation project. |
| Independent-noise generation | Procedural noise and conditional Noise-DiT sampling by class and active length. | `train_noise.py`, `train_noise_ddp.py`, `sample_noise.py`, `noise_model.py`, `noise_dataset.py` in the noise-generation project. |
| Window-level sferic identification | Four-component, 240-sample windows at 24 kHz; classification and field-selection analysis. | Release entry points will be listed after the original sferic code is consolidated. |
| Multitask AMT evaluation | Noise classification/localization, two denoising tasks, forecasting, interpolation, imputation, and impedance case analysis. | Release entry points will be listed after the original benchmark code and task-specific configurations are consolidated. |

The two generation modules are separate from the two evaluation groups. Diffusion perturbations used in model training are also distinct from interference samples supplied by the independent noise library.

## Generation interfaces checked in the source

- GeoAMT-DiT generates arrays of shape `(N, 4, 2048)`, ordered `Ex, Ey, Hx, Hy`. Its default backbone uses patch size 8, hidden dimension 384, 12 DiT blocks, and 6 attention heads.
- Conditional Noise-DiT generates arrays of shape `(N, 1, 2048)`. The main multi-length configuration conditions on noise type and active-length ratio; source conditioning is optional and is disabled in the inspected main configuration.
- Noise labels cover Pulse, Quasi-Sinusoidal, Square-wave, Step-wave, Triangular-wave, Power-frequency, White-noise, Color-noise, Harmonic, and Composite.
- Active lengths are 100, 150, 200, 300, 400, 450, 500, 600, 750, 1000, 1500, and 2048 samples.

## Sampling command reference

The following flags were checked against the original sampling scripts. Run them only after the corresponding code, dependencies, checkpoint, and release instructions are available. They are **not commands that can be run from this documentation-only directory**.

From the future reference-generation module:

```bash
python sample.py --ckpt <checkpoint.pt> --n 8 --num_steps 50 --seed 42 --out_path <new-output-directory>/reference_preview.npy
```

From the future noise-generation module:

```bash
python sample_noise.py --ckpt <checkpoint.pt> --n 20 --type_name all --lengths 100,300,750,1500,2048 --num_steps 50 --seed 42 --out_path <new-output-directory>/noise_preview.npz
```

For `sample_noise.py`, `--n` is the total number of generated samples, not a per-class or per-length count. Sampling uses DDIM when the requested step count is below the configured diffusion length, and full DDPM when it reaches that length.

The original reference sampler stores HDF5 waveforms under `data`. The noise sampler stores `samples`, `type_id`, `length`, and `source_id`. Some original NPZ exports additionally contain object-valued metadata arrays; the small release preparation uses numeric arrays and separate JSON metadata so that loading can keep `allow_pickle=False`.

## Verification scope

Preparation inspected selected source files, configuration files, small metadata, and sample-array headers. No training or inference experiment was rerun. No source directory was modified, no model checkpoint was loaded, and no benchmark result was regenerated.

Before a full code release, the authors need to select the maintained entry points, document dependencies and checkpoints, confirm split provenance and dataset access, and choose the applicable code and data licenses. The website does not claim those release steps are already complete.
