# Window-level sferic identification (24 kHz)

This directory contains the actual local benchmark implementations used for the
sferic experiments, together with a publication adapter for the compact
[cigAMT dataset](https://huggingface.co/datasets/ZhangQing123/cigAMT).
The original experiment directories are not required or modified.

## Data and input contract

Download the `sferic_24khz/` directory. Preserve its layout:

```text
sferic_24khz/
  train/{windows.h5,windows.csv,records.csv}
  val/{windows.h5,windows.csv,records.csv}
  test/labeled/{windows.h5,windows.csv,records.csv}
```

The reader uses **`windows.h5:x`**, the frozen standardized `float32 [N,4,240]`
inputs in `Ex,Ey,Hx,Hy` order, with `y` as the binary window label. It does not
standardize `x` a second time. `sx` contains the field crop before artificial
augmentation noise and must not replace `x`. `mx` is an annotation mask, not an
observation mask. Split-local CSV files preserve row and source-record identity.
Optional full source records referenced in `records.csv` are not required to
train or evaluate this classification benchmark.

The fixed partition contains 22,520 training entries (11,260 original and 11,260
fixed augmented copies), 3,130 validation windows, and 2,488 test windows. The
27 labeled source records are split 15/8/4. The four test records are from
Nanjing: `R0489,R0493,R0495,R0504`. This partition measures held-out-record
performance; it does not constitute cross-region validation. Augmentation is
restricted to training and is already saved in the dataset.

## Installation

From this directory, use Python 3.9 or newer and an appropriate PyTorch build:

```bash
python -m pip install -r requirements.txt
python benchmark.py list-models
```

Original frozen sklearn estimators were saved with **scikit-learn 1.1.3**. Use
`requirements-pretrained-classical.txt` in a compatible separate environment
(for example Python 3.10) when evaluating those pickle-based files. Training a
new sklearn model with a newer version is a separate run, not the original
frozen estimator. The optional native Razorback backend also requires its own
library installation and `dask[array]`; it is not silently substituted by the
SciPy implementation.

## Audit and train

Commands below assume the downloaded data are under `D:/cigAMT_data`. Replace
that path for your computer. Run commands from `code/sferic/`.

```bash
python benchmark.py audit --config configs/main.json --set data_root=D:/cigAMT_data/sferic_24khz --set output_root=outputs/audit
python benchmark.py train --config configs/main.json --model tcn --seed 42 --set data_root=D:/cigAMT_data/sferic_24khz --set output_root=outputs/main_fp32 --device cpu
```

For the full configured 16-baseline, three-seed suite:

```bash
python benchmark.py suite --config configs/main.json --set data_root=D:/cigAMT_data/sferic_24khz --set output_root=outputs/main_fp32
```

Set `--device cuda:0` if available. Checkpoints are selected by validation AP;
the classification threshold is selected by validation F1 and then frozen for
test evaluation. Training writes only to the chosen output directory. Results
include per-window predictions, run settings, split manifests, training curves,
and dataset/implementation fingerprints.

For a **new run trained using this publication code**, reevaluate with:

```bash
python benchmark.py evaluate --run-dir outputs/main_fp32/tcn/REPLACE_WITH_RUN_FOLDER --data-root D:/cigAMT_data/sferic_24khz --device cpu
```

This command deliberately enforces the saved code and file fingerprints. It
does not accept an original historical run whose source or input storage format
differs from this publication port. Use the dedicated release evaluator below
for prepared pretrained assets.

## Pretrained assets and evaluation

[`pretrained_manifest.json`](pretrained_manifest.json) inventories **48 original
inference assets**: 33 neural best checkpoints (11 models, three seeds each)
and 15 sklearn estimators (four feature classifiers plus the prior reference,
three seeds each). Resume/optimizer `last.pt` files are excluded. The manifest
records proposed asset names; consult the repository release status for actual
download availability. Weight files are supplied separately from Git source.

A prepared neural asset contains `model.pt` and `release_run.json`. The tensor
state, architecture configuration, run identity, and validation threshold are
retained; machine-specific paths/environment metadata are omitted. Evaluate an
extracted asset as follows:

```bash
python evaluate_pretrained.py --data-root D:/cigAMT_data/sferic_24khz --run-dir D:/cigAMT_weights/sferic/tcn/seed_42 --device cpu --output outputs/tcn_seed42_test.json
```

For a prepared sklearn asset, use the compatible sklearn environment and add
`--trust-joblib`. Joblib is pickle-based and can execute code; only opt in for
the trusted release artifact. Neural weights are loaded using
`torch.load(..., weights_only=True)` and strict state-dictionary matching.
The release evaluator uses the stored threshold without reselection and reports
AP, ROC-AUC, F1, precision, recall, balanced accuracy, MCC, and calibration
metrics where applicable. It reports the native SVM margin as a ranking score,
not as a probability.

`export_pretrained.py` is a maintainer utility for preparing publication copies
of original runs. It reads originals and writes to a distinct output directory;
it does not upload files. The publication evaluator explicitly records that it
is a storage-format port rather than asserting the original file fingerprints.

## Architectures and sources

All neural architectures are implemented in [`bench/models.py`](bench/models.py)
and return one binary logit per four-component window. The registry includes
MLP, CNN-1D, VGG16-1D, ResNet-1D, InceptionTime, TCN, BiGRU, Transformer,
PatchTST classifier, STFT-ResNet18, and STFT-ViT. Feature extraction and the
LogReg/SVM/RF/ExtraTrees/prior estimators are in
[`bench/classical.py`](bench/classical.py).

These are local task adaptations, not claims of identical reproduction of the
original papers or official upstream implementations. In particular, PatchTST
here is a classification adaptation; STFT models are trained from scratch on
native four-channel spectrograms rather than ImageNet RGB images. Exact
architecture differences, defaults, and primary-paper links are recorded in
`MODEL_INFO` and [`model_sources.json`](model_sources.json).

## Continuous-record selection and impedance cases

Classification and impedance use different amplitude representations. A
continuous-record case requires an actual `[4,T]` NumPy record at 24 kHz with
the electric/magnetic amplitude relationships retained. Independently
standardized `x` windows must not be stitched together for physical impedance
estimation. The compact labeled release alone does not contain all continuous
field records or the instrument calibration needed to reproduce every field
case.

```bash
python predict_record.py --record D:/field/record.npy --run-dir D:/cigAMT_weights/sferic/tcn/seed_42 --output-dir outputs/field_selection --device cpu
python impedance_case.py --record D:/field/record.npy --fs 24000 --frequencies 1000 1500 2000 3000 --intervals outputs/field_selection/selected_intervals.json --unit-system source --backend scipy --output outputs/field_impedance.npz
```

The selection runner matches the supplied nonoverlapping 240-sample crop and
own-crop z-score convention. It saves scores and selected intervals in original
sample coordinates. Impedance Fourier windows remain inside each selected
interval: short isolated intervals may yield no valid low-frequency estimates.
No ground-truth classification accuracy is claimed for unlabeled field records.

[`field_compare/impedance.py`](field_compare/impedance.py) contains the supplied
Slepian-taper, frequency-local estimator and native Razorback adapter. `scipy`
uses its own complex Huber regression and conditional error calculation;
`razorback` invokes the separately installed native library with
`--library-root /path/to/library_parent`. These are explicit different backends.
The sferic-field implementation is distinct from the 150 Hz Welch/Huber
denoising evaluation. Use the backend and calibration specified for each case.

`--unit-system source` reports an uncalibrated resistivity proxy. Use `SI` only
when stored electric channels are V/m and magnetic channels are A/m, or `field`
only for the documented (mV/km)/nT convention. Supply a calibration JSON when
response/unit corrections are required. See the estimator docstring for the
conversion-factor schema. Error bars reflect the impedance estimator, not
uncertainty across classifier training seeds or independent regions.

## Publication checks

Preparation checks verified that every released HDF5 `x` row equals the legacy
NPY input exactly, across all three partitions. All 33 neural best checkpoints
were safely loaded, matched strictly to the published architecture, and passed
a bounded two-window forward check; every architecture file matched its
training archive. These checks verify interface compatibility, not fresh
training or an independent reproduction of all reported results.
