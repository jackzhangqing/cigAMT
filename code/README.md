# cigAMT benchmark implementation

| Branch | Entry points | Documentation |
|---|---|---|
| 24 kHz sferic identification | `sferic/benchmark.py`, `sferic/evaluate_pretrained.py` | [Training, evaluation, architectures](sferic/README.md) |
| Seven 150 Hz tasks | `amt/run.py`, `amt/run_specialists.py` | [Training, evaluation and replay](amt/README.md) |
| Sferic selection and impedance | `sferic/predict_record.py`, `sferic/impedance_case.py` | [Continuous-record inputs and backends](sferic/README.md#continuous-record-selection-and-impedance-cases) |
| 150 Hz impedance | `amt/impedance/run_impedance.py` | [Welch-Huber estimation](amt/README.md#impedance-estimation) |

Data: [Hugging Face](https://huggingface.co/datasets/ZhangQing123/cigAMT). Weights: [asset instructions](WEIGHTS.md); supplied separately through GitHub Releases.

These are publication copies adapted to shared-core HDF5 storage. Original experiment directories were not modified. Model implementations and numerical evaluation routines are retained; storage readers and portable entry points are documented per module. The static website needs no scientific Python installation.

Generation modules: [reference signals](reference_generation/README.md) and [structured noise](noise_generation/README.md), including training, sampling and quality evaluation. See their checkpoint availability and historical fitting-source notes.
