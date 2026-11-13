# cigAMT pretrained baseline models

Inference-only assets for the 24 kHz sferic branch and seven 150 Hz tasks.

- 48 sferic assets: 33 neural best checkpoints and 15 sklearn estimators.
- 80 AMT checkpoints: shared models and PatchTST/SAITS specialist adapters.
- Extracted architecture/configuration and frozen validation threshold files accompany weights.
- Optimizer/resume state is excluded; original experiment files are unchanged.

See code/WEIGHTS.md and code/pretrained_assets.json in the tagged source for checksums, checkpoint coverage, preprocessing, and evaluation commands. The data core is hosted at https://huggingface.co/datasets/ZhangQing123/cigAMT .

This release provides inference compatibility; it does not assert fresh retraining or statistical significance from seed summaries. Sklearn assets require the compatible environment and are trusted pickle-based files.

- Two additional run-matched Noise-DiT generator checkpoints (model + EMA), with historical NPZ checkpoint linkage explicitly unverified.
- GeoAMT-DiT reference-generator pretrained weights were not found locally and are not included.
