"""Evaluate a prepared release checkpoint on the frozen labeled test split."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import joblib
from bench.data import load_bundle
from bench.engine import _device, _metrics, _predict
from bench.runtime import write_json


def load_release_model(run_dir, device_name='cpu', batch_size=128, trust_joblib=False):
    run_dir = Path(run_dir)
    metadata = json.loads((run_dir / 'release_run.json').read_text(encoding='utf-8'))
    config = dict(metadata['config'], device=device_name, batch_size=batch_size)
    config.setdefault('amp', False)
    path = run_dir / metadata['checkpoint']
    if path.suffix == '.joblib':
        if not trust_joblib:
            raise ValueError('joblib can execute code: allow only a trusted published artifact')
        return metadata, config, joblib.load(path), None
    import torch
    from bench.models import create_model
    device = _device(device_name)
    torch.set_num_threads(int(config.get('cpu_threads', 4)))
    ckpt = torch.load(path, map_location='cpu', weights_only=True)
    if ckpt['run_id'] != metadata['run_id'] or ckpt['model_name'] != config['model']:
        raise ValueError('Checkpoint metadata does not match its release_run.json')
    model = create_model(config['model'], channels=4, length=240, config=config.get('model_config'))
    model.load_state_dict(ckpt['state_dict'], strict=True)
    return metadata, config, model, device


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True, help='Downloaded sferic_24khz directory')
    p.add_argument('--run-dir', required=True, help='Extracted asset containing release_run.json')
    p.add_argument('--output', required=True, help='New output JSON path; original run is not modified')
    p.add_argument('--device', default='cpu')
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--trust-joblib', action='store_true', help='Allow loading a trusted pickle-based sklearn asset')
    args = p.parse_args()
    metadata, config, model, device = load_release_model(args.run_dir, args.device,
        args.batch_size, args.trust_joblib)
    bundle = load_bundle(args.data_root, config.get('version', 'standardized'), config.get('channels', 'all'))
    try:
        ds = bundle.fixed_split(augmentation='all').test
        prediction = _predict(model, ds, config, device)
        result = {'model':config['model'], 'seed':config['seed'], 'run_id':metadata['run_id'],
            'test_windows':len(ds), 'test_records':sorted(set(ds.record_ids)),
            'threshold':metadata['threshold'], 'threshold_source':metadata['threshold_source'],
            'test':_metrics(ds, prediction, metadata['threshold']),
            'scope':'HDF5 publication-port reevaluation; no test-time threshold selection'}
        write_json(args.output, result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        bundle.close()


if __name__ == '__main__':
    main()
