"""Export a local frozen run without private filesystem/environment metadata.

Read only trusted original run metadata. PyTorch uses weights_only=True;
sklearn estimators are copied without unpickling. Never overwrite originals.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil

SAFE_CONFIG = ('model', 'seed', 'model_config', 'version', 'channels', 'fs_hz',
    'augmentation', 'protocol', 'split_seed', 'train_fraction', 'batch_size',
    'cpu_threads', 'amp', 'threshold_method', 'target_recall')


def export_run(run_dir, output_dir):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    if output_dir == run_dir or run_dir in output_dir.parents:
        raise ValueError('Export must be outside the original run directory')
    report = json.loads((run_dir / 'run_metrics.json').read_text(encoding='utf-8'))
    cfg = report['config']
    metadata = {'format_version': 1, 'model': cfg['model'], 'seed': cfg['seed'],
        'run_id': report['run_id'], 'threshold': report['threshold'],
        'threshold_source': 'Frozen validation-selected threshold from original run report',
        'config': {k: cfg[k] for k in SAFE_CONFIG if k in cfg},
        'original_test_metrics': report.get('test'),
        'release_note': 'Publication port reads the frozen HDF5 x tensors. Original implementation/data fingerprints are not asserted for this port.'}
    output_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'best.pt').is_file():
        import torch
        checkpoint = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=True)
        if checkpoint['run_id'] != report['run_id']:
            raise ValueError('Checkpoint/report run IDs disagree')
        public = {'state_dict': checkpoint['model'], 'model_name': cfg['model'],
            'model_config': cfg.get('model_config', {}), 'seed': cfg['seed'],
            'run_id': report['run_id'], 'threshold': report['threshold'],
            'epoch': checkpoint.get('epoch'),
            'validation_pr_auc': checkpoint.get('validation_pr_auc')}
        destination = output_dir / 'model.pt'
        if destination.exists():
            raise FileExistsError(destination)
        torch.save(public, destination)
        metadata['checkpoint'] = 'model.pt'
        metadata['checkpoint_format'] = 'pytorch tensor state plus primitive metadata; weights_only=True'
    else:
        destination = output_dir / 'estimator.joblib'
        if destination.exists():
            raise FileExistsError(destination)
        shutil.copy2(run_dir / 'estimator.joblib', destination)
        metadata['checkpoint'] = 'estimator.joblib'
        metadata['checkpoint_format'] = 'sklearn joblib; pickle based; load only trusted release artifacts'
    (output_dir / 'release_run.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    result = export_run(args.run_dir, args.output_dir)
    print(json.dumps({'model':result['model'], 'seed':result['seed'], 'checkpoint':result['checkpoint']}))
