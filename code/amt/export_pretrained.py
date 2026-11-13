"""Export selected trusted local training checkpoints as portable inference runs.

Never loads arbitrary globals or writes into a source experiment directory.
The private inventory is prepared separately and is not a publication asset.
"""
import argparse
import json
import hashlib
from pathlib import Path
import _codecs
import numpy as np
import torch

KEYS = ('signature', 'task', 'name', 'scheme', 'seed', 'config',
        'model_options', 'sequence_length', 'dataset_id', 'model', 'epoch', 'best')


def safe_load(path):
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([
            np.core.multiarray._reconstruct, np.ndarray, np.dtype,
            type(np.dtype(np.uint32)), _codecs.encode])
    return torch.load(path, map_location='cpu', weights_only=True)


def export_inventory(inventory, output):
    entries = json.loads(Path(inventory).read_text(encoding='utf-8'))['checkpoints']
    output = Path(output).resolve()
    results = []
    for entry in entries:
        if not entry.get('selected'): continue
        source = Path(entry['source']).resolve()
        target = output / entry['archive_path']
        if target.resolve() == source: raise ValueError('Refusing to replace a source checkpoint')
        target.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = safe_load(source)
        exported = {key: checkpoint[key] for key in KEYS if key in checkpoint}
        if exported['task'] != entry['task'] or exported['name'] != entry['model']:
            raise ValueError('Checkpoint identity differs from inventory: ' + str(source))
        if target.exists():
            raise FileExistsError('Use a new output directory: ' + str(target))
        torch.save(exported, target)
        for name, key in [('run_config.json', 'public_run_config'), ('calibration.json', 'public_calibration')]:
            if entry.get(key):
                text = Path(entry[key]).read_text(encoding='utf-8')
                (target.parent / name).write_text(text, encoding='utf-8')
        results.append({'path': target.relative_to(output).as_posix(),
                        'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'task': entry['task'], 'model': entry['model'], 'seed': entry['seed']})
        print(json.dumps(results[-1]), flush=True)
    (output / 'amt_export_manifest.json').write_text(json.dumps({'checkpoints': results}, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    export_inventory(args.inventory, args.output)
