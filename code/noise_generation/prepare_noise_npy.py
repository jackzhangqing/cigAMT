"""Convert only released training field/procedural templates to NoiseNPYDataset layout."""
import argparse
import json
from pathlib import Path
import h5py
import numpy as np

NAMES = ['Pulse', 'Quasi-Sinusoidal', 'Square-wave', 'Step-wave', 'Triangular-wave',
         'Power-frequency', 'White-noise', 'Color-noise', 'Harmonic', 'Composite']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path, help='Downloaded amt_150hz directory')
    parser.add_argument('--output', required=True, type=Path, help='New derived NPY directory')
    args = parser.parse_args()
    root, output = args.data_root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Keep generated NPY files outside the downloaded dataset')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Choose a new empty output directory')
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'noise/noise_manifest.json').read_text())
    index = np.load(root/manifest['pool_arrays_path'], allow_pickle=False)
    keep = np.flatnonzero((index['split_id'] == 0) & np.isin(index['domain_id'], [0,1]))
    counts = {'field': 0, 'procedural': 0}
    with h5py.File(root/manifest['noise_bank_path'], 'r') as bank:
        samples = bank[manifest.get('key', 'samples')]
        for i in keep:
            length, crop = int(index['length'][i]), int(index['crop_start'][i])
            domain = int(index['domain_id'][i])
            folder = (f'Real-Len{length}' if domain == 0 else f'Synthetic-Norm/Synthetic-Len{length}')
            path = output/folder/NAMES[int(index['type_id'][i])-1]/f'template_{int(i):07d}_len{length}.npy'
            path.parent.mkdir(parents=True, exist_ok=True)
            values = np.asarray(samples[int(index['row_id'][i]),crop:crop+length], dtype=np.float32)
            np.save(path, values, allow_pickle=False)
            counts['field' if domain == 0 else 'procedural'] += 1
    (output/'conversion.json').write_text(json.dumps({
        'source': 'cigAMT/amt_150hz released noise bank', 'split': 'train',
        'excluded': 'validation, test, diffusion-generated templates', 'counts': counts,
        'scope': 'Portable training subset; not a reconstruction of historical generator fitting membership.'
    }, indent=2)+'\n')
    print(json.dumps(counts))


if __name__ == '__main__': main()
