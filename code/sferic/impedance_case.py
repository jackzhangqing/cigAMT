"""Run the supplied sferic-field impedance implementation on a continuous record.

Input arrays must retain the physical electric/magnetic amplitude relationships.
Never concatenate independently normalized classification windows for this step.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from field_compare.impedance import estimate_impedance
from bench.runtime import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record', required=True, help='Four-component continuous .npy [4,T], Ex/Ey/Hx/Hy')
    p.add_argument('--fs', type=float, required=True)
    p.add_argument('--frequencies', type=float, nargs='+', required=True)
    p.add_argument('--intervals', help='JSON list of selected [start,stop) sample intervals; omitted uses whole record')
    p.add_argument('--calibration', help='JSON response/unit conversion factors; see estimate_impedance docstring')
    p.add_argument('--unit-system', choices=('source','SI','field'), default='source')
    p.add_argument('--backend', choices=('scipy','razorback'), default='scipy')
    p.add_argument('--library-root', help='Path to separately installed native Razorback source')
    p.add_argument('--output', required=True, help='New .npz output; metadata saved alongside as .json')
    args = p.parse_args()
    load_json = lambda path: json.loads(Path(path).read_text(encoding='utf-8')) if path else None
    sx = np.load(args.record, mmap_mode='r', allow_pickle=False)
    result = estimate_impedance(sx, args.fs, args.frequencies,
        intervals=load_json(args.intervals), calibration=load_json(args.calibration),
        unit_system=args.unit_system, backend=args.backend, library_root=args.library_root)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{k:v for k,v in result.items() if isinstance(v, np.ndarray)})
    write_json(path.with_suffix('.json'), {k:v for k,v in result.items() if not isinstance(v,np.ndarray)})
    print(str(path))


if __name__ == '__main__':
    main()
