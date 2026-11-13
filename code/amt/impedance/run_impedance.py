"""Estimate TS4 Welch-Huber impedance from a four-component NPY array."""
import argparse
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / 'code'))
from impedance_welch import estimate_impedance, save_estimate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path, help='NPY array [4,T], Ex,Ey,Hx,Hy')
    parser.add_argument('--output', required=True, type=Path, help='Output basename, without extension')
    parser.add_argument('--frequencies', type=float, nargs='+')
    parser.add_argument('--unit', choices=['si', 'field'], default='si')
    parser.add_argument('--long-record', action='store_true')
    args = parser.parse_args()
    values = np.load(args.input, allow_pickle=False)
    result = estimate_impedance(values, fs=150., frequencies=args.frequencies,
                                unit=args.unit, long_record=args.long_record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_estimate(result, args.output)


if __name__ == '__main__': main()
