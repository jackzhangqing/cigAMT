"""Download a selected cigAMT branch using an explicitly chosen Hub cache."""
import argparse
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', choices=['sferic', 'amt', 'all', 'indexes'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    a = p.parse_args()
    os.environ['HF_HOME'] = str(a.cache.resolve())
    os.environ['HF_HUB_CACHE'] = str(a.cache.resolve() / 'hub')
    os.environ['HF_XET_CACHE'] = str(a.cache.resolve() / 'xet')
    from huggingface_hub import snapshot_download
    patterns = ['README.md', 'USAGE.md', 'DATA_INVENTORY.md', 'requirements.txt', 'metadata/**']
    if a.branch in ['sferic', 'all']:
        patterns += ['sferic_24khz/**', 'examples/**']
    if a.branch in ['amt', 'all']:
        patterns += ['amt_150hz/**']
    if a.branch in ['indexes', 'all']:
        patterns += ['viewer/**']
    path = snapshot_download('ZhangQing123/cigAMT', repo_type='dataset',
                             local_dir=str(a.output.resolve()), allow_patterns=patterns)
    print(path)


if __name__ == '__main__':
    main()
