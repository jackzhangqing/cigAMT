"""Download, verify and safely extract a listed official cigAMT weight archive."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import urllib.error
import urllib.request
import zipfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--list', action='store_true')
    p.add_argument('--asset')
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'code/pretrained_assets.json').read_text(encoding='utf-8'))
    assets = {x['name']: x for x in manifest['assets']}
    if a.list:
        for name, row in assets.items():
            print(name, '%.1f MiB' % (row['bytes'] / 1024**2))
        return
    if a.asset not in assets or a.output is None:
        p.error('Choose a listed --asset and an --output directory')
    item = assets[a.asset]
    a.output.mkdir(parents=True, exist_ok=True)
    archive = a.output / a.asset
    temporary = archive.with_suffix('.zip.part')
    if not archive.exists():
        try:
            req = urllib.request.Request(item['url'], headers={'User-Agent': 'cigAMT-weights-downloader'})
            with urllib.request.urlopen(req, timeout=60) as r, temporary.open('wb') as f:
                for block in iter(lambda: r.read(1024 * 1024), b''):
                    f.write(block)
        except urllib.error.HTTPError as exc:
            raise SystemExit('Asset unavailable: check whether weights-v1 attachments have been published. HTTP %s' % exc.code)
        temporary.replace(archive)
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    if archive.stat().st_size != item['bytes'] or h.hexdigest() != item['sha256']:
        raise SystemExit('Archive checksum mismatch; do not load these weights.')
    root = a.output.resolve()
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            parts = PurePosixPath(name).parts
            target = (root / name).resolve()
            if name.startswith('/') or '\\' in name or '..' in parts or root not in target.parents:
                raise SystemExit('Unsafe archive member')
            if target.exists():
                raise SystemExit('Refusing to overwrite an extracted file: ' + name)
        z.extractall(root)
    print('Verified and extracted', archive.name)


if __name__ == '__main__':
    main()
