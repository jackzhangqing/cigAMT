"""Create portable inference copies of the two author Noise-DiT checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    results = []
    for item in json.loads(args.inventory.read_text())['checkpoints']:
        if not item.get('selected'): continue
        source = Path(item['source']).resolve()
        target = output / item['archive_path']
        if target.resolve() == source or target.exists():
            raise FileExistsError('Use a new export path; source checkpoints are never replaced')
        ck = torch.load(source, map_location='cpu', weights_only=True)
        cfg = json.loads(Path(item['public_config']).read_text())
        # Preserve the actual architecture/diffusion settings from the checkpoint.
        config = dict(ck['config'])
        for key in ['data_root', 'synthetic_root', 'out_dir', 'resume_from']:
            config[key] = cfg[key]
        exported = {key: ck[key] for key in ['model', 'ema', 'step', 'noise_type_names'] if key in ck}
        exported['config'] = config
        exported['publication_note'] = item['sampling_checkpoint_linkage']
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(exported, target)
        (target.parent / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
        results.append({'path': target.relative_to(output).as_posix(),
                        'bytes': target.stat().st_size,
                        'sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
                        'variant': item['variant'], 'step': ck.get('step'),
                        'contains_model_and_ema': 'ema' in ck,
                        'sampling_checkpoint_linkage': item['sampling_checkpoint_linkage']})
        print(json.dumps(results[-1]), flush=True)
    (output / 'noise_generator_export_manifest.json').write_text(json.dumps({'checkpoints': results}, indent=2)+'\n')


if __name__ == '__main__': main()
