"""Make an inference-only copy of a trusted GeoAMT-DiT training checkpoint.

This utility is provided for when the original checkpoint is recovered. It
does not imply that the pretrained generator is present in this repository.
"""
from pathlib import Path
import argparse
import torch

CONFIG_KEYS=('in_channels','input_length','patch_size','hidden_dim','depth',
             'num_heads','mlp_ratio','num_timesteps','beta_schedule')


def export_generator(source, destination, use_ema=True):
    source, destination=Path(source).resolve(),Path(destination).resolve()
    if source==destination or destination.exists():
        raise ValueError('Destination must be a new file distinct from the source')
    checkpoint=torch.load(source,map_location='cpu',weights_only=True)
    config={key:checkpoint['config'][key] for key in CONFIG_KEYS}
    key='ema' if use_ema else 'model'
    if key not in checkpoint:
        raise KeyError('Requested weight set is absent: '+key)
    state=checkpoint[key]
    if not isinstance(state,dict) or not state or not all(isinstance(v,torch.Tensor) for v in state.values()):
        raise ValueError('Expected a tensor state_dict, not an optimizer or a nested object')
    result={'config':config,key:state,'step':checkpoint.get('step'),
            'weight_source':key,'format':'GeoAMT-DiT inference-only; optimizer and private paths excluded'}
    destination.parent.mkdir(parents=True,exist_ok=True)
    torch.save(result,destination)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',required=True)
    p.add_argument('--destination',required=True)
    p.add_argument('--model-weights',action='store_true',help='Export raw model instead of EMA')
    args=p.parse_args()
    export_generator(args.source,args.destination,not args.model_weights)
