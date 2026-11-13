"""Classify an unnormalized continuous four-component 24 kHz field record.

The 240-sample crops and own-crop float64 z-score match the supplied field
inference implementation. Returned intervals refer to the original record.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from bench.data import CHANNELS, _channels
from bench.engine import _predict
from bench.runtime import write_json
from evaluate_pretrained import load_release_model


class RecordWindows:
    def __init__(self, sx, config):
        self.sx, self.config = sx, config
        self.starts = np.arange(0, sx.shape[1] - 240 + 1, 240, dtype=np.int64)

    def __len__(self):
        return len(self.starts)

    def iter_batches(self, batch_size):
        selected = _channels(self.config['channels'])
        for lo in range(0,len(self),batch_size):
            starts=self.starts[lo:lo+batch_size]
            x=np.stack([self.sx[:4,start:start+240] for start in starts]).astype(np.float32)
            if self.config['version']=='standardized':
                mean=x.mean(axis=-1,keepdims=True,dtype=np.float64)
                std=x.std(axis=-1,keepdims=True,dtype=np.float64)
                x=((x-mean)/np.where(std>0,std,1.)).astype(np.float32)
            if not np.isfinite(x).all():
                raise ValueError('Nonfinite input')
            for i,name in enumerate(CHANNELS):
                if name not in selected:
                    x[:,i]=0
            yield x,None,starts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--record',required=True,help='Continuous [4,T] NPY Ex/Ey/Hx/Hy; physical amplitudes retained')
    p.add_argument('--run-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--trust-joblib',action='store_true')
    args=p.parse_args()
    metadata,config,model,device=load_release_model(args.run_dir,args.device,args.batch_size,args.trust_joblib)
    if float(config['fs_hz']) != 24000:
        raise ValueError('Expected a 24 kHz classifier')
    sx=np.load(args.record,mmap_mode='r',allow_pickle=False)
    if sx.ndim!=2 or sx.shape[0]<4 or sx.shape[1]<240:
        raise ValueError('Record must have at least four channels and 240 samples')
    ds=RecordWindows(sx,config)
    result=_predict(model,ds,config,device)
    positive=np.asarray(result['scores'],dtype=np.float64)>=float(metadata['threshold'])
    intervals=[[int(start),int(start)+240] for start in ds.starts[positive]]
    out=Path(args.output_dir)
    out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'window_scores.npz',starts=ds.starts,scores=result['scores'],positive=positive)
    write_json(out/'selected_intervals.json',intervals)
    write_json(out/'selection.json',dict(model=config['model'],seed=config['seed'],
        threshold=metadata['threshold'],score_kind=result['score_kind'],windows=len(ds),
        selected_windows=int(positive.sum()),sampling_rate_hz=24000,
        interval_convention='[start,stop), original sample coordinates; 240-sample crop boundaries retained',
        ground_truth='unknown; no classification accuracy claimed for unlabeled field data'))


if __name__=='__main__':
    main()
