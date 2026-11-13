"""NumPy-only readers for precomputed shards and replayable dynamic data."""
from __future__ import annotations
from bisect import bisect_right
from collections import OrderedDict
from pathlib import Path
import os
import h5py
import numpy as np
from .utils import json_read,SPLITS,TASKS
from .materialize import SampleFactory


class OfflineDataset:
    def __init__(self,root,task,split='train',scheme='A',include_controls=True,normalization='visible'):
        self.root=Path(root).resolve();self.task=task;self.split=split;self.scheme=scheme
        self.normalization=normalization
        if task not in TASKS or split not in SPLITS:raise ValueError('Unknown task or split')
        if scheme not in ['A','C'] or normalization not in ['visible','legacy']:raise ValueError('Invalid reader settings')
        noise=task in TASKS[:4]
        folders=[self.root/f'offline/noise_tasks/{scheme}/{split}'] if noise else [self.root/f'offline/clean_tasks/{task}/{split}']
        if noise and include_controls:folders.append(self.root/f'offline/clean_controls/{split}')
        self.files=[];self.ends=[];self._handles=OrderedDict();self._pid=os.getpid();total=0
        for folder in folders:
            for path in sorted(folder.glob('shard_*.h5')):
                with h5py.File(path,'r') as f:
                    if not f.attrs.get('complete',False):raise ValueError(f'Incomplete shard {path}')
                    total+=len(f['input']);self.files.append(path);self.ends.append(total)
        self.count=total
        if not total:raise ValueError('No completed offline shards found')

    def __len__(self):return self.count

    def __getstate__(self):
        d=self.__dict__.copy();d['_handles']=OrderedDict();return d

    def close(self):
        for f in self._handles.values():f.close()
        self._handles=OrderedDict()

    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<self.count:raise IndexError(index)
        if self._pid!=os.getpid():self.close();self._pid=os.getpid()
        fileindex=bisect_right(self.ends,index)
        row=index-(self.ends[fileindex-1] if fileindex else 0)
        if fileindex not in self._handles:
            if len(self._handles)>=8:
                _,old=self._handles.popitem(last=False);old.close()
            self._handles[fileindex]=h5py.File(self.files[fileindex],'r',rdcc_nbytes=16*1024**2)
        self._handles.move_to_end(fileindex)
        f=self._handles[fileindex]
        noise=self.task in TASKS[:4]
        keys=['input','target','clean_index','source_id','source_row','sample_index']
        keys+=['class_id','domain_id','noise_mask','scale','noise_index','noise_start','noise_length','is_clean'] if noise else ['observed_mask','loss_mask','horizon','gap_length','missing_ratio_nominal','mean_gap','channel_mode_id']
        if self.task=='frequency_denoising':keys+=['target_low','target_high']
        if not noise and self.normalization=='visible':keys+=['input_visible','target_visible','visible_center','visible_scale','visible_scale_fallback']
        out={k:np.asarray(f[k][row]) for k in keys if k in f}
        if not noise and self.normalization=='visible':
            out['input'],out['target']=out.pop('input_visible'),out.pop('target_visible')
        if noise:
            out['observed_mask']=np.ones_like(out['noise_mask'],np.uint8)
            out['loss_mask']=np.ones_like(out['noise_mask'],np.uint8)
            out['noise']=out['input']-out['target']
        out['task']=self.task;out['split']=self.split
        return out


class DynamicDataset:
    """Epoch 0 replays offline rows exactly; later training epochs re-corrupt.

    Evaluation is kept at epoch 0. It never returns clean targets, true noise
    masks or class labels as network input features.
    """
    def __init__(self,root,task,split='train',scheme='A',epoch=0,normalization='visible'):
        self.root=Path(root).resolve();self.task=task;self.split=split;self.scheme=scheme
        if task not in TASKS or split not in SPLITS:raise ValueError('Unknown task or split')
        if scheme not in ['A','C'] or normalization not in ['visible','legacy']:raise ValueError('Invalid reader settings')
        self.epoch=int(epoch) if split=='train' else 0;self.normalization=normalization
        self.factory=None;self.pid=None
        m=json_read(self.root/'manifest.json')
        n=int(m['clean']['totals'][split]);self.count=n*(len(m['config']['noise']['snr_db']) if task in TASKS[:4] else 1)

    def __len__(self):return self.count

    def __getstate__(self):
        d=self.__dict__.copy();d['factory']=None;d['pid']=None;return d

    def close(self):
        if self.factory is not None:self.factory.close()
        self.factory=None;self.pid=None

    def __getitem__(self,index):
        index=int(index)
        if not 0<=index<self.count:raise IndexError(index)
        if self.factory is None or self.pid!=os.getpid():self.factory=SampleFactory(self.root);self.pid=os.getpid()
        ids=np.asarray([index],np.int64)
        if self.task in TASKS[:4]:
            out=self.factory.noise_batch(self.split,self.scheme,ids,self.epoch)
            out['noise']=out['input']-out['target']
            out['observed_mask']=np.ones_like(out['noise_mask'],np.uint8)
            out['loss_mask']=np.ones_like(out['noise_mask'],np.uint8)
        else:
            out=self.factory.clean_batch(self.split,self.task,ids,self.epoch)
            if self.normalization=='visible':out['input'],out['target']=out['input_visible'],out['target_visible']
        result={k:v[0] for k,v in out.items() if isinstance(v,np.ndarray)}
        result.update(task=self.task,split=self.split)
        return result
