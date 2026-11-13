from __future__ import annotations
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import socket
import sys
import time
import numpy as np

TASKS = ('noise_localization','noise_classification','denoising','frequency_denoising',
         'forecasting','interpolation','imputation')
NOISE_TASKS = TASKS[:4]


def clean_json(x):
    if isinstance(x, dict): return {str(k):clean_json(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [clean_json(v) for v in x]
    if isinstance(x, np.ndarray): return clean_json(x.tolist())
    if isinstance(x, np.generic): return clean_json(x.item())
    if isinstance(x, Path): return str(x)
    if isinstance(x, float) and not math.isfinite(x): return None
    return x


def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(clean_json(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(str(temporary),str(path))


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return hashlib.sha256(json.dumps(clean_json(value),sort_keys=True,separators=(',',':')).encode()).hexdigest()


def code_fingerprint():
    # A distinct protocol identity, preserving the legacy checkpoint guards.
    base=Path(__file__).resolve().parents[1]
    paths=list(base.rglob('*.py'))+list((base/'vendor').rglob('UPSTREAM.json'))
    result={p.relative_to(base).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths) if 'tests' not in p.relative_to(base).parts and not p.name.startswith('test_')}
    root=base.parent
    shared=[root/'publication_reader.py',root/'export_pretrained.py']
    shared+=list((root/'dataset_core/denoising_dataset').rglob('*.py'))
    result.update({p.relative_to(root).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(shared) if p.is_file()})
    return result


def environment():
    out={'python':sys.version,'platform':platform.platform(),'host':socket.gethostname(),
         'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'numpy':np.__version__}
    import h5py
    out['h5py']=h5py.__version__
    try:
        import torch
        out.update(torch=torch.__version__,cuda_runtime=torch.version.cuda,
                   cuda_available=torch.cuda.is_available(),
                   devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
    except (ImportError,OSError) as e: out['torch_error']=str(e)
    return out


def seed_all(seed, deterministic=False):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=not deterministic
    if deterministic:torch.use_deterministic_algorithms(True)


class RunLock:
    """OS lock automatically released even after a crash; no stale-lock removal."""
    def __init__(self, directory):self.path=Path(directory)/'.run.lock';self.handle=None
    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.handle=open(self.path,'a+b')
        self.handle.seek(0,2)
        if not self.handle.tell():self.handle.write(b'0');self.handle.flush()
        self.handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as e:
            self.handle.close();self.handle=None
            raise RuntimeError('Run is active in another process: '+str(self.path)) from e
        write_json(self.path.parent/'lock_owner.json',{'pid':os.getpid(),'host':socket.gethostname(),'started_unix':time.time()})
        return self
    def __exit__(self,*args):
        if self.handle:
            self.handle.seek(0)
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(self.handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                import fcntl
                fcntl.flock(self.handle.fileno(),fcntl.LOCK_UN)
            self.handle.close();self.handle=None


class RowWriter:
    def __init__(self,path):self.path=Path(path);self.handle=None;self.writer=None
    def write(self,row):
        if self.handle is None:
            self.path.parent.mkdir(parents=True,exist_ok=True)
            self.handle=open(self.path,'w',encoding='utf-8-sig',newline='')
            self.writer=csv.DictWriter(self.handle,fieldnames=list(row),extrasaction='raise');self.writer.writeheader()
        self.writer.writerow(clean_json(row))
    def close(self):
        if self.handle:self.handle.close();self.handle=None


def write_table(path,rows):
    rows=list(rows)
    if not rows:return
    keys=list(dict.fromkeys(k for r in rows for k in r))
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with open(path,'w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader()
        for r in rows:writer.writerow(clean_json(r))
