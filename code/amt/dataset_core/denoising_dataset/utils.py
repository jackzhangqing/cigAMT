from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import numpy as np

SPLITS = ['train', 'validation', 'test']
TASKS = ['noise_localization','noise_classification','denoising','frequency_denoising',
         'forecasting','interpolation','imputation']

def json_read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def json_write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.partial')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temporary,path)

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()

def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def seed64(*parts):
    return int.from_bytes(hashlib.sha256(json.dumps(parts,separators=(',',':'),ensure_ascii=False).encode()).digest()[:8],'little')

def make_clean_index(clean):
    sources=[];rows=[];roles=[]
    for sid,source in enumerate(clean['sources']):
        for role,split in enumerate(SPLITS):
            indices=np.asarray(source['split_indices'][split],dtype=np.int64)
            sources.append(np.full(len(indices),sid,np.int16));rows.append(indices)
            roles.append(np.full(len(indices),role,np.uint8))
    return {'source_id':np.concatenate(sources),'source_row':np.concatenate(rows),
            'split_id':np.concatenate(roles)}
