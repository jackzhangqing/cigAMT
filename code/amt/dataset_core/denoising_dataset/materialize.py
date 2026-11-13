"""Deterministic, resumable HDF5 shards of all task inputs and labels."""
from __future__ import annotations

import os
import time
from pathlib import Path
from collections import defaultdict
import h5py
import numpy as np

from .protocols import mix_and_labels, split_frequency_target, clean_task_labels, visible_standardization
from .utils import json_read, digest, seed64, SPLITS


class SampleFactory:
    def __init__(self,root):
        self.root=Path(root)
        self.manifest=json_read(self.root/'manifest.json')
        self.config=self.manifest['config']
        self.clean=self.manifest['clean']
        self.noise=self.manifest['noise']
        self.index=dict(np.load(self.root/'metadata/clean_index.npz',allow_pickle=False))
        self.noise_index=dict(np.load(self.root/self.noise['pool_arrays_path'],allow_pickle=False))
        self.handles={}
        bank=self.root/self.noise['noise_bank_path']
        with h5py.File(bank,'r') as f:
            d=f[self.noise.get('key','samples')]
            offset=d.id.get_offset()
            if offset is None or d.chunks is not None or d.compression is not None:
                raise ValueError('Noise sample bank must be contiguous and uncompressed for shared memmap')
            self.noise_values=np.memmap(bank,mode='r',dtype=d.dtype,offset=offset,shape=d.shape)
        pools=defaultdict(lambda:defaultdict(lambda:defaultdict(dict)))
        # Partition once. No conversion of a 700k catalog during each sample draw.
        ni=self.noise_index
        for split in np.unique(ni['split_id']):
            for domain in np.unique(ni['domain_id'][ni['split_id']==split]):
                base=(ni['split_id']==split)&(ni['domain_id']==domain)
                for kind in np.unique(ni['type_id'][base]):
                    typed=base&(ni['type_id']==kind)
                    for length in np.unique(ni['length'][typed]):
                        pools[int(split)][int(domain)][int(kind)][int(length)]=np.flatnonzero(typed&(ni['length']==length))
        self.pools={s:{d:dict(k) for d,k in domains.items()} for s,domains in pools.items()}
        self.split_rows={s:np.flatnonzero(self.index['split_id']==i) for i,s in enumerate(SPLITS)}
        self.schedules={}
        for split in SPLITS:
            path=self.root/f'metadata/noise_schedule_{split}.npz'
            if path.is_file():self.schedules[split]=dict(np.load(path,allow_pickle=False))

    def close(self):
        for f in self.handles.values():f.close()
        self.handles={}

    def read_clean(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        out=np.empty((len(ids),4,2048),np.float32)
        for sid in np.unique(self.index['source_id'][ids]):
            selected=np.flatnonzero(self.index['source_id'][ids]==sid)
            source=self.clean['sources'][int(sid)]
            if int(sid) not in self.handles:
                self.handles[int(sid)]=h5py.File(self.root/source['path'],'r',rdcc_nbytes=64*1024**2)
            rows=self.index['source_row'][ids[selected]]
            unique,inverse=np.unique(rows,return_inverse=True)
            value=np.asarray(self.handles[int(sid)][source.get('key','data')][unique],np.float32)
            out[selected]=value[inverse]
        return out

    def noise_batch(self,split,scheme,positions,epoch=0):
        positions=np.asarray(positions,np.int64)
        snrs=np.asarray(self.config['noise']['snr_db'],np.float32)
        basepos,conditions=positions//len(snrs),positions%len(snrs)
        clean_ids=self.split_rows[split][basepos]
        clean=self.read_clean(clean_ids)
        role=SPLITS.index(split);pools=self.pools[role]
        domains=sorted(pools)
        templates=[];starts=[];seeds=[];nids=[]
        for pos,base,cid,condition in zip(positions,basepos,clean_ids,conditions):
            if epoch==0 and split in self.schedules:
                schedule=self.schedules[split]
                nid=int(schedule['noise_index'][int(pos)])
                bankrow=int(self.noise_index['row_id'][nid])
                crop=int(self.noise_index['crop_start'][nid])
                length=int(self.noise_index['length'][nid])
                templates.append(self.noise_values[bankrow,crop:crop+length])
                starts.append(int(schedule['noise_start'][int(pos)]))
                seeds.append(int(schedule['draw_seed'][int(pos)]));nids.append(nid)
                continue
            # Rotate by both base index and SNR to avoid SNR/domain confounding.
            domain=domains[(int(base)+int(condition))%len(domains)]
            kinds=sorted(pools[domain])
            ordinal=int(base)//len(domains)+int(condition)*73
            kind=kinds[ordinal%len(kinds)]
            lengths=sorted(pools[domain][kind])
            length=lengths[(ordinal//len(kinds))%len(lengths)]
            choices=pools[domain][kind][length]
            draw=seed64(self.manifest['dataset_id'],'noise',split,int(cid),int(condition),int(epoch))
            rng=np.random.default_rng(draw)
            nid=int(choices[int(rng.integers(len(choices)))])
            bankrow=int(self.noise_index['row_id'][nid])
            crop=int(self.noise_index['crop_start'][nid])
            length=int(self.noise_index['length'][nid])
            templates.append(self.noise_values[bankrow,crop:crop+length])
            starts.append(int(rng.integers(2048-length+1)))
            seeds.append(seed64(draw,'channel_perturbation'))
            nids.append(nid)
        result=mix_and_labels(clean,templates,starts,snrs[conditions],scheme,seeds,
                sampling_rate_hz=self.config['sampling_rate_hz'],
                cutoff_hz=self.config['frequency']['cutoff_hz'],
                perturbation_std=self.config['noise']['perturbation_std'])
        nids=np.asarray(nids,np.int64)
        result.update(clean_index=clean_ids,source_id=self.index['source_id'][clean_ids],
                      source_row=self.index['source_row'][clean_ids],
                      noise_index=nids,noise_bank_row=self.noise_index['row_id'][nids],
                      class_id=self.noise_index['type_id'][nids].astype(np.int16),
                      domain_id=self.noise_index['domain_id'][nids].astype(np.int8),
                      noise_start=np.asarray(starts,np.int16),
                      noise_length=self.noise_index['length'][nids].astype(np.int16),
                      noise_crop_start=self.noise_index['crop_start'][nids].astype(np.int16),
                      draw_seed=np.asarray(seeds,np.uint64),
                      sample_index=positions,snr_condition_index=conditions.astype(np.int8),
                      is_clean=np.zeros(len(positions),np.uint8))
        # Validate EVERY emitted SNR, not just examples. Same raw coordinate scope.
        wanted=snrs[conditions]
        error=(np.max(np.abs(result['snr_per_channel_db']-wanted[:,None])) if scheme=='A' else
               max(np.max(np.abs(result[k]-wanted)) for k in ['snr_total_db','snr_e_db','snr_h_db']))
        if error>1e-3:raise ValueError(f'Exported SNR error {error}')
        return result

    def control_batch(self,split,positions):
        positions=np.asarray(positions,np.int64)
        ids=self.split_rows[split][positions]
        clean=self.read_clean(ids)
        scale=np.sqrt(np.mean(clean.astype(np.float64)**2,axis=-1,keepdims=True)).astype(np.float32)
        if not np.isfinite(scale).all() or np.any(scale<=0):raise ValueError('Invalid clean control scale')
        target=clean/scale
        low,high=split_frequency_target(target,self.config['sampling_rate_hz'],self.config['frequency']['cutoff_hz'])
        n=len(ids)
        return {'input':target.copy(),'target':target,'target_low':low,'target_high':high,
                'noise_mask':np.zeros_like(target,np.uint8),'scale':scale,
                'class_id':np.zeros(n,np.int16),'domain_id':np.full(n,-1,np.int8),
                'clean_index':ids,'source_id':self.index['source_id'][ids],'source_row':self.index['source_row'][ids],
                'sample_index':positions,'noise_index':np.full(n,-1,np.int64),'noise_bank_row':np.full(n,-1,np.int64),
                'noise_start':np.full(n,-1,np.int16),'noise_length':np.zeros(n,np.int16),
                'is_clean':np.ones(n,np.uint8),'snr_target_db':np.full(n,np.nan,np.float32)}

    def clean_batch(self,split,task,positions,epoch=0):
        positions=np.asarray(positions,np.int64)
        ids=self.split_rows[split][positions]
        values=self.read_clean(ids)
        seeds=np.asarray([seed64(self.manifest['dataset_id'],task,split,int(i),int(epoch)) for i in ids],np.uint64)
        settings={'sample_indices':positions,
                  'horizon_choices':self.config['clean_tasks']['forecasting_horizons'],
                  'gap_choices':self.config['clean_tasks']['interpolation_gaps'],
                  'missing_ratio_choices':self.config['clean_tasks']['imputation_ratios'],
                  'mean_gap_choices':self.config['clean_tasks']['imputation_mean_gaps']}
        result=clean_task_labels(values,task,seeds,settings)
        metadata=result.pop('metadata')
        result['observed_mask']=result['observed_mask'].astype(np.uint8)
        result['loss_mask']=result['loss_mask'].astype(np.uint8)
        if self.config['clean_tasks']['store_visible_standardized']:
            result.update(visible_standardization(result['input'],result['target'],result['observed_mask']))
        for k,v in metadata.items():
            if k=='task':continue
            if k=='channel_mode':
                result['channel_mode_id']=np.asarray([{'concurrent':0,'separate':1,'single_channel':2}[str(x)] for x in v],np.int8)
            else:result[k]=np.asarray(v)
        result.update(clean_index=ids,source_id=self.index['source_id'][ids],source_row=self.index['source_row'][ids],
                      sample_index=positions,draw_seed=seeds)
        return result

    def batch(self,job,positions):
        if job['kind']=='noise':return self.noise_batch(job['split'],job['scheme'],positions)
        if job['kind']=='control':return self.control_batch(job['split'],positions)
        return self.clean_batch(job['split'],job['task'],positions)


_FACTORY=None
def initialize_worker(root):
    global _FACTORY
    for variable in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:
        os.environ.setdefault(variable,'1')
    _FACTORY=SampleFactory(root)


def materialize_shard(job):
    factory=_FACTORY
    if factory is None:raise RuntimeError('Worker not initialized')
    root=factory.root;cfg=factory.config
    final=root/job['path'];temporary=final.with_name(final.name+'.partial')
    final.parent.mkdir(parents=True,exist_ok=True)
    signature=digest({'dataset_id':factory.manifest['dataset_id'],'job':job})
    if final.is_file():
        with h5py.File(final,'r') as f:
            if f.attrs.get('signature')!=signature or not f.attrs.get('complete',False):
                raise ValueError(f'Existing shard signature mismatch: {final}')
            return {'path':job['path'],'rows':job['stop']-job['start'],'bytes':final.stat().st_size,'reused':True}
    mode='r+' if temporary.is_file() else 'w'
    begun=time.monotonic()
    with h5py.File(temporary,mode) as f:
        if mode=='r+':
            if f.attrs.get('signature')!=signature:raise ValueError('Partial shard signature mismatch')
            finished=int(f.attrs['committed_rows'])
        else:
            finished=0
            f.attrs.update(signature=signature,dataset_id=factory.manifest['dataset_id'],complete=False,
                           committed_rows=0,sampling_rate_hz=150.,sequence_length=2048,
                           kind=job['kind'],split=job['split'],scheme=job.get('scheme',''),task=job.get('task',''),
                           global_start=job['start'],global_stop=job['stop'])
        rows=job['stop']-job['start'];batchsize=int(cfg['materialization']['batch_rows'])
        for local in range(finished,rows,batchsize):
            amount=min(batchsize,rows-local)
            positions=np.arange(job['start']+local,job['start']+local+amount,dtype=np.int64)
            batch=factory.batch(job,positions)
            for key,values in batch.items():
                values=np.asarray(values)
                if values.shape[0]!=amount:raise ValueError(f'Unbatched field {key}: {values.shape}')
                if values.dtype.kind not in 'biuf':raise ValueError(f'Unsupported metadata dtype: {key}/{values.dtype}')
                if values.dtype.kind=='f' and not np.isfinite(values).all() and not (job['kind']=='control' and key=='snr_target_db'):
                    raise ValueError(f'Nonfinite export field {key}')
                if key not in f:
                    chunks=(min(int(cfg['storage']['write_chunk_rows']),rows),)+values.shape[1:]
                    f.create_dataset(key,shape=(rows,)+values.shape[1:],dtype=values.dtype,chunks=chunks,
                                     compression=cfg['storage']['compression'],shuffle=True,
                                     fletcher32=bool(cfg['storage']['fletcher32']))
                f[key][local:local+amount]=values
            # Resume never exposes an uncommitted block as a completed sample.
            f.flush();f.attrs.modify('committed_rows',local+amount);f.flush()
        f.attrs.modify('complete',True);f.flush()
    os.replace(temporary,final)
    return {'path':job['path'],'rows':rows,'bytes':final.stat().st_size,'seconds':time.monotonic()-begun,'reused':False}


def shard_jobs(manifest):
    rows=int(manifest['config']['materialization']['shard_rows'])
    snr_count=len(manifest['config']['noise']['snr_db'])
    jobs=[]
    for split in SPLITS:
        n=int(manifest['clean']['totals'][split])
        specifications=[]
        for scheme in manifest['config']['noise']['schemes']:
            specifications.append(('noise',n*snr_count,f'offline/noise_tasks/{scheme}/{split}',{'scheme':scheme}))
        if manifest['config']['materialization']['clean_control_per_clip']:
            specifications.append(('control',n,f'offline/clean_controls/{split}',{}))
        for task in ['forecasting','interpolation','imputation']:
            specifications.append(('clean',n,f'offline/clean_tasks/{task}/{split}',{'task':task}))
        for kind,count,folder,extra in specifications:
            for k,start in enumerate(range(0,count,rows)):
                jobs.append({'kind':kind,'split':split,'start':start,'stop':min(start+rows,count),
                             'path':f'{folder}/shard_{k:05d}.h5',**extra})
    return jobs
