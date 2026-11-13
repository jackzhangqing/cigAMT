from __future__ import annotations
import contextlib
import csv
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from .data import H5Dataset, BlockShuffleSampler
from .specialist_models import build_model
from .losses import task_loss
from .metrics import compute_metrics, MetricAccumulator
from .utils import (NOISE_TASKS, RunLock, RowWriter, write_json, read_json, digest,
                    code_fingerprint, seed_all, environment, write_table, clean_json)

TENSOR_KEYS={'input','target','observed_mask','loss_mask','scale','noise_mask',
             'class_id','target_low','target_high','visible_center','visible_scale'}
META_KEYS=('sample_uid','scheme','shard','row','sample_index','clean_index','source_id',
           'source_row','domain_id','class_id','noise_length','is_clean','snr_target_db',
           'condition_index','horizon','gap_length','missing_ratio_nominal',
           'missing_ratio_actual','mean_gap','channel_mode_id')


def collate(samples):
    out={}
    for key in samples[0]:
        values=[s[key] for s in samples]
        if isinstance(values[0],(str,Path)):out[key]=[str(v) for v in values]
        else:
            try:out[key]=np.stack(values)
            except (TypeError,ValueError):out[key]=values
        if key in TENSOR_KEYS and isinstance(out[key],np.ndarray):
            # Integers stay integer; masks/coordinates become float in task_loss.
            out[key]=torch.from_numpy(np.ascontiguousarray(out[key]))
    return out


def cpu_numpy(batch):
    return {k:v.detach().cpu().numpy() if torch.is_tensor(v) else v for k,v in batch.items()}


def on_device(batch,device):
    return {k:v.to(device,non_blocking=True) if torch.is_tensor(v) else v for k,v in batch.items()}


def network_input(batch,task):
    x=batch['input'].float()
    return x if task in NOISE_TASKS else {'input':x,'observed_mask':batch['observed_mask'].float()}


def amp_context(device,precision):
    if device.type!='cuda' or precision=='fp32':return contextlib.nullcontext()
    return torch.autocast(device_type='cuda',dtype=torch.bfloat16 if precision=='bf16' else torch.float16)


def loader(dataset,config,training=False,seed=42):
    sampler=BlockShuffleSampler(dataset,seed=seed,block_size=config.get('block_size',256),
                               shuffle=training,samples_per_epoch=config.get('samples_per_epoch') if training else None)
    workers=int(config.get('num_workers',0))
    args=dict(dataset=dataset,batch_size=int(config.get('batch_size',64)),sampler=sampler,
              num_workers=workers,collate_fn=collate,pin_memory=True,
              persistent_workers=workers>0,drop_last=False,
              generator=torch.Generator().manual_seed(int(seed)+(17 if training else 31)))
    if workers:args['prefetch_factor']=int(config.get('prefetch_factor',2))
    return DataLoader(**args),sampler


def make_dataset(root,task,split,scheme,config,max_samples=None):
    sources=config.get('train_source_names') if split=='train' else config.get('evaluation_source_names')
    if sources is None:sources=config.get('source_names')
    return H5Dataset(root,task,split,scheme=scheme if task in NOISE_TASKS else 'none',
                     include_controls=config.get('include_controls',True),normalization='visible',
                     snr_db=config.get('snr_db'),source_names=sources,
                     max_samples=max_samples,selection_seed=config.get('selection_seed',2025),
                     selection_mode=config.get('selection_mode','exact') if max_samples is not None else 'exact')


def save_torch(path,payload):
    path=Path(path);temporary=path.with_name(path.name+'.partial')
    torch.save(payload,str(temporary));os.replace(str(temporary),str(path))


def architecture_options(config,name):
    options=dict(config.get('model',{}))
    options.update(config.get('model_overrides',{}).get(name,{}))
    return options


def load_torch(path,device):
    from export_pretrained import safe_load
    # Always load through the restricted known-type loader; move tensors later.
    return safe_load(path)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch'].cpu())
    if s.get('cuda') and torch.cuda.is_available():torch.cuda.set_rng_state_all([x.cpu() for x in s['cuda']])


def slice_batch(batch, start, stop):
    """Slice batched fields while preserving metadata order."""
    n=len(batch['input'])
    return {k:(v[start:stop] if isinstance(v,(torch.Tensor,np.ndarray,list,tuple))
               and len(v)==n else v) for k,v in batch.items()}


def micro_size(config, name):
    value=config.get('micro_batch_size_by_model',{}).get(name,config.get('batch_size',256))
    if isinstance(value,bool) or not isinstance(value,int) or value<1:
        raise ValueError('micro_batch_size_by_model must contain positive integers')
    return value


def batch_prediction(model, batch, task, device, precision, micro_batch_size):
    """Inference without retaining attention for the full effective batch."""
    outputs={}
    for start in range(0,len(batch['input']),micro_batch_size):
        b=on_device(slice_batch(batch,start,start+micro_batch_size),device)
        with amp_context(device,precision):piece=model(network_input(b,task))
        for key,value in piece.items():outputs.setdefault(key,[]).append(value.detach().float().cpu())
    return {key:torch.cat(values,dim=0) for key,values in outputs.items()}


def backward_batch(model, cpu, task, device, precision, scaler, micro_batch_size):
    """Accumulate case-mean gradients; the caller performs one optimizer step."""
    n=len(cpu['input']);total=0.0
    for start in range(0,n,micro_batch_size):
        b=on_device(slice_batch(cpu,start,start+micro_batch_size),device)
        with amp_context(device,precision):out=model(network_input(b,task));loss=task_loss(task,b,out)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; previous complete epoch is preserved')
        weight=len(b['input'])/n
        scaler.scale(loss*weight).backward()
        total+=float(loss.detach())*weight
    return total


def validation_loss(model,data,task,device,precision,micro_batch_size=256):
    model.eval();total=count=0
    with torch.inference_mode():
        for cpu in data:
            for start in range(0,len(cpu['input']),micro_batch_size):
                b=on_device(slice_batch(cpu,start,start+micro_batch_size),device)
                with amp_context(device,precision):out=model(network_input(b,task));loss=task_loss(task,b,out)
                n=len(b['input']);total+=float(loss)*n;count+=n
    if not count:raise ValueError('Empty validation data')
    return total/count


def calibration(model,data,device,precision,bins=1000):
    """Choose a global localization threshold on validation only; histogram resolution .001."""
    positive=np.zeros(bins,np.int64);negative=positive.copy();model.eval()
    with torch.inference_mode():
        for cpu in data:
            b=on_device(cpu,device)
            with amp_context(device,precision):logits=model(b['input'].float())['logits']
            p=logits.float().sigmoid().mean(1).cpu().numpy()
            y=cpu['noise_mask'].numpy().mean(1)>.5
            ix=np.minimum((p*bins).astype(np.int64),bins-1)
            positive+=np.bincount(ix[y],minlength=bins);negative+=np.bincount(ix[~y],minlength=bins)
    tp=np.cumsum(positive[::-1])[::-1];fp=np.cumsum(negative[::-1])[::-1];fn=positive.sum()-tp
    f1=2*tp/np.maximum(2*tp+fp+fn,1)
    # Deterministic tie: threshold nearest 0.5 among maximizers.
    candidates=np.flatnonzero(f1==f1.max());best=candidates[np.argmin(abs(candidates/bins-.5))]
    return {'threshold':float(best/bins),'validation_point_micro_f1':float(f1[best]),
            'bins':bins,'scope':'fixed validation subset only; no test-based calibration'}


def train(root,output,task,name,scheme,seed,config,device='cuda:0',resume=True,skip_existing=True):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    with RunLock(output):
        device=torch.device(device)
        if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
        if device.type=='cpu' and config.get('precision','bf16')!='fp32':
            raise ValueError('CPU training requires precision=fp32; use the smoke configuration or --precision fp32')
        if config.get('precision','bf16')=='bf16' and device.type=='cuda' and not torch.cuda.is_bf16_supported():
            raise RuntimeError('bf16 unsupported on this GPU; select fp16 or fp32 explicitly')
        seed_all(seed,config.get('deterministic',False));torch.set_num_threads(int(config.get('cpu_threads',4)))
        training=make_dataset(root,task,'train',scheme,config,config.get('max_train_samples'))
        validation=make_dataset(root,task,'validation',scheme,config,config.get('max_validation_samples'))
        # Execution placement/worker counts may change; optimization and selection may not.
        fixed={k:v for k,v in config.items() if k not in ('num_workers','prefetch_factor','cpu_threads','log_interval')}
        identity={'task':task,'model':name,'train_scheme':scheme,'seed':seed,'config':fixed,
                  'training':training.fingerprints(),'validation':validation.fingerprints(),
                  'code':code_fingerprint()}
        signature=digest(identity)
        if (output/'training_complete.json').exists():
            saved=read_json(output/'training_complete.json')
            if saved['signature']!=signature:raise ValueError('Existing run has different data/code/config. Use a new output directory.')
            if not skip_existing:raise ValueError('Completed training is preserved. Use a new output directory to retrain.')
            print('SKIP completed training '+str(output),flush=True);training.close();validation.close();return saved
        if (output/'last.pt').exists():
            old=load_torch(output/'last.pt','cpu')
            if old['signature']!=signature:raise ValueError('Resume rejected: optimization/data/core code differs. Use a new run directory.')
            if not resume:raise ValueError('Incomplete run exists. Resume or use a new output directory.')
        write_json(output/'run_config.json',{'signature':signature,'identity':identity,'config':config,
                                           'data_root':str(Path(root).resolve()),'environment':environment(),
                                           'train_data':training.describe(),'validation_data':validation.describe()})
        options=architecture_options(config,name)
        model=build_model(name,task,sequence_length=training.sequence_length,**options).to(device)
        optimizer=torch.optim.AdamW(model.parameters(),lr=config.get('learning_rate',3e-4),weight_decay=config.get('weight_decay',1e-4))
        epochs=int(config.get('epochs',30));scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
        precision=config.get('precision','bf16');scaler=torch.cuda.amp.GradScaler(enabled=device.type=='cuda' and precision=='fp16')
        micro_batch_size=micro_size(config,name)
        print(json.dumps({'model':name,'effective_batch_size':int(config.get('batch_size',256)),
                          'micro_batch_size':micro_batch_size,'precision':precision,
                          'loss':'per-case missing-target MSE; official architectures with task adapters'}),flush=True)
        trainloader,sampler=loader(training,config,True,seed);valloader,_=loader(validation,config)
        start=0;best=math.inf;bad=0;history=[]
        if (output/'last.pt').exists():
            if not resume:raise ValueError('Incomplete run exists. Resume or use a new output directory.')
            ck=load_torch(output/'last.pt',device)
            if ck['signature']!=signature:raise ValueError('Resume rejected: optimization/data/core code differs. Use a new run directory.')
            model.load_state_dict(ck['model']);optimizer.load_state_dict(ck['optimizer']);scheduler.load_state_dict(ck['scheduler'])
            scaler.load_state_dict(ck['scaler']);restore_rng(ck['rng'])
            start=ck['epoch'];best=ck['best'];bad=ck['bad_epochs'];history=ck['history']
            print('RESUME at epoch '+str(start+1),flush=True)
            if bad>=int(config.get('patience',7)):start=epochs
        epoch0=time.perf_counter();parameters=sum(p.numel() for p in model.parameters())
        for epoch in range(start,epochs):
            sampler.set_epoch(epoch);model.train();total=count=0;t=time.perf_counter()
            for step,cpu in enumerate(trainloader):
                optimizer.zero_grad(set_to_none=True)
                loss=backward_batch(model,cpu,task,device,precision,scaler,micro_batch_size)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(),config.get('gradient_clip',1.0))
                scaler.step(optimizer);scaler.update()
                n=len(cpu['input']);total+=loss*n;count+=n
                if (step+1)%int(config.get('log_interval',100))==0:
                    print(json.dumps({'epoch':epoch+1,'batch':step+1,'train_loss_so_far':total/count}),flush=True)
            if not count:raise ValueError('No training batches')
            train_seconds=time.perf_counter()-t
            print(json.dumps({'epoch':epoch+1,'phase':'validation','rows':len(validation)}),flush=True)
            val=validation_loss(model,valloader,task,device,precision,micro_batch_size);scheduler.step()
            row={'epoch':epoch+1,'train_loss':total/count,'validation_loss':val,'seconds':time.perf_counter()-t,
                 'train_seconds':train_seconds,'train_rows':count,'rows_per_second':count/max(train_seconds,1e-9),
                 'learning_rate':optimizer.param_groups[0]['lr']}
            improved=val<best-float(config.get('min_delta',0))
            if improved:best=val;bad=0
            else:bad+=1
            history.append(row);write_table(output/'history.csv',history)
            payload={'signature':signature,'task':task,'name':name,'scheme':scheme,'seed':seed,
                     'config':config,'model_options':options,'sequence_length':training.sequence_length,'dataset_id':training.dataset_id,
                     'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                     'scaler':scaler.state_dict(),'rng':rng_state(),'epoch':epoch+1,'best':best,'bad_epochs':bad,'history':history}
            if improved:save_torch(output/'best.pt',payload)
            save_torch(output/'last.pt',payload);print(json.dumps(row),flush=True)
            if bad>=int(config.get('patience',7)):break
        bestck=load_torch(output/'best.pt',device);model.load_state_dict(bestck['model'])
        cal=calibration(model,valloader,device,precision) if task=='noise_localization' else {'threshold':.5,'scope':'not applicable'}
        write_json(output/'calibration.json',cal)
        result={'signature':signature,'state':'training_complete','task':task,'model':name,'train_scheme':scheme,
                'seed':seed,'parameters':parameters,'best_epoch':bestck['epoch'],'best_validation_loss':best,
                'epochs_completed':len(history),'seconds_this_invocation':time.perf_counter()-epoch0,
                'max_gpu_memory_bytes':torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
                'dataset_id':training.dataset_id,'code':code_fingerprint()}
        write_json(output/'training_complete.json',result);training.close();validation.close();return result


class Moments:
    """Bounded-memory per-case means, finite counts, class confusion and point counts."""
    def __init__(self):self.accumulator=MetricAccumulator()
    def update(self,row):
        self.accumulator.update(row)
    def result(self):
        s=self.accumulator.summary()
        out={'n':s['count'],'mean':{k:v['mean'] for k,v in s['metrics'].items()},
             'std':{k:v['std'] for k,v in s['metrics'].items()},
             'valid_counts':{k:v['count'] for k,v in s['metrics'].items()}}
        for key in ('pooled_pointwise_f1','pooled_pointwise_iou','localization_totals'):
            if key in s:out[key]=s[key]
        if 'confusion_matrix' in s:
            out.update(confusion_matrix=s['confusion_matrix'],macro_f1_present=s['macro_f1_present_true'],
                       macro_f1_all11=s['macro_f1_all_classes'],accuracy=s['pooled_accuracy'])
        return out


def strata(row):
    for key in ('scheme','is_clean','snr_target_db','domain_id','class_id','noise_length','source_id',
                'horizon','gap_length','missing_ratio_nominal','mean_gap','channel_mode_id'):
        v=row.get(key)
        if v is not None and v!='' and not (isinstance(v,float) and not math.isfinite(v)):
            yield key,str(v)
    if row.get('snr_target_db') is not None and not row.get('is_clean',0):
        yield 'scheme_x_snr',str(row.get('scheme'))+'|'+str(row['snr_target_db'])
    if row.get('class_id',-1)>=0 and row.get('noise_length',0)>0:
        yield 'class_x_length',str(row['class_id'])+'|'+str(row['noise_length'])
    if row.get('is_clean',0)==0 and row.get('snr_target_db') is not None:
        yield 'scheme_x_snr_x_domain',str(row.get('scheme'))+'|'+str(row['snr_target_db'])+'|'+str(row.get('domain_id'))
    if row.get('missing_ratio_nominal',-1)>0:
        yield 'imputation_condition','|'.join(str(row.get(k)) for k in ('missing_ratio_nominal','mean_gap','channel_mode_id'))


def evaluate(root,run,split='test',scheme='A',device='cuda:0',config=None,max_samples=None,
             skip_existing=True,classical=None,task=None,output_tag=None):
    from .classical import classical_predict
    run=Path(run);config=dict(config or {});device=torch.device(device)
    if classical:
        name=classical;seed=0;train_scheme='not_trained';threshold=.5;model=None;ck=None
    else:
        info=read_json(run/'run_config.json');ck=load_torch(run/'best.pt',device)
        if info['identity']['code']!=code_fingerprint():
            # Publication copies adapt paths/readers; model.load_state_dict remains strict.
            if not info.get('publication_export', False):
                raise ValueError('Use a publication-exported run_config.json for this portable code copy.')
        task=ck['task'];name=ck['name'];seed=ck['seed'];train_scheme=ck['scheme']
        merged=dict(ck['config']);merged.update(config);config=merged
        model=build_model(name,task,sequence_length=ck['sequence_length'],
                          **ck.get('model_options',architecture_options(ck['config'],name))).to(device)
        model.load_state_dict(ck['model']);model.eval()
        threshold=read_json(run/'calibration.json')['threshold']
    data=make_dataset(root,task,split,scheme,config,max_samples)
    if ck and data.dataset_id!=ck['dataset_id']:raise ValueError('Evaluation dataset_id differs from training')
    if output_tag is not None:
        import re
        if not re.fullmatch(r'[A-Za-z0-9_-]+',output_tag):raise ValueError('Evaluation tag allows ASCII letters, numbers, underscore and hyphen only')
    where=run/'evaluation'/((output_tag+'__' if output_tag else '')+split+'_'+(scheme if task in NOISE_TASKS else 'none'))
    evaluation_identity={'dataset':data.fingerprints(),'checkpoint_signature':ck['signature'] if ck else None,
                         'classical':classical,'task':task,'code':code_fingerprint(),'threshold':threshold,
                         'inference_device_type':'cpu' if classical else device.type,
                         'inference_precision':'numpy_fp32' if classical else ('fp32' if device.type=='cpu' else config.get('precision','bf16')),
                         'batch_size':int(config.get('batch_size',64))}
    signature=digest(evaluation_identity)
    with RunLock(where):
        if (where/'summary.json').exists():
            old=read_json(where/'summary.json')
            if old['evaluation_signature']!=signature:raise ValueError('Existing evaluation uses other data/precision/settings. Preserve it and use a new --tag.')
            if skip_existing:
                data.close();print('SKIP evaluation '+str(where),flush=True);return old
        stream,_=loader(data,config);overall=Moments();groups={};writer=RowWriter(where/'evaluation_rows.csv.partial')
        anchors=set(np.linspace(0,len(data)-1,min(len(data),int(config.get('examples',8))),dtype=int).tolist())
        cursor=0;start=time.perf_counter();conf=np.zeros((11,11),np.int64)
        try:
            with torch.inference_mode():
                for cpu in stream:
                    bnp=cpu_numpy(cpu)
                    if classical:outnp=classical_predict(classical,task,bnp)
                    else:
                        out=batch_prediction(model,cpu,task,device,config.get('precision','bf16'),micro_size(config,name))
                        outnp={k:v.numpy() for k,v in out.items()}
                    metrics=compute_metrics(task,bnp,outnp,threshold=threshold)
                    for j,metric in enumerate(metrics):
                        row={'dataset_id':data.dataset_id,'task':task,'model':name,'train_scheme':train_scheme,
                             'seed':seed,'split':split,'evaluation_index':cursor+j}
                        for key in META_KEYS:
                            if key in bnp:
                                val=bnp[key][j];row[key]=val.item() if isinstance(val,np.generic) else val
                        row.update(metric);writer.write(row);overall.update(metric)
                        for key,value in strata(row):groups.setdefault((key,value),Moments()).update(metric)
                        if cursor+j in anchors:
                            folder=where/'examples';folder.mkdir(parents=True,exist_ok=True)
                            arrays={k:v[j] for k,v in bnp.items() if isinstance(v,np.ndarray) and v.ndim>0}
                            arrays.update({'output_'+k:v[j] for k,v in outnp.items()})
                            arrays['metadata_json']=np.asarray(json.dumps(clean_json(row),allow_nan=False))
                            np.savez_compressed(folder/f'case_{cursor+j:08d}.npz',**arrays)
                    cursor+=len(metrics)
                    if cursor%max(int(config.get('batch_size',64))*100,1)==0:
                        print(json.dumps({'evaluation':str(where),'rows':cursor,'total':len(data)}),flush=True)
        finally:writer.close();data.close()
        os.replace(str(where/'evaluation_rows.csv.partial'),str(where/'evaluation_rows.csv'))
        group_rows=[]
        for (key,value),acc in sorted(groups.items()):
            r=acc.result();group_rows.append({'stratum':key,'value':value,'n':r['n'],**r['mean'],
                                             **({'macro_f1_present':r['macro_f1_present'],'macro_f1_all11':r['macro_f1_all11']} if 'macro_f1_present' in r else {})})
        write_table(where/'strata.csv',group_rows)
        result={'state':'evaluation_complete','evaluation_signature':signature,'dataset_id':data.dataset_id,
                'task':task,'model':name,'train_scheme':train_scheme,'eval_scheme':scheme if task in NOISE_TASKS else 'none',
                'seed':seed,'split':split,'rows':cursor,'data':data.describe(),'metrics':overall.result(),
                'data_fingerprints':data.fingerprints(),'evaluation_identity':evaluation_identity,
                'threshold':threshold,'seconds':time.perf_counter()-start,'environment':environment(),
                'classical':bool(classical),'rows_path':'evaluation_rows.csv','scope':'all selected rows; undefined metrics excluded with valid_counts'}
        write_json(where/'summary.json',result);print(json.dumps({'evaluation_done':str(where),'rows':cursor}),flush=True);return result
