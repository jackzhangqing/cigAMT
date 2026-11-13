"""One isolated experiment per GPU; A/C/SNR evaluation does not multiply training jobs."""
from __future__ import annotations
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from .utils import TASKS,NOISE_TASKS,write_json


def jobs_from_config(config,tasks=None,models=None,schemes=None,seeds=None,include_classical=None):
    tasks=tasks or config.get('tasks',list(TASKS));seeds=seeds or config.get('seeds',[42,43,44])
    schemes=schemes or config.get('train_schemes',['AC']);jobs=[]
    for task in tasks:
        names=models or config['models_by_task'][task]
        for name in names:
            allowed=('linear','tcn','unet1d','bigru','patch_transformer','spectral_fno','amt_maskformer')+(('dlinear',) if task=='forecasting' else ())
            if name not in allowed:raise ValueError(f'Model {name} is unsupported for {task}; choose {allowed}')
            for scheme in schemes if task in NOISE_TASKS else ['none']:
                for seed in seeds:jobs.append({'task':task,'model':name,'scheme':scheme,'seed':int(seed),'classical':False})
        if (include_classical if include_classical is not None else config.get('include_classical',True)):
            for name in config.get('classical_by_task',{}).get(task,[]):
                jobs.append({'task':task,'model':name,'scheme':'none','seed':0,'classical':True})
    identities=[tuple(j.values()) for j in jobs]
    if len(identities)!=len(set(identities)):raise ValueError('Duplicate experiment in suite specification')
    return jobs


def job_path(root,j):
    return Path(root)/j['task']/j['model']/('train_'+j['scheme'])/('seed_'+str(j['seed']))


def mapped_device(name):
    if name=='cpu':return os.environ.copy(),'cpu'
    if not name.startswith('cuda:'):raise ValueError('Device must be cpu or cuda:N')
    index=int(name.split(':')[1]);env=os.environ.copy();visible=env.get('CUDA_VISIBLE_DEVICES')
    if visible:
        entries=[x.strip() for x in visible.split(',')]
        if index>=len(entries):raise ValueError('Device index outside CUDA_VISIBLE_DEVICES')
        target=entries[index]
    else:target=str(index)
    env['CUDA_VISIBLE_DEVICES']=target
    return env,'cuda:0'


def stop_process(proc):
    if proc.poll() is not None:return
    if os.name!='nt':
        try:os.killpg(proc.pid,signal.SIGTERM)
        except ProcessLookupError:return
    else:
        try:
            import psutil
            parent=psutil.Process(proc.pid)
            for child in reversed(parent.children(recursive=True)):
                try:child.terminate()
                except psutil.NoSuchProcess:pass
            parent.terminate()
        except ImportError:proc.terminate()


def launch(config_path,config,data_root,output_root,devices,jobs,dry_run=False,skip_existing=True,num_workers=None):
    devices=list(dict.fromkeys(devices))
    if not devices:raise ValueError('At least one device required')
    output_root=Path(output_root);output_root.mkdir(parents=True,exist_ok=True)
    write_json(output_root/'suite_plan.json',{'training_jobs':sum(not j['classical'] for j in jobs),
               'classical_jobs':sum(j['classical'] for j in jobs),'devices':devices,'jobs':jobs,
               'data_access':'direct completed HDF5 fields, no dataset manifest used'})
    print(f"{sum(not j['classical'] for j in jobs)} training jobs + {sum(j['classical'] for j in jobs)} classical jobs; {len(devices)} concurrent slots",flush=True)
    if dry_run:
        for j in jobs:print(str(job_path(output_root,j)),flush=True)
        return 0
    pending=list(jobs);active={};completed=[];failed=[];script=Path(__file__).parents[1]/'run.py'
    try:
        while pending or active:
            for device in devices:
                if device in active or not pending:continue
                job=pending.pop(0);where=job_path(output_root,job);where.mkdir(parents=True,exist_ok=True)
                env,child_device=mapped_device(device)
                command=[sys.executable,'-B','-u',str(script),'classical' if job['classical'] else 'train',
                         '--config',str(Path(config_path).resolve()),'--data-root',str(Path(data_root).resolve()),
                         '--output',str(where.resolve()),'--task',job['task'],'--model',job['model'],
                         '--scheme',job['scheme'],'--seed',str(job['seed']),'--device',child_device,'--eval-after']
                if skip_existing:command+=['--skip-existing']
                if num_workers is not None:command+=['--num-workers',str(num_workers)]
                log=open(where/'launcher.log','a',encoding='utf-8',buffering=1)
                log.write('\n'+str({'device':device,'CUDA_VISIBLE_DEVICES':env.get('CUDA_VISIBLE_DEVICES'),'command':command})+'\n');log.flush()
                proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt')
                active[device]=(proc,log,job,where);print(f'[{device}] START {where} | PID={proc.pid}',flush=True)
            for device,(proc,log,job,where) in list(active.items()):
                code=proc.poll()
                if code is None:continue
                log.close();del active[device]
                if code==0:completed.append(job);print(f'[{device}] DONE {where}',flush=True)
                elif code==3:print(f'[{device}] ACTIVE ELSEWHERE {where}',flush=True)
                else:failed.append({'job':job,'exit_code':code,'log':str(where/'launcher.log')});print(f'[{device}] FAILED {where}; see launcher.log',flush=True)
                write_json(output_root/'suite_status.json',{'completed':completed,'failed':failed,'pending':len(pending),'active':len(active)})
            if active:time.sleep(.5)
    except KeyboardInterrupt:
        for proc,log,job,where in active.values():stop_process(proc);log.close()
        print('Interrupted. Completed epochs are preserved; repeat the same command to resume.',flush=True)
        return 130
    return 1 if failed else 0
