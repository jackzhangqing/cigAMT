"""Dataset preparation, capacity plan, pilot and resumable full materialization."""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import datetime
import os
import platform
import shutil
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import h5py
import numpy as np
from .utils import json_read,json_write,digest,file_sha,make_clean_index,SPLITS,TASKS
from .clean_sources import prepare_clean
from .noise_sources import prepare_noise
from .coverage_schedule import prepare_schedule
from .materialize import shard_jobs,initialize_worker,materialize_shard,SampleFactory


@contextmanager
def build_lock(root):
    """OS-managed lock is automatically released if the builder terminates."""
    path=Path(root)/'.build.lock'
    with path.open('a+b') as handle:
        handle.seek(0,2)
        if handle.tell()==0:handle.write(b'1');handle.flush()
        handle.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('Another builder is actively writing this dataset root') from exc
        try:yield
        finally:
            handle.seek(0)
            if os.name=='nt':msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def check_identity(manifest):
    if digest(manifest['identity'])!=manifest['dataset_id']:raise ValueError('Dataset identity metadata mismatch')
    for name,expected in manifest['identity']['kernel_sha256'].items():
        if file_sha(Path(__file__).parent/name)!=expected:raise ValueError('Generation kernel differs from frozen dataset: '+name)


def capacity_plan(config,clean_count,noise_count=718359,free_bytes=None):
    n=int(clean_count);ns=len(config['noise']['snr_db']);schemes=len(config['noise']['schemes'])
    waveform=4*2048*4;mask=4*2048
    noisy_row=4*waveform+mask+256
    # Legacy input/target plus observed-only input/target and two uint8 masks.
    clean_row=waveform*(4 if config['clean_tasks']['store_visible_standardized'] else 2)+2*mask+384
    items={'noise_tasks_shared_four_tasks':n*ns*schemes*noisy_row,
           'clean_controls_shared_schemes':n*noisy_row,
           'clean_tasks_three_tasks':n*3*clean_row,
           'clean_source_banks_raw_and_normalized_upper':n*2*waveform,
           'noise_bank_and_raw_upper':int(noise_count)*2048*4+60000*2048*4,
           'metadata_code_audits_and_hdf5_overhead_allowance':10000000000}
    total=sum(items.values());reserve=int(config['storage']['minimum_free_reserve_bytes'])
    return {'estimate_kind':'uncompressed payload upper estimate; compression savings not required',
            'clean_candidate_count':n,'noisy_rows':n*ns*schemes,'clean_control_rows':n,
            'clean_task_rows_each':n,'bytes_per_noisy_row_with_metadata_allowance':noisy_row,
            'bytes_per_clean_task_row_with_metadata_allowance':clean_row,
            'components_bytes':items,'planned_bytes':total,'planned_decimal_tb':total/1e12,
            'free_bytes_at_estimate':free_bytes,'reserved_free_bytes':reserve,
            'fits_with_reserve':None if free_bytes is None else total+reserve<free_bytes,
            'ram_policy':'bounded batches and shards; never load all offline samples into RAM'}


def candidate_count(audit_dir):
    old=json_read(audit_dir/'existing_clean_inventory.json')
    new=json_read(audit_dir/'clean_new_inventory.json')
    return sum(int(s['data_shape'][0]) for s in old['sources'].values())+sum(int(s['clips_before_quality_filter']) for s in new['sources'].values())


def prepare(root,config,audit_dir):
    root=Path(root).resolve();audit_dir=Path(audit_dir).resolve()
    before=capacity_plan(config,candidate_count(audit_dir),free_bytes=shutil.disk_usage(root).free)
    json_write(root/'qa/capacity_plan_before_banks.json',before)
    if not before['fits_with_reserve']:raise ValueError('Insufficient free disk space for complete package and reserve')
    clean=prepare_clean(root,config,json_read(audit_dir/'existing_clean_inventory.json'),json_read(audit_dir/'clean_new_inventory.json'))
    noise=prepare_noise(root,config,audit_dir)
    (root/'metadata').mkdir(exist_ok=True)
    np.savez_compressed(root/'metadata/clean_index.npz',**make_clean_index(clean))
    kernel=['utils.py','protocols.py','materialize.py','coverage_schedule.py','clean_sources.py','noise_sources.py']
    package=Path(__file__).parent
    identity={'dataset_name':config['dataset_name'],'config':config,'clean_manifest_id':clean['manifest_id'],
              'noise_build_signature':noise.get('build_signature'),'noise_manifest':noise,
              'kernel_sha256':{p:file_sha(package/p) for p in kernel}}
    manifest={'schema_version':1,'dataset_id':digest(identity),'identity':identity,
              'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'config':config,'clean':clean,'noise':noise,'tasks':TASKS,
              'data_model':{'four_noise_tasks_share_physical_shards':True,'controls_shared_A_C':True,
                            'clean_task_default_normalization':'visible','snr_scope':'whole_2048_window',
                            'input_does_not_include_true_noise_mask_or_clean_target':True},
              'environment':{'python':platform.python_version(),'numpy':np.__version__,'h5py':h5py.__version__},
              'status':'banks_prepared'}
    if (root/'manifest.json').is_file():
        prior=json_read(root/'manifest.json')
        if prior['dataset_id']!=manifest['dataset_id']:raise ValueError('Existing dataset identity differs; use a separate root')
        manifest['created_at']=prior['created_at']
    json_write(root/'manifest.json',manifest)
    manifest['noise_schedule']=prepare_schedule(root,manifest)
    manifest['offline_plan']=capacity_plan(config,clean['usable_count'])
    manifest['offline_jobs']=shard_jobs(manifest)
    json_write(root/'manifest.json',manifest)
    json_write(root/'metadata/offline_jobs.json',manifest['offline_jobs'])
    print('PREPARED',clean['totals'],'noise_templates',noise['count'],'dataset_id',manifest['dataset_id'],flush=True)
    return manifest


def materialize(root,manifest,workers=4,max_shards=None):
    check_identity(manifest)
    manifest['status']='materializing';json_write(Path(root)/'manifest.json',manifest)
    jobs=manifest['offline_jobs'];selected=jobs if max_shards is None else jobs[:max_shards]
    status_path=Path(root)/'build_status.json'
    prior=json_read(status_path) if status_path.exists() else {}
    results={r['path']:r for r in prior.get('shards',[]) if (Path(root)/r['path']).is_file()}
    started=time.monotonic();pending=[]
    for job in selected:
        final=Path(root)/job['path']
        if final.is_file():
            with h5py.File(final,'r') as f:
                signature=digest({'dataset_id':manifest['dataset_id'],'job':job})
                if f.attrs.get('signature')!=signature or not f.attrs.get('complete',False):raise ValueError(f'Invalid existing shard: {final}')
            results[job['path']]={'path':job['path'],'rows':job['stop']-job['start'],'bytes':final.stat().st_size,'reused':True}
        else:pending.append(job)
    def save(state):
        json_write(status_path,{'dataset_id':manifest['dataset_id'],'state':state,'expected_shards':len(jobs),
                   'execution_workers':int(workers),
                   'completed_shards':len(results),'completed_rows':sum(r['rows'] for r in results.values()),
                   'stored_shard_bytes':sum(r['bytes'] for r in results.values()),'shards':list(results.values()),
                   'elapsed_this_invocation_seconds':time.monotonic()-started})
    save('materializing')
    try:
        with futures.ProcessPoolExecutor(max_workers=int(workers),initializer=initialize_worker,initargs=(str(root),)) as pool:
            submitted={pool.submit(materialize_shard,j):j for j in pending}
            for future in futures.as_completed(submitted):
                result=future.result();results[result['path']]=result;save('materializing')
                print(f"SHARD {len(results)}/{len(jobs)} {result['path']} rows={result['rows']} bytes={result['bytes']} seconds={result.get('seconds',0):.1f}",flush=True)
    except BaseException:
        save('interrupted_or_failed');raise
    complete=len(results)==len(jobs);save('materialized' if complete else 'partial_by_request')
    manifest['status']='materialized' if complete else 'partial_by_request';json_write(Path(root)/'manifest.json',manifest)
    return complete


def pilot(root,manifest):
    initialize_worker(root);begin=time.monotonic();results=[]
    factory=SampleFactory(root)
    for split in SPLITS:
        for scheme in ['A','C']:
            positions=np.arange(min(96,len(factory.split_rows[split])*6))
            a=factory.noise_batch(split,scheme,positions);b=factory.noise_batch(split,scheme,positions)
            for k in a:
                if not np.array_equal(a[k],b[k]):raise AssertionError('Noise replay mismatch '+k)
        for task in TASKS[4:]:
            a=factory.clean_batch(split,task,np.arange(min(32,len(factory.split_rows[split]))))
            assert np.all(a['input'][a['observed_mask']==0]==0)
            assert np.all(a['loss_mask']==1-a['observed_mask'])
    factory.close()
    for kind,extra in [('noise',{'scheme':'A'}),('noise',{'scheme':'C'}),('control',{})]+[('clean',{'task':t}) for t in TASKS[4:]]:
        label=extra.get('scheme',extra.get('task','control'))
        job={'kind':kind,'split':'train','start':0,'stop':768,'path':f'qa/pilot/{label}/shard_00000.h5',**extra}
        results.append(materialize_shard(job))
    report={'passed':True,'rows_per_kind':768,'results':results,'seconds':time.monotonic()-begin,
            'scope':'pilot integration, all tasks and replay checks; not full materialization'}
    json_write(Path(root)/'qa/pilot_report.json',report);print('PILOT',report,flush=True)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True,type=Path);p.add_argument('--config',type=Path)
    p.add_argument('--audit-dir',type=Path);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--estimate-only',action='store_true');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--pilot-only',action='store_true');p.add_argument('--resume',action='store_true')
    p.add_argument('--max-shards',type=int,help='QA only: stop after this many shards; dataset remains partial')
    p.add_argument('--skip-verification',action='store_true')
    args=p.parse_args(argv);root=args.root.resolve()
    if args.workers<1:p.error('--workers must be positive')
    if args.max_shards is not None and args.max_shards<1:p.error('--max-shards must be positive')
    if args.estimate_only:
        config=json_read(args.config);n=candidate_count(args.audit_dir)
        volume=root if root.exists() else Path(root.anchor)
        report=capacity_plan(config,n,free_bytes=shutil.disk_usage(volume).free)
        print(report,flush=True);return 0
    root.mkdir(parents=True,exist_ok=True)
    if args.resume and (root/'manifest.json').exists():
        manifest=json_read(root/'manifest.json');check_identity(manifest)
    else:
        if args.config is None or args.audit_dir is None:p.error('Preparing requires --config and --audit-dir')
        if (root/'manifest.json').exists():
            prior=json_read(root/'manifest.json');check_identity(prior)
            if digest(prior['config'])!=digest(json_read(args.config)):
                raise ValueError('A dataset configuration is immutable; use a new root for another version')
        manifest=prepare(root,json_read(args.config),args.audit_dir)
    if args.prepare_only:return 0
    pilot(root,manifest)
    if args.pilot_only:return 0
    complete=materialize(root,manifest,args.workers,args.max_shards)
    if complete and not args.skip_verification:
        from .verify import verify_dataset
        report=verify_dataset(root)
        if not report['passed']:raise ValueError('Final verification failed')
        manifest['status']='verified_complete';json_write(root/'manifest.json',manifest)
    from .statistics import write_statistics
    write_statistics(root)
    return 0


def locked_main(argv=None):
    # Parse root without changing the full CLI, before acquiring the writer lock.
    argv=sys.argv[1:] if argv is None else argv
    if '-h' in argv or '--help' in argv:return main(argv)
    probe=argparse.ArgumentParser(add_help=False);probe.add_argument('--root',required=True,type=Path)
    probe.add_argument('--estimate-only',action='store_true')
    args,_=probe.parse_known_args(argv)
    if args.estimate_only:return main(argv)
    args.root.mkdir(parents=True,exist_ok=True)
    with build_lock(args.root):return main(argv)


if __name__=='__main__':raise SystemExit(locked_main())
