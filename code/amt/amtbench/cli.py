from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .utils import TASKS,NOISE_TASKS,read_json,write_json


def config_with_args(args):
    config=read_json(args.config)
    out=dict(config.get('training',{}))
    for key in ('batch_size','epochs','num_workers','precision','max_train_samples','max_validation_samples','max_test_samples','samples_per_epoch'):
        if hasattr(args,key) and getattr(args,key) is not None:out[key]=getattr(args,key)
    if getattr(args,'train_sources',None):out['train_source_names']=args.train_sources
    if getattr(args,'eval_sources',None):out['evaluation_source_names']=args.eval_sources
    return config,out


def main(argv=None):
    parser=argparse.ArgumentParser(description='AMT seven-task direct-HDF5 benchmark')
    subs=parser.add_subparsers(dest='command',required=True)
    default_root=Path(__file__).resolve().parents[2]
    default_config=Path(__file__).resolve().parents[1]/'configs/main.json'
    def common(p):
        p.add_argument('--data-root',type=Path,default=default_root)
        p.add_argument('--config',type=Path,default=default_config)
        p.add_argument('--num-workers',type=int)
        p.add_argument('--device',default='cuda:0')
    p=subs.add_parser('inspect');common(p);p.add_argument('--task',choices=TASKS,default='denoising');p.add_argument('--split',choices=['train','validation','test'],default='test');p.add_argument('--scheme',choices=['A','C','AC','none'],default='AC');p.add_argument('--index',type=int,default=0);p.add_argument('--output',type=Path)
    for command in ('train','classical'):
        p=subs.add_parser(command);common(p)
        p.add_argument('--task',choices=TASKS,required=True);p.add_argument('--model',required=True)
        p.add_argument('--scheme',choices=['A','C','AC','none'],default='AC');p.add_argument('--seed',type=int,default=42)
        p.add_argument('--output',type=Path,required=True);p.add_argument('--eval-after',action='store_true')
        p.add_argument('--skip-existing',action='store_true');p.add_argument('--no-resume',action='store_true')
        p.add_argument('--batch-size',type=int);p.add_argument('--epochs',type=int);p.add_argument('--precision',choices=['fp32','fp16','bf16'])
        p.add_argument('--max-train-samples',type=int);p.add_argument('--max-validation-samples',type=int);p.add_argument('--max-test-samples',type=int)
        p.add_argument('--samples-per-epoch',type=int);p.add_argument('--train-sources',nargs='+');p.add_argument('--eval-sources',nargs='+')
    p=subs.add_parser('evaluate');common(p);p.add_argument('--run',type=Path,required=True);p.add_argument('--split',choices=['validation','test'],default='test');p.add_argument('--schemes',nargs='+',choices=['A','C','AC','none'],default=['A','C']);p.add_argument('--max-samples',type=int);p.add_argument('--tag');p.add_argument('--skip-existing',action='store_true')
    p=subs.add_parser('suite');common(p);p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--devices',nargs='+',default=['cuda:0']);p.add_argument('--tasks',nargs='+',choices=TASKS)
    p.add_argument('--models',nargs='+');p.add_argument('--schemes',nargs='+',choices=['A','C','AC']);p.add_argument('--seeds',type=int,nargs='+')
    p.add_argument('--without-classical',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--skip-existing',action='store_true')
    p=subs.add_parser('aggregate');p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('plot');p.add_argument('--run',type=Path);p.add_argument('--root',type=Path);p.add_argument('--output',type=Path,required=True)
    p=subs.add_parser('compare');p.add_argument('--left',type=Path,required=True);p.add_argument('--right',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--metric',default='snr_improvement_db');p.add_argument('--bootstrap',type=int,default=2000);p.add_argument('--seed',type=int,default=2025);p.add_argument('--cluster',default='clean_index');p.add_argument('--cluster-map',type=Path);p.add_argument('--direction',choices=['higher','lower'])
    args=parser.parse_args(argv)
    if args.command=='suite':
        from .suite import jobs_from_config,launch
        cfg,_=config_with_args(args)
        jobs=jobs_from_config(cfg,args.tasks,args.models,args.schemes,args.seeds,False if args.without_classical else None)
        return launch(args.config,cfg,args.data_root,args.output_root,args.devices,jobs,args.dry_run,args.skip_existing,args.num_workers)
    if args.command=='inspect':
        from .data import H5Dataset
        ds=H5Dataset(args.data_root,args.task,args.split,scheme=args.scheme if args.task in NOISE_TASKS else 'none')
        sample=ds.get_metadata(args.index)
        report={'data':ds.describe(),'fingerprints':ds.fingerprints(),'example_metadata':sample}
        if args.output:write_json(args.output,report)
        from .utils import clean_json
        print(json.dumps(clean_json(report),ensure_ascii=False,indent=2));ds.close();return 0
    if args.command in ('train','classical'):
        from .engine import train,evaluate
        from .utils import RunLock
        cfg,training=config_with_args(args)
        scheme=args.scheme if args.task in NOISE_TASKS else 'none'
        with RunLock(args.output/'.job_guard'):
            if args.command=='train':train(args.data_root,args.output,args.task,args.model,scheme,args.seed,training,args.device,not args.no_resume,args.skip_existing)
            if args.eval_after or args.command=='classical':
                for test_scheme in ['A','C'] if args.task in NOISE_TASKS else ['none']:
                    evaluate(args.data_root,args.output,'test',test_scheme,args.device,training,
                             training.get('max_test_samples'),args.skip_existing,
                             args.model if args.command=='classical' else None,args.task)
        return 0
    if args.command=='evaluate':
        from .engine import evaluate
        info=read_json(args.run/'run_config.json');task=info['identity']['task']
        _,execution=config_with_args(args)
        # Checkpoint controls optimization/architecture. Only worker placement is overridden here.
        overrides={k:execution[k] for k in ('num_workers','prefetch_factor','cpu_threads') if k in execution}
        for scheme in args.schemes if task in NOISE_TASKS else ['none']:
            evaluate(args.data_root,args.run,args.split,scheme,args.device,overrides,args.max_samples,args.skip_existing,output_tag=args.tag)
        return 0
    if args.command=='aggregate':
        from .reporting import aggregate_runs
        print(aggregate_runs(args.root,args.output));return 0
    if args.command=='plot':
        from .reporting import plot_run,plot_overview
        if args.run:print(plot_run(args.run,args.output))
        elif args.root:print(plot_overview(args.root,args.output))
        else:parser.error('plot requires --run or --root')
        return 0
    if args.command=='compare':
        from .compare import compare_cases
        print(compare_cases(args.left,args.right,args.output,metric=args.metric,cluster=args.cluster,bootstrap=args.bootstrap,seed=args.seed,cluster_map=args.cluster_map,direction=args.direction));return 0


def entry():
    try:return main()
    except RuntimeError as e:
        if 'Run is active in another process' in str(e):print(str(e),file=sys.stderr);return 3
        raise
