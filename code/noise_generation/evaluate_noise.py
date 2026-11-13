"""Reproducible, streaming audit of generated noise diversity and training similarity.

Run with the scientific Python environment (torch is not required)::
    python evaluate_noise.py --generated runs/noise_dit_ddp_all_lengths_lr5e-5/generated/all_lengths.npz

This is a descriptor-space adaptation of the published metrics, not an evaluation
in a learned one-class embedding. References default to the configured training
files, or can be selected explicitly as comparison data. Reference identity and
independence must be established separately before interpreting memorization.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')

import numpy as np
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits

from noise_eval_features import NOISE_TYPE_NAMES, extract_features, get_windows_for_length
from noise_eval_metrics import distribution_metrics, nearest_reference


def log(message):
    print(time.strftime('%H:%M:%S') + ' ' + message, flush=True)


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=json_default), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        Path(path).write_text('', encoding='utf-8')
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def normalize_training(x, mode):
    x = np.nan_to_num(np.asarray(x, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if mode == 'rms':
        x = x - float(np.mean(x))
        scale = float(np.sqrt(np.mean(x * x)))
    elif mode == 'maxabs':
        scale = float(np.max(np.abs(x)))
    elif mode == 'none':
        return x
    else:
        raise ValueError('Unsupported training normalization: ' + mode)
    return x / scale if scale > 1e-8 else x


def unit_shape(x):
    x = np.asarray(x, dtype=np.float32)
    centered = x - x.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(centered * centered, axis=1, keepdims=True))
    return centered / np.maximum(rms, 1e-8)


def digest(x):
    return hashlib.sha256(np.ascontiguousarray(x, dtype='<f4').tobytes()).digest()


def effective_digest(x):
    """Ignore only exact boundary padding zeros, retaining every interior value.

    Max-energy windows can tie when a copied training clip starts/ends in zeros.
    Trimming these zeros makes the byte check independent of the chosen tie.
    Declared length is still a separate key, so different conditions do not mix.
    """
    x = np.asarray(x).reshape(-1)
    support = np.flatnonzero(x != 0)
    return digest(x[support[0]:support[-1]+1] if len(support) else x[:0])


def scan_training(root, synthetic_root, target, config, reference_dirs=None, reference_lengths=None):
    import re
    rows = []
    selected = reference_lengths if reference_dirs else config.get('train_lengths')
    sources = []
    if reference_dirs:
        for source in ('real', 'synthetic'):
            if source in reference_dirs:
                folder = Path(reference_dirs[source])
                if not folder.is_dir():
                    raise ValueError(f'Reference directory not found: {folder}')
                sources.append((source, [folder]))
    else:
        if config.get('include_real', True):
            sources.append(('real', sorted(root.iterdir())))
        if config.get('include_synthetic', True):
            sources.append(('synthetic', sorted(synthetic_root.iterdir())))
    for source, folders in sources:
        pattern = re.compile(r'^Real-Len(\d+)$' if source == 'real' else r'^Synthetic-Len(\d+)$')
        for folder in folders:
            match = pattern.match(folder.name)
            if source == 'real' and folder.name == 'Real-2048':
                directory_length = 2048
            elif match:
                directory_length = int(match.group(1))
            else:
                if reference_dirs:
                    raise ValueError(f'Cannot infer reference length from directory: {folder}')
                continue
            if directory_length > target:
                continue
            for path in sorted(folder.glob('*/*.npy')):
                if path.parent.name not in NOISE_TYPE_NAMES:
                    continue
                match_len = re.search(r'_len(\d+)', path.name)
                length = int(match_len.group(1)) if match_len else directory_length
                if length > target or (selected is not None and length not in selected):
                    continue
                rows.append(dict(index=len(rows), source=source,
                                 type_id=NOISE_TYPE_NAMES.index(path.parent.name),
                                 type_name=path.parent.name, length=length, path=str(path.resolve())))
    if not rows:
        raise ValueError('No training references found.')
    return rows


def cache_generated(args, out, type_id, lengths):
    cache = out / 'cache'
    cache.mkdir(exist_ok=True)
    manifest_path = cache / 'generated_manifest.json'
    stat = args.generated.stat()
    identity = dict(path=str(args.generated.resolve()), size=stat.st_size, mtime_ns=stat.st_mtime_ns,
                    feature_version=1)
    if args.reuse_cache and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        if manifest['identity'] == identity:
            log('Reusing complete generated cache.')
            return (np.load(cache/'generated_raw.npy', mmap_mode='r'),
                    np.load(cache/'generated_features.npy', mmap_mode='r'),
                    np.load(cache/'generated_signatures.npy', mmap_mode='r'),
                    dict(np.load(cache/'generated_diagnostics.npz')), manifest)
    with zipfile.ZipFile(args.generated) as archive:
        with archive.open('samples.npy') as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            if fortran or len(shape) != 3 or shape[1] != 1 or dtype != np.dtype('float32'):
                raise ValueError(f'Expected C-order float32 (N,1,L), got {shape}, {dtype}, fortran={fortran}')
            n, _, width = shape
            if n != len(type_id) or len(lengths) != n:
                raise ValueError('Generated metadata and samples have different row counts.')
            raw = np.lib.format.open_memmap(cache/'generated_raw.npy', mode='w+', dtype=np.float32, shape=(n,width))
            feature = signature = None
            diagnostics = {}
            seen = set()
            repeated = 0
            sample_hash = hashlib.sha256()
            for start in range(0, n, args.chunk_size):
                stop = min(n, start+args.chunk_size)
                block = stream.read((stop-start)*width*dtype.itemsize)
                batch = np.frombuffer(block, dtype=dtype).reshape(stop-start, width)
                raw[start:stop] = batch
                sample_hash.update(block)
                for row in batch:
                    key = digest(row)
                    if key in seen:
                        repeated += 1
                    seen.add(key)
                result = extract_features(batch, lengths[start:stop])
                if feature is None:
                    names = result['feature_names']
                    feature = np.lib.format.open_memmap(cache/'generated_features.npy', mode='w+', dtype=np.float32, shape=(n,len(names)))
                    signature = np.lib.format.open_memmap(cache/'generated_signatures.npy', mode='w+', dtype=np.float32, shape=(n,128))
                    diagnostics = {name: np.empty(n, dtype=np.asarray(values).dtype) for name,values in result['diagnostics'].items()}
                feature[start:stop] = result['features']
                signature[start:stop] = result['signature']
                for name, values in result['diagnostics'].items():
                    diagnostics[name][start:stop] = values
                if start == 0 or stop == n or start // 25000 != stop // 25000:
                    log(f'Generated feature extraction: {stop:,}/{n:,}')
            # Reading through EOF also completes the ZIP CRC integrity check.
            if stream.read(1):
                raise ValueError('Unexpected bytes after generated array.')
    raw.flush(); feature.flush(); signature.flush()
    np.savez_compressed(cache/'generated_diagnostics.npz', **diagnostics)
    manifest = dict(identity=identity, shape=shape, feature_names=names,
                    raw_data_sha256=sample_hash.hexdigest(), exact_duplicate_excess=repeated,
                    exact_unique_rows=len(seen))
    write_json(manifest_path, manifest)
    return raw, feature, signature, diagnostics, manifest


def load_training(args, out, target, config):
    cache = out / 'cache'
    rows = scan_training(args.data_root, args.synthetic_root, target, config,
                         args.reference_dirs, args.reference_lengths)
    raw = np.lib.format.open_memmap(cache/'training_raw.npy', mode='w+', dtype=np.float32, shape=(len(rows), target))
    features = signatures = None
    exact = defaultdict(dict)
    canonical_seen = set()
    duplicate_excess = 0
    for start in range(0, len(rows), args.chunk_size):
        block_rows = rows[start:start+args.chunk_size]
        batch = np.zeros((len(block_rows), target), np.float32)
        lengths = np.array([row['length'] for row in block_rows])
        for j, row in enumerate(block_rows):
            original = np.asarray(np.load(row['path'], allow_pickle=False), dtype=np.float32).reshape(-1)
            if len(original) != row['length']:
                raise ValueError(f"Training length mismatch: {row['path']}: {len(original)} != {row['length']}")
            row['content_sha256'] = digest(original).hex()
            row['finite_fraction'] = float(np.isfinite(original).mean())
            x = normalize_training(original, config.get('data_normalize', 'rms'))
            offset = (target-len(x))//2
            batch[j,offset:offset+len(x)] = x
            keys = [effective_digest(v) for v in (x,-x,x[::-1],-x[::-1])]
            canonical_key = (len(x),min(keys))
            if canonical_key in canonical_seen:
                duplicate_excess += 1
            canonical_seen.add(canonical_key)
            for key in keys:
                exact[len(x)].setdefault(key, {}).setdefault(row['source'],row['index'])
        result = extract_features(batch, lengths)
        if features is None:
            features = np.empty((len(rows),result['features'].shape[1]),np.float32)
            signatures = np.empty((len(rows),128),np.float32)
        raw[start:start+len(block_rows)] = batch
        features[start:start+len(block_rows)] = result['features']
        signatures[start:start+len(block_rows)] = result['signature']
        if start == 0 or start // 5000 != (start+len(block_rows))//5000:
            log(f'Reference signals: {start+len(block_rows):,}/{len(rows):,}')
    raw.flush()
    np.save(cache/'training_features.npy',features)
    np.save(cache/'training_signatures.npy',signatures)
    write_csv(out/'training_manifest.csv',rows)
    return rows,raw,features,signatures,exact,duplicate_excess


def closest_waveforms(generated_windows, generated_signature, reference_windows, reference_signature, candidates=5):
    """Approximate candidate search; complete-L-point recheck of each candidate.

    Signature search allows polarity and reversal. This is a screen, not a proof
    that the globally closest full-resolution waveform has been found.
    """
    gs = unit_shape(generated_signature)
    rs = unit_shape(reference_signature)
    score = np.maximum(np.abs(gs @ rs.T), np.abs(gs @ rs[:,::-1].T))/gs.shape[1]
    k = min(candidates, len(rs))
    indices = np.argpartition(-score, kth=k-1, axis=1)[:,:k]
    g = unit_shape(generated_windows)
    r = unit_shape(reference_windows)
    best_rmse = np.full(len(g),np.inf,np.float32)
    best_index = np.zeros(len(g),np.int64)
    best_corr = np.zeros(len(g),np.float32)
    for column in range(k):
        chosen = r[indices[:,column]]
        c1 = np.mean(g*chosen,axis=1)
        c2 = np.mean(g*chosen[:,::-1],axis=1)
        reverse = np.abs(c2) > np.abs(c1)
        corr = np.where(reverse,c2,c1)
        aligned = np.where(reverse[:,None],chosen[:,::-1],chosen)
        aligned = aligned * np.where(corr < 0,-1.0,1.0)[:,None]
        rmse = np.sqrt(np.mean((g-aligned)**2,axis=1))
        update = rmse < best_rmse
        best_index[update] = indices[update,column]
        best_rmse[update] = rmse[update]
        best_corr[update] = np.abs(corr[update])
    return best_index,best_rmse,best_corr


def evaluate(args):
    out = args.output_dir
    out.mkdir(parents=True,exist_ok=True)
    run_dir = args.generated.parent.parent
    config_path = args.config or run_dir/'config.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    with np.load(args.generated,allow_pickle=False) as archive:
        type_ids = np.asarray(archive['type_id'],np.int64)
        lengths = np.asarray(archive['length'],np.int64)
        source_ids = np.asarray(archive['source_id'],np.int64) if 'source_id' in archive.files else None
    if np.any((type_ids < 0)|(type_ids >= len(NOISE_TYPE_NAMES))) or np.any((lengths <= 0)|(lengths > config['target_length'])):
        raise ValueError('Invalid generated class or length labels.')
    counts = Counter(zip(type_ids.tolist(), lengths.tolist()))
    args.reference_dirs = {source: path for source, path in
                           (('real', args.real_reference_dir), ('synthetic', args.synthetic_reference_dir))
                           if path is not None}
    args.reference_lengths = sorted(set(lengths.tolist()))
    write_csv(out/'generated_counts.csv',[dict(type_id=t,type_name=NOISE_TYPE_NAMES[t],length=l,count=n) for (t,l),n in sorted(counts.items())])
    graw,gfeatures,gsignatures,diagnostics,manifest = cache_generated(args,out,type_ids,lengths)
    trows,traw,tfeatures,tsignatures,exact,training_duplicates = load_training(args,out,config['target_length'],config)
    source_names = np.array([row['source'] for row in trows])
    train_types = np.array([row['type_id'] for row in trows])
    train_lengths = np.array([row['length'] for row in trows])
    actual_info = dict(n_samples=len(trows),source_distribution=dict(Counter(source_names)),
                       type_distribution={name:int(np.sum(train_types==i)) for i,name in enumerate(NOISE_TYPE_NAMES)},
                       length_distribution={str(k):v for k,v in sorted(Counter(train_lengths.tolist()).items())})
    recorded_path = run_dir/'dataset_info.json'
    recorded_info = json.loads(recorded_path.read_text(encoding='utf-8')) if recorded_path.exists() else {}
    distribution_matches = all(actual_info[key] == recorded_info.get(key) for key in actual_info) if recorded_info else None
    scope_warning = None
    if args.reference_dirs:
        scope_warning = ('Evaluation uses explicitly selected reference directories. Their identity as the '
                         'original training files has not been verified; authenticity and similarity '
                         'results describe proximity to these references, not a complete training-copy audit.')
        if distribution_matches is False:
            scope_warning += ' Reference counts/distributions differ from the saved training record.'
        log(scope_warning)
    n = len(type_ids)
    audit = {}
    for source in ('real','synthetic','pooled'):
        for key in ('feature_distance','authentic','shape_rmse','shape_corr'):
            audit[source+'_'+key] = np.full(n,np.nan,np.float32)
        for key in ('feature_neighbor','shape_neighbor','exact_neighbor'):
            audit[source+'_'+key] = np.full(n,-1,np.int32)
    group_rows, curves, candidates, sampling_manifest, calibration = [],[],[],[],[]
    visualization = defaultdict(list)
    rng = np.random.default_rng(args.seed)
    group_keys = sorted(counts,key=lambda key:(key[1],key[0]))
    for group_number,(type_id,length) in enumerate(group_keys,1):
        gi = np.flatnonzero((type_ids==type_id)&(lengths==length))
        ti_all = np.flatnonzero((train_types==type_id)&(train_lengths==length))
        if not len(ti_all):
            continue
        mean = tfeatures[ti_all].mean(axis=0,dtype=np.float64)
        std = tfeatures[ti_all].std(axis=0,dtype=np.float64)
        scale = np.where(std > 1e-6,std,1.0)
        reference_features = (tfeatures[ti_all]-mean)/scale
        generated_features = (np.asarray(gfeatures[gi])-mean)/scale
        gwindows,_,_ = get_windows_for_length(np.asarray(graw[gi]),length)
        r_all,_,_ = get_windows_for_length(np.asarray(traw[ti_all]),length)
        # Exact checks are across all reference classes at this declared length.
        for local_index,window in enumerate(gwindows):
            matches = exact[length].get(effective_digest(window),{})
            for source,index in matches.items():
                audit[source+'_exact_neighbor'][gi[local_index]] = index
            if matches:
                audit['pooled_exact_neighbor'][gi[local_index]] = next(iter(matches.values()))
        wave_matches = {}
        for source in ('real','synthetic'):
            local = np.flatnonzero(source_names[ti_all]==source)
            if len(local):
                near,rmse,corr = closest_waveforms(gwindows,np.asarray(gsignatures[gi]),r_all[local],tsignatures[ti_all[local]],args.waveform_candidates)
                wave_matches[source] = (ti_all[local[near]],rmse,corr)
        if len(wave_matches)==2:
            a,b = wave_matches['real'],wave_matches['synthetic']
            choose_a = a[1] <= b[1]
            wave_matches['pooled'] = tuple(np.where(choose_a,x,y) for x,y in zip(a,b))
        elif wave_matches:
            wave_matches['pooled'] = next(iter(wave_matches.values()))
        for source in ('real','synthetic','pooled'):
            local = np.arange(len(ti_all)) if source=='pooled' else np.flatnonzero(source_names[ti_all]==source)
            ti = ti_all[local]
            row = dict(source=source,type_id=type_id,type_name=NOISE_TYPE_NAMES[type_id],length=length,
                       n_reference=len(ti),n_generated=len(gi),n_balanced=0,
                       coverage=np.nan,integrated_beta_recall=np.nan,authenticity_full=np.nan,
                       near_copy_rate=np.nan,near_copy_rate_001=np.nan,near_copy_rate_010=np.nan,
                       exact_matches=int(np.sum(audit[source+'_exact_neighbor'][gi]>=0)),
                       small_sample=bool(len(ti)<20),status='missing_reference')
            if not len(ti):
                group_rows.append(row)
                continue
            rf = reference_features[local]
            nearest = nearest_reference(rf,generated_features)
            audit[source+'_feature_neighbor'][gi] = ti[nearest['nearest_index']]
            audit[source+'_feature_distance'][gi] = nearest['nearest_distance']
            if len(ti)>=2:
                auth = nearest['authentic'].astype(np.float32)
                audit[source+'_authentic'][gi] = auth
                row['authenticity_full'] = float(auth.mean())
                row['reference_nn_zero_fraction'] = float(np.mean(nearest['reference_nn_distance']==0))
            wave_index,rmse,corr = wave_matches[source]
            audit[source+'_shape_neighbor'][gi] = wave_index
            audit[source+'_shape_rmse'][gi] = rmse
            audit[source+'_shape_corr'][gi] = corr
            row['near_copy_rate'] = float(np.mean(rmse<=args.near_copy_threshold))
            row['near_copy_rate_001'] = float(np.mean(rmse<=0.01))
            row['near_copy_rate_010'] = float(np.mean(rmse<=0.10))
            row['median_shape_rmse'] = float(np.median(rmse))
            row['status'] = ('insufficient_reference' if len(ti)<2 else
                             'insufficient_generated' if len(gi)<2 else 'ok')
            if len(ti)>=2 and len(gi)>=2:
                m = min(args.balanced_limit,len(ti),len(gi))
                ri = rng.choice(len(ti),m,replace=False)
                gj = rng.choice(len(gi),m,replace=False)
                metric = distribution_metrics(rf[ri],generated_features[gj],k=args.k)
                row.update(n_balanced=m,coverage=metric['coverage'],
                           integrated_beta_recall=metric['integrated_beta_recall'],
                           k=metric['k'],zero_reference_nn_fraction=metric.get('zero_reference_nn_fraction',np.nan))
                curves.extend(dict(source=source,type_id=type_id,type_name=NOISE_TYPE_NAMES[type_id],length=length,beta=float(beta),recall=float(recall)) for beta,recall in zip(metric['beta_grid'],metric['beta_recall']))
                sampling_manifest.append(dict(source=source,type_id=type_id,length=length,
                                               training_indices=ti[ri].tolist(),generated_indices=gi[gj].tolist()))
                # Equal-size reference-vs-reference calibration; this is a
                # post-training split, explicitly NOT a model holdout set.
                if len(ti)>=4:
                    cm = min(args.balanced_limit,len(ti)//2,len(gi))
                    split = rng.choice(len(ti),2*cm,replace=False)
                    cg = rng.choice(len(gi),cm,replace=False)
                    reference_control = distribution_metrics(rf[split[:cm]],rf[split[cm:]],k=args.k)
                    generated_control = distribution_metrics(rf[split[:cm]],generated_features[cg],k=args.k)
                    calibration.append(dict(source=source,type_id=type_id,type_name=NOISE_TYPE_NAMES[type_id],length=length,n_per_set=cm,
                                            reference_coverage=reference_control['coverage'],generated_coverage=generated_control['coverage'],
                                            reference_irbeta=reference_control['integrated_beta_recall'],generated_irbeta=generated_control['integrated_beta_recall'],
                                            coverage_gap=generated_control['coverage']-reference_control['coverage']))
                    sampling_manifest.append(dict(purpose='reference_split_calibration',source=source,type_id=type_id,length=length,
                                                   reference_a=ti[split[:cm]].tolist(),reference_b=ti[split[cm:]].tolist(),generated_indices=gi[cg].tolist()))
            for j in np.argsort(rmse)[:args.candidates_per_group]:
                training_index = int(wave_index[j])
                feature_distance = np.linalg.norm((tfeatures[training_index]-mean)/scale-generated_features[j])
                candidates.append(dict(source=source,type_id=type_id,type_name=NOISE_TYPE_NAMES[type_id],length=length,
                                       generated_index=int(gi[j]),training_index=training_index,training_path=trows[training_index]['path'],
                                       rmse=float(rmse[j]),corr=float(corr[j]),feature_distance=float(feature_distance),
                                       authentic=float(audit[source+'_authentic'][gi[j]])))
            group_rows.append(row)
        # Visualization is stratified; all lengths are represented.
        visual_limit = 100 if length==2048 else 20
        for source,indices,feats in ((2,gi,gfeatures),(0,ti_all[source_names[ti_all]=='real'],tfeatures),(1,ti_all[source_names[ti_all]=='synthetic'],tfeatures)):
            if len(indices):
                chosen = rng.choice(indices,min(visual_limit,len(indices)),replace=False)
                visualization['features'].append(np.asarray(feats[chosen]))
                visualization['source'].append(np.full(len(chosen),source))
                visualization['type_id'].append(np.full(len(chosen),type_id))
                visualization['length'].append(np.full(len(chosen),length))
        if group_number%5==0 or group_number==1:
            write_csv(out/'group_metrics.csv',group_rows)
            log(f'Group metrics / full waveform checks: {group_number}/{len(group_keys)} ({NOISE_TYPE_NAMES[type_id]}, L={length})')
    write_csv(out/'group_metrics.csv',group_rows)
    write_csv(out/'beta_curves.csv',curves)
    candidates.sort(key=lambda row:row['rmse'])
    write_csv(out/'copy_candidates.csv',candidates)
    write_csv(out/'reference_calibration.csv',calibration)
    write_json(out/'balanced_sample_indices.json',sampling_manifest)
    np.savez_compressed(out/'per_sample_audit.npz',generated_index=np.arange(n),type_id=type_ids,length=lengths,**audit)
    np.savez_compressed(out/'visualization.npz',**{key:np.concatenate(values) for key,values in visualization.items()},feature_names=np.array(manifest['feature_names']))
    example_rows = []
    used_types = set()
    for row in candidates:
        if row['source']=='pooled' and row['type_id'] not in used_types:
            example_rows.append(row); used_types.add(row['type_id'])
    for row in candidates:
        if len(example_rows)>=12:
            break
        if row['source']=='pooled' and row not in example_rows:
            example_rows.append(row)
    examples_g = np.zeros((len(example_rows),config['target_length']),np.float32)
    examples_r = np.zeros_like(examples_g)
    for j,row in enumerate(example_rows):
        length = row['length']
        g,_,_ = get_windows_for_length(np.asarray(graw[row['generated_index']:row['generated_index']+1]),length)
        r,_,_ = get_windows_for_length(np.asarray(traw[row['training_index']:row['training_index']+1]),length)
        g,r = unit_shape(g)[0],unit_shape(r)[0]
        variants = np.stack([r,-r,r[::-1],-r[::-1]])
        r = variants[np.argmin(np.mean((variants-g)**2,axis=1))]
        examples_g[j,:length] = g; examples_r[j,:length] = r
    np.savez_compressed(out/'waveform_examples.npz',generated=examples_g,training=examples_r,
                        **{key:np.array([row[key] for row in example_rows]) for key in ('length','type_name','source','generated_index','training_path','rmse')})
    aggregates = []
    for source in ('real','synthetic','pooled'):
        rows = [row for row in group_rows if row['source']==source and row['status']=='ok']
        values = audit[source+'_authentic']
        shape = audit[source+'_shape_rmse']
        valid = np.isfinite(values)
        aggregates.append(dict(source=source,groups=len(rows),total_generated_evaluated=int(valid.sum()),
                               coverage_macro=float(np.mean([row['coverage'] for row in rows])),
                               integrated_beta_recall_macro=float(np.mean([row['integrated_beta_recall'] for row in rows])),
                               authenticity_full=float(np.nanmean(values)),
                               near_copy_rate=float(np.mean(shape[np.isfinite(shape)]<=args.near_copy_threshold)),
                               near_copy_evaluated=int(np.isfinite(shape).sum()),
                               exact_matches=int(np.sum(audit[source+'_exact_neighbor']>=0))))
        calibrated = [row for row in calibration if row['source']==source]
        if calibrated:
            aggregates[-1].update(calibration_groups=len(calibrated),
                                  reference_split_coverage_macro=float(np.mean([row['reference_coverage'] for row in calibrated])),
                                  matched_generated_coverage_macro=float(np.mean([row['generated_coverage'] for row in calibrated])),
                                  reference_split_irbeta_macro=float(np.mean([row['reference_irbeta'] for row in calibrated])),
                                  matched_generated_irbeta_macro=float(np.mean([row['generated_irbeta'] for row in calibrated])))
    diagnostics_summary = {}
    for name,values in diagnostics.items():
        values = np.asarray(values)
        diagnostics_summary[name] = dict(mean=float(np.mean(values)),p05=float(np.quantile(values,.05)),median=float(np.median(values)),p95=float(np.quantile(values,.95)))
    diagnostic_rows = []
    for (type_id,length),count in sorted(counts.items()):
        selected = (type_ids==type_id)&(lengths==length)
        row = dict(type_id=type_id,type_name=NOISE_TYPE_NAMES[type_id],length=length,n=count)
        for name in ('window_energy_fraction','window_rms','full_rms','finite_fraction'):
            if name in diagnostics:
                row[name+'_median'] = float(np.median(diagnostics[name][selected]))
        diagnostic_rows.append(row)
    write_csv(out/'window_diagnostics.csv',diagnostic_rows)
    source_comparison = '; '.join(
        f"{source}: {sum(row['source']==source and row['n_reference']==0 for row in group_rows)} missing groups, "
        f"{sum(row['source']==source and row['n_reference']==1 for row in group_rows)} singleton groups"
        for source in ('real','synthetic'))
    source_comparison += '; metrics invalid with fewer than 2 references; n<20 marked small_sample'
    summary = dict(provenance=dict(generated_file=str(args.generated.resolve()),generated_shape=manifest['shape'],
                                   generated_count=n,generated_combinations=len(counts),counts_per_combination=sorted(set(counts.values())),
                                   raw_data_sha256=manifest['raw_data_sha256'],training_config_path=str(config_path),
                                   training_distribution_matches_record=distribution_matches,training=actual_info,
                                   reference=actual_info, recorded_training=recorded_info,
                                   reference_selection='explicit_directories' if args.reference_dirs else 'training_config',
                                   reference_directories={source:str(path.resolve()) for source,path in args.reference_dirs.items()},
                                   reference_scope_warning=scope_warning,
                                   recorded_include_real=config.get('include_real',True),
                                   recorded_include_synthetic=config.get('include_synthetic',True),
                                   recorded_synthetic_root=config.get('synthetic_root'),
                                   generated_source_ids=sorted(set(source_ids.tolist())) if source_ids is not None else None,
                                   condition_on_source=config.get('condition_on_source',False),
                                   training_file_identity_not_provable_from_counts=True),
                   methodology=dict(feature_space='handcrafted time/frequency descriptors; reference-only scaling per type/length',
                                    feature_count=len(manifest['feature_names']),effective_window='maximum energy contiguous window of declared conditional length',
                                    diversity='equal-count reference/generated subsamples, beta-subset nearest neighbors (Alaa Eq.9), PRDC coverage',
                                    balanced_limit=args.balanced_limit,k=args.k,seed=args.seed,
                                    calibration='disjoint halves of the selected reference with matched generated count; this is not a verified independent model holdout',
                                    authenticity='all generated rows, full available same-type/same-length selected reference; strict nearest-neighbor inequality',
                                    waveform_screen=f'top {args.waveform_candidates} resampled-signature candidates per source, full-length normalized RMSE with sign/reversal; pooled merges the two sources',
                                    near_copy_threshold=args.near_copy_threshold,exact_match='float32 byte equality of effective nonzero-boundary-trimmed sequences against normalized reference clips; internal zeros retained, sign/reversal allowed, all classes at declared length',
                                    holdout='none verified; reference-proximity audit, not out-of-sample generalization validation',
                                    source_comparison=source_comparison,
                                    official_embedding='descriptor-space adaptation, not reproduction of the paper one-class neural embedding'),
                   aggregate=aggregates,diagnostics=diagnostics_summary,
                   generated_exact_duplicate_excess=manifest['exact_duplicate_excess'],generated_exact_unique_rows=manifest['exact_unique_rows'],
                   training_augmented_exact_duplicate_excess=training_duplicates,
                   limitations=([scope_warning] if scope_warning else []) + ['High authenticity alone does not establish quality or prove absence of memorization.',
                                'Near-identical simple periodic shapes can occur without training-data copying.',
                                'Short-window extraction can omit generated background artifacts; see window diagnostics.',
                                'Waveform nearest-neighbor candidates are approximate; exact byte matching is a separate complete scan at each declared length.',
                                ('Reference identity/completeness is unverified; no downstream-task generalization test was performed.'
                                 if args.reference_dirs else 'No independent held-out reference set was available.')],
                   sources=['https://proceedings.mlr.press/v162/alaa22a.html','https://proceedings.mlr.press/v119/naeem20a.html'])
    write_json(out/'summary.json',summary)
    from noise_eval_report import build_report
    log('Building figures and HTML report.')
    build_report(out)
    log('Evaluation complete: '+str(out/'report.html'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generated',type=Path,default=Path('runs/noise_dit_ddp_all_lengths_lr5e-5/generated/all_lengths.npz'))
    parser.add_argument('--data_root',type=Path,default=Path('.'))
    parser.add_argument('--synthetic_root',type=Path,default=Path('Synthetic-Norm'))
    parser.add_argument('--real_reference_dir',type=Path,default=None,
                        help='Explicit length-specific real reference folder (e.g. Real-2048). If either reference folder is set, only explicit folders are evaluated.')
    parser.add_argument('--synthetic_reference_dir',type=Path,default=None,
                        help='Explicit length-specific synthetic reference folder; overrides config inclusion flags for evaluation only.')
    parser.add_argument('--config',type=Path,default=None)
    parser.add_argument('--output_dir',type=Path,default=None)
    parser.add_argument('--chunk_size',type=int,default=1024)
    parser.add_argument('--balanced_limit',type=int,default=512)
    parser.add_argument('--k',type=int,default=5)
    parser.add_argument('--waveform_candidates',type=int,default=5)
    parser.add_argument('--near_copy_threshold',type=float,default=.05)
    parser.add_argument('--candidates_per_group',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--reuse_cache',action='store_true')
    args = parser.parse_args()
    args.output_dir = args.output_dir or args.generated.parent/'evaluation'
    if min(args.chunk_size,args.balanced_limit,args.k,args.waveform_candidates,args.threads)<1:
        parser.error('Count and thread parameters must be positive.')
    if args.balanced_limit<2:
        parser.error('balanced_limit must be at least 2 for reference neighborhoods.')
    try:
        limit_context = threadpool_limits(limits=args.threads)
    except AttributeError as exc:
        log(f'Threadpool library detection unavailable ({exc}); using process thread environment limits.')
        limit_context = nullcontext()
    with limit_context:
        evaluate(args)


if __name__=='__main__':
    main()
