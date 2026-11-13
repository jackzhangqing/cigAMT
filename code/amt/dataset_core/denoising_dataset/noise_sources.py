"""Self-contained, audited single-channel noise bank and leakage-safe splits.

``prepare_noise(bundle_root, config, audit_dir)`` packs original noise and the
complete diffusion outputs. It never loads Python objects from NPZ archives.
Generated conditional length is an explicit crop policy, not a measured support
label. All generated noise remains train-only because its training inventory
does not prove exclusion of original held-out noise records.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path
from typing import Any
import zipfile

import h5py
import numpy as np

CLASS_NAMES = ['clean', 'pulse', 'quasi_sinusoidal', 'square_wave', 'step_wave',
               'triangular_wave', 'power_frequency', 'white_noise', 'color_noise',
               'harmonic', 'composite']
DOMAIN_NAMES = ['real', 'synthetic', 'generated']
SOURCE_KIND_NAMES = ['field_original', 'synthetic_norm', 'generated_all_lengths',
                     'generated_2048']
SPLIT_NAMES = ['train', 'validation', 'test']
SEQUENCE_LENGTH = 2048


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8-sig'))


def _save_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode('utf-8')).hexdigest()


def _npy_header(stream):
    version = np.lib.format.read_magic(stream)
    return np.lib.format._read_array_header(stream, version)


def _fingerprint(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {'size_bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}


def _stable_rms(value: np.ndarray) -> float:
    value = np.asarray(value, dtype=np.float64)
    if not np.isfinite(value).all() or value.size < 2:
        return float('nan')
    peak = float(np.max(np.abs(value)))
    return 0.0 if peak == 0 else peak * float(np.sqrt(np.mean((value / peak) ** 2)))


def _read_original(row: dict[str, Any]) -> dict[str, Any]:
    path = Path(row['path'])
    expected = {key: row[key] for key in ('size_bytes', 'mtime_ns')}
    if _fingerprint(path) != expected:
        raise ValueError(f'Noise changed after energy audit: {path}')
    raw_bytes = path.read_bytes()
    value = np.load(io.BytesIO(raw_bytes), allow_pickle=False).squeeze()
    if value.ndim != 1 or len(value) != int(row['length']):
        raise ValueError(f'Invalid original noise shape: {path}, {value.shape}')
    array_hash = hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    if row.get('array_sha256') and array_hash != row['array_sha256']:
        raise ValueError(f'Noise content differs from the completed energy audit: {path}')
    result = {'samples': value.astype(np.float32), 'file_sha256': hashlib.sha256(raw_bytes).hexdigest(),
              'array_sha256': array_hash}
    partner = row.get('raw_corresponding_path')
    if partner:
        if row.get('raw_corresponding_fingerprint') and _fingerprint(Path(partner)) != row['raw_corresponding_fingerprint']:
            raise ValueError(f'Corresponding Raw representation changed after audit: {partner}')
        partner_bytes = Path(partner).read_bytes()
        partner_value = np.load(io.BytesIO(partner_bytes), allow_pickle=False).squeeze()
        if partner_value.ndim != 1 or len(partner_value) != len(value) or not np.isfinite(partner_value).all():
            raise ValueError(f'Invalid corresponding Raw representation: {partner}')
        partner_array_hash = hashlib.sha256(np.ascontiguousarray(partner_value).tobytes()).hexdigest()
        if row.get('raw_corresponding_array_sha256') and partner_array_hash != row['raw_corresponding_array_sha256']:
            raise ValueError(f'Corresponding Raw content changed after audit: {partner}')
        result.update(raw_samples=partner_value.astype(np.float32),
                      raw_file_sha256=hashlib.sha256(partner_bytes).hexdigest(),
                      raw_array_sha256=partner_array_hash)
    return result


def _legacy_assignments(path: Path | None) -> tuple[dict[str, str], dict[str, Any]]:
    if path is None or not path.is_file():
        return {}, {'mode': 'new_parent_group_split', 'legacy_manifest': None}
    manifest = _json(path)
    parents: dict[str, str] = {}
    for entry in manifest.get('noise', []):
        parent = entry['parent_id']
        split = entry['split']
        if split not in SPLIT_NAMES:
            raise ValueError(f'Invalid legacy noise split {split!r}')
        if parent in parents and parents[parent] != split:
            raise ValueError(f'Legacy manifest has one noise parent in multiple splits: {parent}')
        parents[parent] = split
    return parents, {'mode': 'inherit_legacy_parent_assignments_then_assign_new_parents',
                     'legacy_manifest': str(path), 'legacy_manifest_sha256': _sha256_file(path),
                     'legacy_manifest_id': manifest.get('manifest_id'),
                     'legacy_parent_count': len(parents)}


def _assign_original_splits(rows: list[dict[str, Any]], config: dict[str, Any]):
    noise_config = config.get('noise', {})
    seed = int(noise_config.get('seed', config.get('seed', 2025)))
    validation_fraction = float(noise_config.get('validation_fraction', .15))
    test_fraction = float(noise_config.get('test_fraction', .20))
    if not 0 < validation_fraction < 1 or not 0 < test_fraction < 1 or validation_fraction + test_fraction >= 1:
        raise ValueError('Noise validation/test fractions must be positive and sum to less than one')
    legacy = noise_config.get('legacy_manifest_path', config.get('paths', {}).get('legacy_noise_manifest'))
    if legacy is None:
        candidate = Path(r'legacy-source:8c429b4dc593e420e0eb/manifest_windows_linux_order.json')
        legacy = str(candidate) if candidate.is_file() else None
    assignment, audit = _legacy_assignments(Path(legacy) if legacy else None)
    # Real crops/channels/types/lengths are grouped by complete field record.
    # Synthetic global sample IDs stay together, regardless of Norm/Raw format.
    bucket_parents = defaultdict(set)
    parent_buckets = defaultdict(set)
    for row in rows:
        bucket = 'real_records' if row['domain'] == 'real' else f"synthetic:{row['noise_type']}:{row['length']}"
        bucket_parents[bucket].add(row['parent_id'])
        parent_buckets[row['parent_id']].add(bucket)
    inherited = len({r['parent_id'] for r in rows if r['parent_id'] in assignment})
    # A global parent occurs in more than one synthetic stratum only if the input
    # ID conventions changed. Hash assignment remains global in that situation.
    cross_bucket = {parent for parent, buckets in parent_buckets.items() if len(buckets) > 1}
    for bucket, parents in sorted(bucket_parents.items()):
        new = sorted(parent for parent in parents if parent not in assignment)
        bucket_seed = int.from_bytes(hashlib.sha256(f'{seed}:{bucket}'.encode()).digest()[:8], 'little')
        new = np.asarray(new, dtype=object)
        if len(new):
            new = new[np.random.default_rng(bucket_seed).permutation(len(new))]
        n_validation = int(round(len(new) * validation_fraction))
        n_test = int(round(len(new) * test_fraction))
        if len(new) >= 3:
            n_validation = max(1, n_validation)
            n_test = max(1, n_test)
        if n_validation + n_test >= len(new) and len(new):
            n_validation, n_test = (1, 1) if len(new) >= 3 else (0, 0)
        for index, parent in enumerate(new):
            if parent in cross_bucket:
                score = int.from_bytes(hashlib.sha256(f'{seed}:noise-parent:{parent}'.encode()).digest()[:8], 'big') / 2**64
                assignment[parent] = 'validation' if score < validation_fraction else 'test' if score < validation_fraction + test_fraction else 'train'
            else:
                assignment[parent] = 'validation' if index < n_validation else 'test' if index < n_validation + n_test else 'train'
    splits = np.asarray([SPLIT_NAMES.index(assignment[r['parent_id']]) for r in rows], dtype=np.uint8)
    used_parents = {r['parent_id'] for r in rows}
    audit.update(seed=seed, validation_fraction=validation_fraction, test_fraction=test_fraction,
                 inherited_current_parent_count=inherited, current_parent_count=len(used_parents),
                 new_current_parent_count=len(used_parents) - inherited,
                 cross_stratum_parent_count=len(cross_bucket), parent_overlap_between_splits=0)
    return splits, audit


def _counts(domain_id, type_id, lengths, split_id):
    c = Counter(zip(domain_id.tolist(), type_id.tolist(), lengths.tolist(), split_id.tolist()))
    return [{'domain': DOMAIN_NAMES[d], 'type_id': t, 'noise_type': CLASS_NAMES[t],
             'length': length, 'split': SPLIT_NAMES[split], 'count': count}
            for (d, t, length, split), count in sorted(c.items())]


def prepare_noise(root: str | Path, config: dict[str, Any], audit_dir: str | Path) -> dict[str, Any]:
    """Pack verified noise and return a portable manifest with a numeric pool.

    Main datasets: samples[N,2048] float32, support_length[N], crop_start[N],
    type_id[N] (clean=0 is reserved), domain_id[N] (real=0/synthetic=1/generated=2),
    split_id[N] (train=0/validation=1/test=2), source_kind[N], parent_group_id[N].
    Generated samples retain their complete 2048 outputs. Templates are read as
    samples[row,crop_start:crop_start+support_length] before event embedding.
    """
    root = Path(root).resolve()
    audit_dir = Path(audit_dir).resolve()
    energy_path = audit_dir / 'noise_energy_audit.json'
    inventory_path = audit_dir / 'noise_inventory.json'
    catalog_path = audit_dir / 'original_noise_catalog.json'
    report = _json(energy_path)
    inventory = _json(inventory_path)
    if not report.get('complete'):
        raise ValueError('A completed full noise energy audit is required before packing')
    if inventory.get('canonical_noise_types', CLASS_NAMES[1:]) != CLASS_NAMES[1:]:
        raise ValueError('Audited generator type order differs from the bank class mapping')
    noise_config = config.get('noise', {})
    min_rms = float(noise_config.get('minimum_usable_rms', 1e-15))
    if min_rms < 0 or not np.isfinite(min_rms):
        raise ValueError('minimum_usable_rms must be finite and nonnegative')
    all_original = _json(catalog_path)['records']
    originals = [r for r in all_original if r['valid'] and float(r['rms']) > min_rms]
    originals.sort(key=lambda r: str(r['path']).casefold())
    original_splits, split_audit = _assign_original_splits(originals, config)
    original_count = len(originals)
    generated = []
    for source in report['generated']:
        metrics_path = Path(source['energy_metrics_path'])
        if not metrics_path.is_file():
            metrics_path = audit_dir / metrics_path.name
        with np.load(metrics_path, allow_pickle=False) as metrics:
            data = {k: metrics[k].copy() for k in metrics.files}
        if len(data['flags']) != int(source['count']):
            raise ValueError(f'Generated noise audit shape mismatch: {source["path"]}')
        selected = (data['flags'] == 0) & (data['rms'] > min_rms) & (data['crop_rms'] > min_rms)
        if np.any(data['length'][selected] > SEQUENCE_LENGTH) or np.any(data['crop_start'][selected] < 0):
            raise ValueError('Invalid generated conditional crop metadata')
        if np.any(data['crop_start'][selected] + data['length'][selected] > SEQUENCE_LENGTH):
            raise ValueError('Generated crop exceeds retained sample length')
        data['selected_rows'] = np.flatnonzero(selected)
        data['audit_source'] = source
        data['metrics_file_sha256'] = _sha256_file(metrics_path)
        generated.append(data)
    generated_count = sum(len(g['selected_rows']) for g in generated)
    count = original_count + generated_count
    if count == 0:
        raise ValueError('No eligible noise remains after energy filtering')
    row_id = np.arange(count, dtype=np.int64)
    domain_id = np.zeros(count, dtype=np.uint8)
    type_id = np.zeros(count, dtype=np.int16)
    lengths = np.zeros(count, dtype=np.int32)
    crop_start = np.zeros(count, dtype=np.int32)
    storage_length = np.full(count, SEQUENCE_LENGTH, dtype=np.int32)
    split_id = np.zeros(count, dtype=np.uint8)
    source_kind = np.zeros(count, dtype=np.uint8)
    source_file_id = np.zeros(count, dtype=np.int32)
    source_row_index = np.full(count, -1, dtype=np.int64)
    generator_source_id = np.full(count, -1, dtype=np.int16)
    raw_row_index = np.full(count, -1, dtype=np.int64)
    parent_names = sorted({r['parent_id'] for r in originals})
    parent_lookup = {parent: i for i, parent in enumerate(parent_names)}
    parent_group_id = np.zeros(count, dtype=np.int64)
    raw_count = 0
    for index, row in enumerate(originals):
        domain_id[index] = DOMAIN_NAMES.index(row['domain'])
        type_id[index] = CLASS_NAMES.index(row['noise_type'])
        lengths[index] = storage_length[index] = int(row['length'])
        source_kind[index] = 0 if row['domain'] == 'real' else 1
        source_file_id[index] = index
        parent_group_id[index] = parent_lookup[row['parent_id']]
        if row.get('raw_corresponding_path'):
            raw_row_index[index] = raw_count
            raw_count += 1
    split_id[:original_count] = original_splits
    offsets = []
    offset = original_count
    for source_index, data in enumerate(generated):
        selected = data['selected_rows']
        stop = offset + len(selected)
        offsets.append((offset, stop))
        domain_id[offset:stop] = 2
        type_id[offset:stop] = data['type_id'][selected] + 1
        lengths[offset:stop] = data['length'][selected]
        crop_start[offset:stop] = data['crop_start'][selected]
        source_kind[offset:stop] = 2 if 'all_lengths' in data['audit_source']['source'] else 3
        source_file_id[offset:stop] = original_count + source_index
        source_row_index[offset:stop] = selected
        generator_source_id[offset:stop] = data['source_id'][selected]
        parent_group_id[offset:stop] = len(parent_names)
        parent_names.append('noise_generator:' + data['audit_source']['source'])
        offset = stop
    if np.any(type_id <= 0) or np.any(type_id >= len(CLASS_NAMES)):
        raise ValueError('Noise type IDs must be one of the ten non-clean classes')
    write_chunk_rows = int(config.get('write_chunk_rows', config.get('storage', {}).get('write_chunk_rows', 64)))
    write_chunk_rows = max(1, min(write_chunk_rows, count))
    compression = config.get('compression', config.get('storage', {}).get('compression', 'lzf'))
    compression = None if compression in ('none', None, False) else compression
    samples_contiguous = bool(config.get('storage', {}).get('noise_samples_contiguous', True))
    build_signature = _digest({'schema': 1, 'audit_sha256': _sha256_file(energy_path),
                              'catalog_sha256': _sha256_file(catalog_path),
                              'inventory_sha256': _sha256_file(inventory_path),
                              'generated_metrics_sha256': [g['metrics_file_sha256'] for g in generated],
                              'minimum_usable_rms': min_rms, 'split': split_audit,
                              'compression': compression, 'write_chunk_rows': write_chunk_rows,
                              'samples_contiguous': samples_contiguous})
    noise_dir = root / 'noise'
    noise_dir.mkdir(parents=True, exist_ok=True)
    bank_path = noise_dir / 'noise_bank.h5'
    index_path = noise_dir / 'noise_index.npz'
    manifest_path = noise_dir / 'noise_manifest.json'
    if bank_path.is_file() and manifest_path.is_file() and index_path.is_file():
        saved = _json(manifest_path)
        with h5py.File(bank_path, 'r') as handle:
            if saved.get('build_signature') == build_signature and handle.attrs.get('complete') and len(handle['samples']) == count:
                print(f'NOISE BANK REUSE {count} templates', flush=True)
                return saved
        raise ValueError('Existing noise bank was built from different audited sources or settings; use a separate output root')
    partial_path = noise_dir / 'noise_bank.h5.partial'
    metadata_checkpoint = noise_dir / 'source_metadata.jsonl.partial'
    source_metadata = {}
    if metadata_checkpoint.is_file():
        with metadata_checkpoint.open(encoding='utf-8') as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                    source_metadata[int(row['source_file_id'])] = row
                except (ValueError, KeyError):
                    pass
    workers = int(noise_config.get('read_workers', 8))
    generated_block_rows = int(noise_config.get('generated_block_rows', 4096))
    vectors = {'row_id': row_id, 'domain_id': domain_id, 'type_id': type_id, 'length': lengths,
               'support_length': lengths, 'crop_start': crop_start, 'storage_length': storage_length,
               'split_id': split_id, 'source_kind': source_kind, 'parent_group_id': parent_group_id,
               'source_file_id': source_file_id, 'source_row_index': source_row_index, 'raw_row_index': raw_row_index}
    vectors['generator_source_id'] = generator_source_id
    with h5py.File(partial_path, 'a') as handle, metadata_checkpoint.open('a', encoding='utf-8') as metadata_stream:
        if 'samples' not in handle:
            handle.attrs.update(build_signature=build_signature, complete=False,
                                original_written=0, sequence_length=SEQUENCE_LENGTH,
                                sampling_rate_hz=150., sampling_rate_semantics='150 Hz index convention; absent generator fs is not inferred as measured',
                                class_names=json.dumps(CLASS_NAMES), domain_names=json.dumps(DOMAIN_NAMES),
                                source_kind_names=json.dumps(SOURCE_KIND_NAMES), split_names=json.dumps(SPLIT_NAMES),
                                generated_noise_split='train_only_unverified_training_provenance',
                                conditional_crop='maximum-energy contiguous window of requested condition length',
                                amplitude_policy='Retain source coordinates; no demeaning, clipping or normalization during packing')
            handle.create_dataset('samples', shape=(count, SEQUENCE_LENGTH), dtype='<f4',
                                  chunks=None if samples_contiguous else (write_chunk_rows, SEQUENCE_LENGTH),
                                  compression=None if samples_contiguous else compression, fillvalue=0)
            handle.attrs['samples_contiguous'] = samples_contiguous
            if raw_count:
                handle.create_dataset('raw_synthetic_samples', shape=(raw_count, SEQUENCE_LENGTH), dtype='f4',
                                      chunks=(min(write_chunk_rows, raw_count), SEQUENCE_LENGTH), compression=compression, fillvalue=0)
            for name, values in vectors.items():
                handle.create_dataset(name, data=values, compression=compression)
            handle.create_dataset('parent_names', data=np.asarray(parent_names, dtype=object), dtype=h5py.string_dtype('utf-8'))
        elif handle.attrs.get('build_signature') != build_signature or handle['samples'].shape != (count, SEQUENCE_LENGTH):
            raise ValueError('Partial noise bank does not match current audited sources/settings')
        original_written = int(handle.attrs.get('original_written', 0))
        for first in range(original_written, original_count, write_chunk_rows):
            last = min(original_count, first + write_chunk_rows)
            block = np.zeros((last - first, SEQUENCE_LENGTH), dtype=np.float32)
            with ThreadPoolExecutor(max_workers=workers) as executor:
                loaded = list(executor.map(_read_original, originals[first:last]))
            for local, item in enumerate(loaded):
                index = first + local
                length = int(lengths[index])
                if _stable_rms(item['samples']) <= min_rms or not np.isfinite(item['samples']).all():
                    raise ValueError(f'Float32 storage produced unusable original noise: {originals[index]["path"]}')
                block[local, :length] = item['samples']
                if raw_row_index[index] >= 0:
                    raw = np.zeros(SEQUENCE_LENGTH, dtype=np.float32)
                    raw[:length] = item['raw_samples']
                    handle['raw_synthetic_samples'][raw_row_index[index]] = raw
                meta = {**originals[index], 'source_file_id': index,
                        'file_sha256': item['file_sha256'], 'array_sha256': item['array_sha256'],
                        'raw_file_sha256': item.get('raw_file_sha256'),
                        'raw_array_sha256': item.get('raw_array_sha256'),
                        'raw_bank_row_index': int(raw_row_index[index]),
                        'source_sampling_rate_hz': originals[index].get('sampling_rate_hz'),
                        'label_quality': originals[index].get('label_quality', 'folder_weak_label' if domain_id[index] == 0 else 'generator_type_label')}
                source_metadata[index] = meta
                metadata_stream.write(json.dumps(meta, ensure_ascii=False, allow_nan=False) + '\n')
            handle['samples'][first:last] = block
            metadata_stream.flush()
            handle.attrs['original_written'] = last
            handle.flush()
            if last % (write_chunk_rows * 80) == 0 or last == original_count:
                print(f'NOISE ORIGINAL PACK {last}/{original_count}', flush=True)
        for source_index, data in enumerate(generated):
            source = data['audit_source']
            path = Path(source['path'])
            if _fingerprint(path) != source['file_fingerprint']:
                raise ValueError(f'Generated noise archive changed after audit: {path}')
            offset, stop = offsets[source_index]
            selected = data['selected_rows']
            progress_attr = f'generated_{source_index}_written'
            already = int(handle.attrs.get(progress_attr, 0))
            if already < len(selected):
                print(f'NOISE GENERATED PACK {source["source"]}: {len(selected)} rows', flush=True)
                # Stream all rows. This also supports a resumed subset by consuming
                # the ZIP member in order without a full in-memory decompression.
                with zipfile.ZipFile(path) as archive, archive.open('samples.npy') as stream:
                    shape, fortran, dtype = _npy_header(stream)
                    if fortran or tuple(shape) != (int(source['count']), 1, SEQUENCE_LENGTH) or dtype.hasobject:
                        raise ValueError(f'Unsupported generated samples layout: {shape}, {dtype}')
                    row_bytes = SEQUENCE_LENGTH * dtype.itemsize
                    packed = 0
                    for first in range(0, int(source['count']), generated_block_rows):
                        last = min(int(source['count']), first + generated_block_rows)
                        payload = stream.read((last - first) * row_bytes)
                        if len(payload) != (last - first) * row_bytes:
                            raise IOError('Unexpected end of generated samples archive')
                        select_first = int(np.searchsorted(selected, first))
                        select_last = int(np.searchsorted(selected, last))
                        packed = select_last
                        write_first = max(select_first, already)
                        if write_first >= select_last:
                            continue
                        samples = np.frombuffer(payload, dtype=dtype).reshape(last - first, SEQUENCE_LENGTH)
                        samples = samples[selected[write_first:select_last] - first].astype(np.float32)
                        if not np.isfinite(samples).all():
                            raise ValueError(f'Generated noise became nonfinite while packing: {path}')
                        # Full/crop energy had a full float64 audit; verify the stored
                        # crop again without removing a legitimate constant/step.
                        for local, original_index in enumerate(selected[write_first:select_last]):
                            start = int(data['crop_start'][original_index])
                            length = int(data['length'][original_index])
                            if _stable_rms(samples[local, start:start + length]) <= min_rms:
                                raise ValueError(f'Zero/tiny stored generated crop: {path}:{original_index}')
                        handle['samples'][offset + write_first:offset + select_last] = samples
                        handle.attrs[progress_attr] = select_last
                        handle.flush()
                        if packed % (generated_block_rows * 20) == 0 or packed == len(selected):
                            print(f'NOISE GENERATED PACK {source["source"]} {packed}/{len(selected)}', flush=True)
            metadata_id = original_count + source_index
            if metadata_id not in source_metadata:
                meta = {'source_file_id': metadata_id, 'path': str(path),
                        'source_kind': SOURCE_KIND_NAMES[int(source_kind[offset])],
                        'file_fingerprint': source['file_fingerprint'], 'file_sha256': _sha256_file(path),
                        'source_shape': [int(source['count']), 1, SEQUENCE_LENGTH],
                        'stored_count': len(selected), 'key': 'samples',
                        'type_names': CLASS_NAMES[1:], 'type_names_source': 'canonical type order verified against noise_dataset.py and numeric type_id',
                        'object_metadata_policy': 'No unpickling; only numeric samples/type_id/length/source_id loaded',
                        'label_quality': 'conditional_generator_label_not_human_verified',
                        'length_semantics': 'condition label, not measured nonzero support',
                        'crop_policy': 'maximum-energy contiguous window; crop_start saved per template',
                        'source_sampling_rate_hz': None, 'split': 'train',
                        'generator_training_provenance': 'unverified_original_noise_holdout_exclusion',
                        'audit_source': source}
                meta['audit_metrics_file_sha256'] = data['metrics_file_sha256']
                generator_root = path.parent.parent
                for filename, metadata_key in [('config.json', 'generator_config'),
                                               ('dataset_info.json', 'generator_training_dataset_info')]:
                    metadata_path = generator_root / filename
                    if metadata_path.is_file():
                        meta[metadata_key] = _json(metadata_path)
                        meta[metadata_key + '_sha256'] = _sha256_file(metadata_path)
                source_metadata[metadata_id] = meta
                metadata_stream.write(json.dumps(meta, ensure_ascii=False, allow_nan=False) + '\n')
                metadata_stream.flush()
        if len(source_metadata) != original_count + len(generated):
            raise ValueError('Noise source metadata checkpoint is incomplete')
        handle.attrs['complete'] = True
        handle.attrs['template_count'] = count
        if samples_contiguous:
            if handle['samples'].chunks is not None or handle['samples'].compression is not None:
                raise ValueError('Requested a contiguous uncompressed noise samples dataset')
            samples_offset = handle['samples'].id.get_offset()
            if samples_offset is None:
                raise ValueError('Contiguous noise samples have no allocated file offset')
            handle.attrs['samples_offset_bytes'] = int(samples_offset)
        handle.flush()
    partial_path.replace(bank_path)
    np.savez_compressed(index_path, **vectors)
    _save_json(noise_dir / 'source_metadata.json', {'sources': [source_metadata[k] for k in sorted(source_metadata)]})
    manifest = {'schema_version': 1, 'build_signature': build_signature,
                'noise_bank_path': 'noise/noise_bank.h5', 'path': 'noise/noise_bank.h5', 'key': 'samples',
                'pool_arrays_path': 'noise/noise_index.npz', 'pool': {'path': 'noise/noise_index.npz',
                'columns': list(vectors)}, 'source_metadata_path': 'noise/source_metadata.json',
                'class_names': CLASS_NAMES, 'classNames': CLASS_NAMES,
                'domain_names': DOMAIN_NAMES, 'source_kind_names': SOURCE_KIND_NAMES,
                'split_names': SPLIT_NAMES, 'count': count, 'original_count': original_count,
                'generated_count': generated_count, 'raw_correspondence_count': raw_count,
                'counts': _counts(domain_id, type_id, lengths, split_id),
                'split_counts': {name: int(np.count_nonzero(split_id == i)) for i, name in enumerate(SPLIT_NAMES)},
                'split_audit': split_audit,
                'energy_audit_sha256': _sha256_file(energy_path), 'noise_inventory_sha256': _sha256_file(inventory_path),
                'energy_policy': {'minimum_usable_rms': min_rms, 'stable_float64_rms': True,
                                  'zero_ac_rms': 'retain; nonzero constant/step is usable noise',
                                  'filename_max0': 'never used as exclusion criterion',
                                  'excluded_invalid_original_count': sum(not r['valid'] for r in all_original),
                                  'excluded_tiny_original_count': sum(r['valid'] and float(r['rms']) <= min_rms for r in all_original),
                                  'excluded_generated_count': sum(int(g['audit_source']['count']) - len(g['selected_rows']) for g in generated)},
                'sampling_rate_hz': 150., 'sequence_length': SEQUENCE_LENGTH,
                'array_layout': 'samples [template, time]; each template one channel, expanded by A/C mixing',
                'stored_original_policy': 'original variable lengths padded at end with zeros; use support_length',
                'stored_generated_policy': 'retain full2048 output; apply explicit crop_start/support_length when mixing',
                'samples_contiguous': samples_contiguous,
                'samples_offset_bytes': int(samples_offset) if samples_contiguous else None,
                'samples_dtype': '<f4',
                'generated_split_policy': 'train only; no original-noise holdout exclusion evidence',
                'amplitude_policy': 'source coordinates retained, not demeaned; A/C scaling happens during mixture construction',
                'actual_bank_bytes': bank_path.stat().st_size}
    _save_json(manifest_path, manifest)
    print(f'NOISE BANK COMPLETE {count} templates; {bank_path.stat().st_size} bytes', flush=True)
    return manifest
