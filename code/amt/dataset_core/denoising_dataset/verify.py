"""Independent on-disk verification of a completed, portable dataset bundle.

Final waveform validation is deliberately sampled: first/middle/last and four
deterministic random rows in every shard. It is distinct from the materializer's
checks on every emitted batch and the full source-noise energy audit. With
``thorough_metadata=True`` all small stored metadata are read and checked.
No model, PyTorch dependency, or original training-core import is required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import datetime
import os
from pathlib import Path
import re
import time
from typing import Any

import h5py
import numpy as np

from .materialize import SampleFactory, shard_jobs
from .protocols import split_frequency_target, visible_standardization
from .utils import SPLITS, TASKS, digest, file_sha, json_read, json_write, make_clean_index, seed64


def _require(condition, message):
    if not bool(condition):
        raise ValueError(message)


def _inside(root: Path, relative) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError(f'Bundle path escapes output root: {relative}') from None
    return path


def _arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}


def _text(value):
    return value.decode('utf-8') if isinstance(value, bytes) else str(value)


def _same(actual, expected, label):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape, f'{label}: shape mismatch {actual.shape}/{expected.shape}')
    if actual.dtype.kind == 'f' or expected.dtype.kind == 'f':
        equal = np.array_equal(actual, expected, equal_nan=True)
    else:
        equal = np.array_equal(actual, expected)
    _require(equal, f'{label}: deterministic replay mismatch')


def _near(actual, expected, label, atol=3e-6, rtol=3e-6):
    actual, expected = np.asarray(actual), np.asarray(expected)
    _require(actual.shape == expected.shape and np.allclose(actual, expected, atol=atol, rtol=rtol),
             f'{label}: numeric mismatch')


def _manifest_checks(root, manifest):
    _require(manifest.get('dataset_id') == digest(manifest['identity']), 'Dataset identity SHA-256 mismatch')
    identity = manifest['identity']
    _require(identity['config'] == manifest['config'], 'Manifest config differs from frozen identity')
    _require(identity['clean_manifest_id'] == manifest['clean']['manifest_id'], 'Clean identity differs from dataset identity')
    _require(identity['noise_build_signature'] == manifest['noise']['build_signature'], 'Noise signature differs from dataset identity')
    _require(identity['noise_manifest'] == manifest['noise'], 'Noise manifest differs from frozen dataset identity')
    clean = manifest['clean']
    _require(digest({k: v for k, v in clean.items() if k != 'manifest_id'}) == clean['manifest_id'],
             'Clean manifest SHA-256 mismatch')
    for name, expected in identity.get('kernel_sha256', {}).items():
        path = Path(__file__).parent / name
        _require(path.is_file() and file_sha(path) == expected, f'Frozen generation code changed: {name}')
    jobs = shard_jobs(manifest)
    if 'offline_jobs' in manifest:
        _require(manifest['offline_jobs'] == jobs, 'Stored shard plan differs from source counts and configuration')
    jobs_file = root / 'metadata/offline_jobs.json'
    if jobs_file.is_file():
        _require(json_read(jobs_file) == jobs, 'Portable shard job list differs from manifest')
    _require(float(manifest['config']['sampling_rate_hz']) == 150., 'Expected 150 Hz sampling')
    _require(int(manifest['config']['sequence_length']) == 2048, 'Expected 2048-sample clips')
    return {'dataset_identity_verified': True, 'clean_identity_verified': True,
            'kernel_files_verified': list(identity.get('kernel_sha256', {})), 'expected_shards': len(jobs)}


def _clean_checks(root, manifest):
    clean = manifest['clean']
    index = _arrays(root / 'metadata/clean_index.npz')
    expected = make_clean_index(clean)
    for key, values in expected.items():
        _same(index[key], values, f'Clean index/{key}')
    totals = {split: int(np.count_nonzero(index['split_id'] == role)) for role, split in enumerate(SPLITS)}
    _require(totals == clean['totals'], 'Flattened clean split totals disagree with clean manifest')
    global_groups = {}
    source_reports = []
    for source in clean['sources']:
        count = int(source['count'])
        assigned = np.concatenate([np.asarray(source['split_indices'][s], np.int64) for s in SPLITS])
        _require(len(np.unique(assigned)) == len(assigned), f'Clean split row overlap: {source["id"]}')
        _require(np.all((assigned >= 0) & (assigned < count)), f'Clean row index out of range: {source["id"]}')
        _require(len(assigned) == int(source['usable_count']), f'Clean usable count mismatch: {source["id"]}')
        excluded = np.asarray(source.get('excluded_invalid_indices', []), np.int64)
        _require(not np.intersect1d(assigned, excluded).size, f'Excluded clean rows were selected: {source["id"]}')
        if source.get('generated') or source.get('family') == 'generated':
            _require(not source['split_indices']['validation'] and not source['split_indices']['test'],
                     f'Unverified generated clean signals escaped training: {source["id"]}')
        path = _inside(root, source['path'])
        _require(path.is_file(), f'Missing clean source bank: {source["path"]}')
        if source.get('file_sha256'):
            _require(file_sha(path) == source['file_sha256'], f'Clean file checksum mismatch: {source["id"]}')
        groups = None
        with h5py.File(path, 'r') as handle:
            _require(handle[source.get('key', 'data')].shape == (count, 4, 2048), f'Clean bank shape mismatch: {source["id"]}')
            _require(handle[source.get('key', 'data')].dtype == np.dtype('float32'), f'Clean bank dtype mismatch: {source["id"]}')
            if 'group_id' in handle:
                groups = [_text(v) for v in handle['group_id'][:]]
            elif source.get('split_unit') in handle:
                prefix = 'syn_amt_shared_model' if source.get('family') == 'syn_discrete' else source['id']
                groups = [f'{prefix}:{_text(v)}' for v in handle[source['split_unit']][:]]
        groups_checked = 0
        for split in SPLITS:
            audit = source.get('split_audit', {}).get(split, {})
            if groups is not None:
                actual_groups = {groups[int(i)] for i in source['split_indices'][split]}
                if 'groups' in audit:
                    _require(actual_groups == set(audit['groups']), f'Clean parent audit differs from stored provenance: {source["id"]}/{split}')
            else:
                actual_groups = set(audit.get('groups', []))
            groups_checked += len(actual_groups)
            for group in actual_groups:
                _require(group not in global_groups or global_groups[group] == split, f'Clean physical parent appears in multiple splits: {group}')
                global_groups[group] = split
        raw_path = source.get('raw_path')
        if raw_path:
            path = _inside(root, raw_path)
            _require(path.is_file(), f'Missing raw clean bank: {raw_path}')
            if source.get('raw_file_sha256'):
                _require(file_sha(path) == source['raw_file_sha256'], f'Raw clean checksum mismatch: {source["id"]}')
            with h5py.File(path, 'r') as handle:
                _require(handle[source.get('key', 'data')].shape == (count, 4, 2048), f'Raw clean shape mismatch: {source["id"]}')
        source_reports.append({'id': source['id'], 'count': count, 'usable_count': len(assigned),
                               'split_counts': {s: len(source['split_indices'][s]) for s in SPLITS},
                               'provenance_group_count': groups_checked, 'file_checksum_verified': bool(source.get('file_sha256'))})
    # These exact-excitation links were established on complete long Hx/Hy
    # mother records before splitting; final verification checks their owners,
    # rather than falsely claiming a new complete content audit of all clips.
    excitation = clean.get('excitation_isolation_audit', {})
    assignments = clean.get('new_geology_group_assignment', {})
    linked = 0
    for pair in excitation.get('matched_old_new_pairs', []):
        owner = pair['inherited_split']
        _require(assignments.get('syn_continuous_res', {}).get(pair['new_model']) == owner,
                 f'New continuous excitation link has wrong split: {pair["new_model"]}')
        for old in pair['old_models']:
            match = re.match(r'M(\d+)', old)
            _require(match is not None, f'Cannot parse legacy excitation model: {old}')
            group = f'syn_amt_shared_model:{match.group(1)}'
            if group in global_groups:
                _require(global_groups[group] == owner, f'Shared old/new excitation crosses splits: {old}')
        linked += 1
    if excitation.get('new_3d_cross_parent_H_duplicates'):
        _require(set(assignments.get('syn_3d_res', {}).values()) <= {'train'}, 'Duplicated new 3D excitation escaped training')
    return {'split_totals': totals, 'source_banks': source_reports, 'parent_overlap_between_splits': 0,
            'exact_old_new_excitation_links_owner_checked': linked,
            'excitation_scope': 'Rechecked split ownership of full-record exact-Hx/Hy links established by the preparation audit',
            'generated_clean_train_only': True}, index


def _find_audit(root, name, expected_sha=None):
    candidates = []
    for folder in [root / 'source_audits', root / 'qa']:
        if folder.is_dir():
            candidates.extend(folder.rglob(name))
    for path in sorted(candidates):
        if expected_sha is None or file_sha(path) == expected_sha:
            return path
    return None


def _noise_checks(root, manifest):
    noise = manifest['noise']
    index = _arrays(_inside(root, noise['pool_arrays_path']))
    n = int(noise['count'])
    required = ['row_id', 'split_id', 'domain_id', 'type_id', 'length', 'crop_start', 'parent_group_id']
    for key in required:
        _require(index[key].shape == (n,), f'Invalid noise index column: {key}')
    _require(len(np.unique(index['row_id'])) == n, 'Noise bank row IDs are not unique')
    _require(np.all((index['row_id'] >= 0) & (index['row_id'] < n)), 'Noise bank row IDs out of range')
    _require(np.all((index['split_id'] >= 0) & (index['split_id'] < 3)), 'Invalid noise split ID')
    _require(np.all((index['domain_id'] >= 0) & (index['domain_id'] < len(noise['domain_names']))), 'Invalid noise domain ID')
    _require(np.all((index['type_id'] > 0) & (index['type_id'] < len(noise['class_names']))), 'Noise templates must have valid non-clean class IDs')
    _require(np.all((index['length'] >= 2) & (index['crop_start'] >= 0) &
                    (index['crop_start'].astype(np.int64) + index['length'] <= 2048)), 'Invalid stored noise crop')
    _require(np.all(index['split_id'][index['domain_id'] == 2] == 0), 'Generated noise escaped the training split')
    parent_sets = [set(index['parent_group_id'][index['split_id'] == r].tolist()) for r in range(3)]
    _require(all(not parent_sets[a].intersection(parent_sets[b]) for a in range(3) for b in range(a + 1, 3)), 'Noise parent group overlaps between splits')
    with h5py.File(_inside(root, noise['noise_bank_path']), 'r') as handle:
        data = handle[noise.get('key', 'samples')]
        _require(data.shape == (n, 2048) and data.dtype == np.dtype('float32'), 'Noise bank shape/dtype mismatch')
        _require(handle.attrs.get('build_signature') == noise['build_signature'], 'Noise bank build signature mismatch')
        _require(bool(handle.attrs.get('complete', False)), 'Noise bank is incomplete')
        _require(data.chunks is None and data.compression is None, 'Noise bank is not contiguous, uncompressed shared storage')
        _require(data.id.get_offset() == noise.get('samples_offset_bytes'), 'Noise memmap offset differs from manifest')
        for key, value in index.items():
            if key in handle:
                _same(handle[key][:], value, f'Noise bank index/{key}')
    metadata = json_read(_inside(root, noise['source_metadata_path']))['sources']
    packed_paths = {row.get('path') for row in metadata}
    audit_path = _find_audit(root, 'noise_energy_audit.json', noise.get('energy_audit_sha256'))
    _require(audit_path is not None, 'Missing matching portable evidence of the complete noise energy audit')
    audit = json_read(audit_path)
    _require(audit.get('complete', False), 'Source-noise energy audit is incomplete')
    zero_examples = [row for row in audit.get('original', {}).get('invalid_examples', []) if row.get('zero_energy')]
    _require(not any(row['path'] in packed_paths for row in zero_examples), 'Audited zero-energy noise was copied into the usable bank')
    selected_invalid = audit.get('original', {}).get('invalid_selected_count')
    if selected_invalid is not None:
        _require(int(noise['energy_policy']['excluded_invalid_original_count']) == int(selected_invalid),
                 'Reported original noise exclusions differ from complete energy audit')
    split_counts = {name: int(np.count_nonzero(index['split_id'] == i)) for i, name in enumerate(SPLITS)}
    _require(split_counts == noise['split_counts'], 'Noise split totals differ from manifest')
    # Sample the packed library at random plus every represented type/length/
    # domain/split combination; complete source energy was checked separately.
    selected = {0, n // 2, n - 1}
    rng = np.random.default_rng(seed64(manifest['dataset_id'], 'verify_noise_bank'))
    selected.update(rng.choice(n, size=min(64, n), replace=False).tolist())
    codes = np.column_stack([index[k] for k in ('split_id', 'domain_id', 'type_id', 'length')])
    _, first = np.unique(codes, axis=0, return_index=True)
    selected.update(first.tolist())
    with h5py.File(_inside(root, noise['noise_bank_path']), 'r') as handle:
        data = handle[noise.get('key', 'samples')]
        minimum = float(noise['energy_policy']['minimum_usable_rms'])
        for position in sorted(selected):
            values = np.asarray(data[int(index['row_id'][position])], np.float64)
            begin, length = int(index['crop_start'][position]), int(index['length'][position])
            _require(np.isfinite(values).all(), f'Nonfinite stored noise template: {position}')
            _require(float(np.sqrt(np.mean(values[begin:begin + length] ** 2))) > minimum,
                     f'Zero/tiny stored noise crop: {position}')
    return {'template_count': n, 'split_counts': split_counts, 'parent_overlap_between_splits': 0,
            'generated_noise_train_only': True, 'audited_zero_energy_examples_excluded': len(zero_examples),
            'energy_exclusion_policy': noise['energy_policy'], 'full_source_energy_audit_path': str(audit_path.relative_to(root)).replace('\\', '/'),
            'stored_noise_waveform_rows_sampled': len(selected),
            'stored_noise_waveform_scope': 'Stratified representatives plus up to 64 seeded random rows; not a second full energy scan'}, index


def _schedule_checks(root, manifest, clean_index, noise_index):
    report_path = root / 'metadata/noise_coverage_schedule_report.json'
    _require(report_path.is_file(), 'Missing fixed coverage schedule report')
    report = json_read(report_path)
    _require(report['dataset_id'] == manifest['dataset_id'], 'Coverage schedule belongs to a different dataset')
    n_snr = len(manifest['config']['noise']['snr_db'])
    output = {}
    for role, split in enumerate(SPLITS):
        entry = report['splits'][split]
        path = _inside(root, entry['path'])
        _require(file_sha(path) == entry['sha256'], f'Coverage schedule checksum mismatch: {split}')
        schedule = _arrays(path)
        count = int(np.count_nonzero(clean_index['split_id'] == role)) * n_snr
        _require(all(values.shape == (count,) for values in schedule.values()), f'Schedule row count mismatch: {split}')
        chosen = schedule['noise_index']
        _require(np.all((chosen >= 0) & (chosen < len(noise_index['row_id']))), f'Schedule template out of range: {split}')
        _require(np.all(noise_index['split_id'][chosen] == role), f'Schedule crosses noise split: {split}')
        lengths = noise_index['length'][chosen]
        _require(np.all((schedule['noise_start'] >= 0) & (schedule['noise_start'].astype(np.int64) + lengths <= 2048)), f'Schedule event out of range: {split}')
        available = np.flatnonzero(noise_index['split_id'] == role)
        used = np.unique(chosen)
        _require(len(used) == int(entry['unique_used']), f'Template coverage count mismatch: {split}')
        for domain in entry['domains']:
            pool = available[noise_index['domain_id'][available] == domain['domain_id']]
            unique = int(np.count_nonzero(noise_index['domain_id'][used] == domain['domain_id']))
            _require(unique == int(domain['unique_used']), f'Domain template coverage count mismatch: {split}')
            if domain.get('sufficient_slots'):
                _require(unique == len(pool), f'Coverage guarantee failed despite sufficient slots: {split}/{domain["domain_id"]}')
        output[split] = {'noisy_rows_per_scheme': count, 'available_noise_templates': len(available),
                         'unique_noise_templates_used': len(used), 'all_templates_used': len(used) == len(available)}
    return output


def _samples(rows, dataset_id, path):
    selected = {0, rows // 2, rows - 1}
    selected.update(np.random.default_rng(seed64(dataset_id, 'verify_shard', path)).choice(
        rows, size=min(4, rows), replace=False).tolist())
    return np.asarray(sorted(selected), np.int64)


def _waveform_checks(batch, job, config):
    for key, value in batch.items():
        if value.dtype.kind == 'f' and not (job['kind'] == 'control' and key == 'snr_target_db'):
            _require(np.isfinite(value).all(), f'Nonfinite sampled field: {key}')
    inputs, target = batch['input'], batch['target']
    _require(inputs.shape[1:] == (4, 2048) and target.shape == inputs.shape, 'Sampled waveform shape mismatch')
    if job['kind'] in ('noise', 'control'):
        mask = batch['noise_mask']
        _require(np.all((mask == 0) | (mask == 1)), 'Noise mask is not binary')
        _require(np.all(mask == mask[:, :1, :]), 'Four-channel injected intervals differ')
        scale = batch['scale'].astype(np.float64)
        _require(np.all(scale > 0), 'Noisy-input scale is not positive')
        low, high = split_frequency_target(target, config['sampling_rate_hz'], config['frequency']['cutoff_hz'])
        _near(batch['target_low'], low, 'FFT low target')
        _near(batch['target_high'], high, 'FFT high target')
        _near(batch['target_low'] + batch['target_high'], target, 'Complementary frequency labels')
        if job['kind'] == 'control':
            _same(inputs, target, 'Clean control input/target')
            _require(not np.any(mask), 'Clean control has an interference mask')
            _require(np.all(batch['class_id'] == 0) and np.all(batch['is_clean'] == 1), 'Clean control class/flag invalid')
            _require(np.isnan(batch['snr_target_db']).all(), 'Clean control SNR must be explicitly undefined (NaN)')
            return 0.
        expected = np.zeros_like(mask)
        for row, (start, length) in enumerate(zip(batch['noise_start'], batch['noise_length'])):
            expected[row, :, int(start):int(start) + int(length)] = 1
        _same(mask, expected, 'Noise event interval')
        noise = inputs.astype(np.float64) - target.astype(np.float64)
        _require(np.all(noise[mask == 0] == 0), 'Noise exists outside the labeled injected interval')
        # Reconstruct source coordinates before group SNR. Per-channel input
        # normalization would otherwise destroy the E/H aggregate energy ratio.
        signal_energy = np.mean((target.astype(np.float64) * scale) ** 2, axis=-1)
        noise_energy = np.mean((noise * scale) ** 2, axis=-1)
        _require(np.all(signal_energy > 0) and np.all(noise_energy > 0), 'SNR needs positive signal/noise energy')
        per_channel = 10. * np.log10(signal_energy / noise_energy)
        total = 10. * np.log10(signal_energy.mean(axis=1) / noise_energy.mean(axis=1))
        e = 10. * np.log10(signal_energy[:, :2].mean(axis=1) / noise_energy[:, :2].mean(axis=1))
        h = 10. * np.log10(signal_energy[:, 2:].mean(axis=1) / noise_energy[:, 2:].mean(axis=1))
        for name, value in [('snr_per_channel_db', per_channel), ('snr_total_db', total), ('snr_e_db', e), ('snr_h_db', h)]:
            _near(batch[name], value, f'Reconstructed {name}', atol=1e-4, rtol=0.)
        wanted = batch['snr_target_db']
        error = float(np.max(np.abs(per_channel - wanted[:, None]))) if job['scheme'] == 'A' else float(
            max(np.max(np.abs(value - wanted)) for value in (total, e, h)))
        _require(error <= 1e-3, f'Reconstructed target SNR error is {error} dB')
        return error
    mask, loss = batch['observed_mask'], batch['loss_mask']
    _require(np.all((mask == 0) | (mask == 1)), 'Observed mask is not binary')
    _same(loss, 1 - mask, 'Clean task loss mask')
    _require(np.all(inputs[mask == 0] == 0), 'Clean task reveals hidden input positions')
    _same(inputs[mask == 1], target[mask == 1], 'Observed clean input versus target')
    _require(np.all(mask.sum(axis=-1) > 0) and np.all(loss.sum(axis=(1, 2)) > 0), 'Degenerate clean task masks')
    if job['task'] == 'forecasting':
        for row, horizon in enumerate(batch['horizon']):
            _require(np.all(mask[row, :, :-int(horizon)] == 1) and not np.any(mask[row, :, -int(horizon):]), 'Forecast mask is not the specified terminal horizon')
    elif job['task'] == 'interpolation':
        for row, length in enumerate(batch['gap_length']):
            for channel, start in enumerate(batch['gap_start'][row]):
                expected = np.ones(2048, np.uint8)
                if start >= 0:
                    _require(start > 0 and start + length < 2048, 'Interpolation gap lacks both-side context')
                    expected[int(start):int(start) + int(length)] = 0
                _same(mask[row, channel], expected, 'Interpolation gap')
    if config['clean_tasks']['store_visible_standardized']:
        visible = visible_standardization(inputs, target, mask)
        for key, value in visible.items():
            _same(batch[key], value, f'Observed-only normalization/{key}')
        restored = batch['target_visible'].astype(np.float64) * batch['visible_scale'] + batch['visible_center']
        _near(restored, target, 'Visible target inverse transform', atol=1e-5, rtol=2e-6)
        _require(np.all(batch['input_visible'][mask == 0] == 0), 'Visible-standardized task reveals missing inputs')
    return 0.


def _metadata_checks(handle, job, factory, counts, coverage):
    """Read all low-dimensional fields, never all waveform arrays."""
    small = {key: np.asarray(value[:]) for key, value in handle.items()
             if isinstance(value, h5py.Dataset) and value.ndim >= 1 and int(np.prod(value.shape[1:])) <= 32}
    rows = job['stop'] - job['start']
    positions = np.arange(job['start'], job['stop'], dtype=np.int64)
    n_snr = len(factory.config['noise']['snr_db'])
    base = positions // n_snr if job['kind'] == 'noise' else positions
    ids = factory.split_rows[job['split']][base]
    for key, value in small.items():
        _require(value.shape[0] == rows, f'Metadata row count mismatch: {key}')
        if value.dtype.kind == 'f':
            if job['kind'] == 'control' and key == 'snr_target_db':
                _require(np.isnan(value).all(), 'Clean controls have a numeric SNR label')
            else:
                _require(np.isfinite(value).all(), f'Nonfinite metadata field: {key}')
    for key, expected in [('sample_index', positions), ('clean_index', ids),
                          ('source_id', factory.index['source_id'][ids]), ('source_row', factory.index['source_row'][ids])]:
        _same(small[key], expected, f'Full metadata/{key}')
    group = (job['kind'], job['split'], job.get('scheme', ''), job.get('task', ''))
    if job['kind'] == 'noise':
        conditions = positions % n_snr
        _same(small['snr_condition_index'], conditions, 'Full metadata/SNR condition')
        wanted = np.asarray(factory.config['noise']['snr_db'], np.float32)[conditions]
        _same(small['snr_target_db'], wanted, 'Full metadata/target SNR')
        _require(np.all(small['is_clean'] == 0), 'Corrupted rows marked as clean')
        _require(np.all(small['scale'] > 0), 'Metadata has invalid normalization scale')
        ni = small['noise_index']
        _require(np.all((ni >= 0) & (ni < len(factory.noise_index['row_id']))), 'Stored noise index out of range')
        _require(np.all(factory.noise_index['split_id'][ni] == SPLITS.index(job['split'])), 'Stored sample uses another split noise')
        for key, lookup in [('noise_bank_row', 'row_id'), ('class_id', 'type_id'), ('domain_id', 'domain_id'),
                            ('noise_length', 'length'), ('noise_crop_start', 'crop_start')]:
            _same(small[key], factory.noise_index[lookup][ni], f'Full metadata/template {key}')
        schedule = factory.schedules[job['split']]
        for key in ['noise_index', 'noise_start', 'draw_seed']:
            _same(small[key], schedule[key][positions], f'Full metadata/fixed schedule {key}')
        _require(np.all((small['noise_start'] >= 0) & (small['noise_start'].astype(np.int64) + small['noise_length'] <= 2048)), 'Stored noise interval out of range')
        if job['scheme'] == 'A':
            _require(np.max(np.abs(small['snr_per_channel_db'] - wanted[:, None])) <= 1e-3, 'Full stored channel-SNR diagnostics fail target')
        else:
            _require(max(np.max(np.abs(small[k] - wanted)) for k in ['snr_total_db', 'snr_e_db', 'snr_h_db']) <= 1e-3, 'Full stored group-SNR diagnostics fail target')
        coverage[job['split']][ni] = True
        fields = np.column_stack([small['source_id'], small['domain_id'], small['class_id'], small['noise_length'], conditions])
    elif job['kind'] == 'control':
        _require(np.all(small['class_id'] == 0) and np.all(small['is_clean'] == 1), 'Control class/clean flag invalid')
        _require(np.all(small['noise_index'] == -1) and np.all(small['noise_length'] == 0), 'Clean controls refer to noise')
        fields = small['source_id'][:, None]
    else:
        _require(np.all((small['missing_ratio_actual'] > 0) & (small['missing_ratio_actual'] < 1)), 'Clean task missing ratio invalid')
        if factory.config['clean_tasks']['store_visible_standardized']:
            _require(np.all(small['visible_scale'] > 0) and np.all(small['visible_observed_count'] > 0), 'Observed-only normalization metadata invalid')
        fields = np.column_stack([small['source_id'], small['condition_index'], small['channel_mode_id']])
    unique, numbers = np.unique(fields, axis=0, return_counts=True)
    counts[group].update({tuple(int(x) for x in row): int(number) for row, number in zip(unique, numbers)})
    return len(small)


def _reader_checks(root, factory):
    """Check actual reader-visible fields in both normalization modes.

    Readers share the already-open verification factory here so the 700k-row
    template catalog is not repeatedly partitioned for each task. This bypasses
    only lazy resource initialization, never sample generation or field mapping.
    """
    from .readers import OfflineDataset, DynamicDataset
    reports = []
    for split in SPLITS:
        for task in TASKS:
            if not len(factory.split_rows[split]):
                continue
            schemes = factory.config['noise']['schemes'] if task in TASKS[:4] else ['A']
            modes = ['visible', 'legacy'] if task in TASKS[4:] and factory.config['clean_tasks']['store_visible_standardized'] else ['legacy']
            for scheme in schemes:
                for normalization in modes:
                    offline = OfflineDataset(root, task, split, scheme, include_controls=False, normalization=normalization)
                    dynamic = DynamicDataset(root, task, split, scheme, epoch=0, normalization=normalization)
                    dynamic.factory, dynamic.pid = factory, os.getpid()
                    try:
                        _require(len(offline) == len(dynamic), f'Offline/dynamic reader length mismatch: {task}/{split}/{scheme}')
                        selected = sorted({0, len(dynamic) // 2, len(dynamic) - 1, min(5, len(dynamic) - 1)})
                        checked = set()
                        for position in selected:
                            a, b = offline[position], dynamic[position]
                            required = ['input', 'target', 'clean_index', 'source_id', 'source_row', 'sample_index']
                            required += ['noise_mask', 'class_id', 'scale'] if task in TASKS[:4] else ['observed_mask', 'loss_mask']
                            if task == 'frequency_denoising':
                                required += ['target_low', 'target_high']
                            _require(all(k in a and k in b for k in required), 'Reader omitted a required task field')
                            for key in sorted(set(a).intersection(b)):
                                if key in ('task', 'split'):
                                    continue
                                _same(a[key], b[key], f'Actual offline/dynamic reader/{task}/{split}/{scheme}/{normalization}/{key}')
                                checked.add(key)
                        if split != 'train':
                            _require(DynamicDataset(root, task, split, scheme, epoch=7).epoch == 0,
                                     'Evaluation reader allowed a varying augmentation epoch')
                        reports.append({'task': task, 'split': split, 'scheme': scheme, 'normalization': normalization,
                                        'rows': len(selected), 'common_fields_checked': sorted(checked)})
                    finally:
                        offline.close()
    return reports


def verify_dataset(root, thorough_metadata=True) -> dict[str, Any]:
    """Verify a complete bundle and write ``qa/final_verification.json``.

    Errors are collected in the report instead of silently accepting partial
    builds. ``passed`` is true only if every expected shard is complete and all
    enabled checks pass. The caller may raise or exit nonzero on false.
    """
    root = Path(root).resolve()
    begun = time.monotonic()
    report = {'schema_version': 1, 'passed': False, 'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'bundle_root': str(root), 'errors': [],
              'scope': {'source_file_checksums': 'Complete normalized and raw clean banks with recorded SHA-256',
                        'generation_validation': 'Materializer checks every generated batch: finite exports and SNR; preparation audits source noise energy in full',
                        'final_disk_waveforms': 'First/middle/last plus up to four deterministic random rows per expected shard',
                        'final_metadata': 'All stored low-dimensional numeric fields' if thorough_metadata else 'Only sampled rows; full metadata verification disabled',
                        'final_disk_waveforms_full_rescan': False,
                        'chunk_checksums': 'HDF5 Fletcher32 is checked automatically for chunks actually read; unread waveform chunks are not claimed re-read'},
              'shards': []}
    factory = None
    try:
        manifest = json_read(root / 'manifest.json')
        report['dataset_id'] = manifest['dataset_id']
        for label, action in [('identity', lambda: _manifest_checks(root, manifest)),
                              ('clean', lambda: _clean_checks(root, manifest)),
                              ('noise', lambda: _noise_checks(root, manifest))]:
            try:
                value = action()
                if label in ('clean', 'noise'):
                    report[label], _ = value
                else:
                    report[label] = value
            except Exception as error:
                report['errors'].append({'stage': label, 'message': f'{type(error).__name__}: {error}'})
        factory = SampleFactory(root)
        try:
            report['fixed_schedule'] = _schedule_checks(root, manifest, factory.index, factory.noise_index)
        except Exception as error:
            report['errors'].append({'stage': 'schedule', 'message': f'{type(error).__name__}: {error}'})
        jobs = shard_jobs(manifest)
        expected = {j['path'] for j in jobs}
        found = {str(p.relative_to(root)).replace('\\', '/') for p in (root / 'offline').rglob('*.h5')}
        missing, unexpected = sorted(expected - found), sorted(found - expected)
        report['completion'] = {'expected_shards': len(jobs), 'found_shards': len(found), 'missing_paths': missing,
                                'unexpected_paths': unexpected, 'expected_physical_rows': sum(j['stop'] - j['start'] for j in jobs)}
        if missing or unexpected:
            report['errors'].append({'stage': 'completion', 'message': 'Missing or unexpected offline shards'})
        counts = defaultdict(Counter)
        coverage = {split: np.zeros(len(factory.noise_index['row_id']), np.bool_) for split in SPLITS}
        replay_count = Counter()
        sampled_rows = 0
        metadata_rows = 0
        max_snr_error = 0.
        stored_rows = 0
        stored_bytes = 0
        for number, job in enumerate(jobs):
            path = _inside(root, job['path'])
            if not path.is_file():
                continue
            record = {'path': job['path'], 'passed': False}
            try:
                rows = job['stop'] - job['start']
                with h5py.File(path, 'r', rdcc_nbytes=8 * 1024**2) as handle:
                    _require(bool(handle.attrs.get('complete', False)), 'Shard is incomplete')
                    _require(handle.attrs.get('dataset_id') == manifest['dataset_id'], 'Shard dataset ID mismatch')
                    _require(handle.attrs.get('signature') == digest({'dataset_id': manifest['dataset_id'], 'job': job}), 'Shard job signature mismatch')
                    _require(int(handle.attrs.get('committed_rows', -1)) == rows, 'Committed rows do not equal final shard rows')
                    _require(int(handle.attrs.get('global_start', -1)) == job['start'] and int(handle.attrs.get('global_stop', -1)) == job['stop'], 'Shard global range differs from job')
                    for key, values in handle.items():
                        _require(isinstance(values, h5py.Dataset) and values.shape[0] == rows, f'Shard field row count mismatch: {key}')
                        _require(values.dtype.kind in 'biuf', f'Unsafe/non-numeric stored dataset: {key}')
                        if manifest['config']['storage']['fletcher32']:
                            _require(values.fletcher32, f'Expected Fletcher32 protection: {key}')
                    selected = _samples(rows, manifest['dataset_id'], job['path'])
                    batch = {key: np.asarray(value[selected]) for key, value in handle.items()}
                    error = _waveform_checks(batch, job, manifest['config'])
                    max_snr_error = max(max_snr_error, error)
                    sampled_rows += len(selected)
                    if thorough_metadata:
                        record['metadata_columns_checked_full'] = _metadata_checks(handle, job, factory, counts, coverage)
                        metadata_rows += rows
                    # Replay one representative shard of every kind/scheme/
                    # task in every split, with six SNR rows for noisy tasks.
                    key = (job['kind'], job['split'], job.get('scheme', ''), job.get('task', ''))
                    if not replay_count[key]:
                        positions = np.arange(job['start'], min(job['stop'], job['start'] + (6 if job['kind'] == 'noise' else 2)), dtype=np.int64)
                        actual = {field: np.asarray(values[positions - job['start']]) for field, values in handle.items()}
                        regenerated = factory.batch(job, positions)
                        _require(set(actual) == set(regenerated), 'Offline/dynamic export fields differ')
                        for field, values in regenerated.items():
                            _same(actual[field], values, f'Offline versus dynamic epoch0/{field}')
                        replay_count[key] += len(positions)
                    stored_rows += rows
                    stored_bytes += path.stat().st_size
                    record.update(passed=True, rows=rows, sampled_local_rows=selected.tolist(), maximum_sampled_snr_error_db=error)
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                report['errors'].append({'stage': 'shard', 'path': job['path'], 'message': record['error']})
            report['shards'].append(record)
            if (number + 1) % 10 == 0 or number + 1 == len(jobs):
                print(f'VERIFY SHARDS {number + 1}/{len(jobs)} sampled_waveform_rows={sampled_rows} metadata_rows={metadata_rows}', flush=True)
        report['completion'].update(verified_complete_rows=stored_rows, verified_shard_bytes=stored_bytes)
        report['disk_waveform_rows_sampled'] = sampled_rows
        report['disk_metadata_rows_checked_full'] = metadata_rows
        report['maximum_sampled_snr_error_db'] = max_snr_error
        report['dynamic_epoch0_replay'] = [{'kind': k[0], 'split': k[1], 'scheme': k[2], 'task': k[3], 'rows': value}
                                          for k, value in sorted(replay_count.items())]
        if not missing:
            try:
                report['actual_reader_epoch0_replay'] = _reader_checks(root, factory)
            except Exception as error:
                report['errors'].append({'stage': 'readers', 'message': f'{type(error).__name__}: {error}'})
        report['metadata_histograms'] = [{'kind': key[0], 'split': key[1], 'scheme': key[2], 'task': key[3],
                'columns': ['clean_source_id', 'noise_domain_id', 'class_id', 'noise_length', 'snr_condition_index'] if key[0] == 'noise' else
                           ['clean_source_id'] if key[0] == 'control' else ['clean_source_id', 'condition_index', 'channel_mode_id'],
                'rows': [{'values': list(values), 'count': count} for values, count in sorted(entries.items())]}
                for key, entries in sorted(counts.items())]
        if thorough_metadata:
            report['physical_shard_noise_template_coverage'] = {}
            for role, split in enumerate(SPLITS):
                available = factory.noise_index['split_id'] == role
                used = coverage[split]
                report['physical_shard_noise_template_coverage'][split] = {
                    'unique_templates_used': int(used.sum()), 'available_templates': int(available.sum()),
                    'unused_template_count': int(np.count_nonzero(available & ~used)), 'all_available_used': bool(np.all(used[available]))}
        report['passed'] = not report['errors'] and stored_rows == report['completion']['expected_physical_rows']
    except Exception as error:
        report['errors'].append({'stage': 'fatal', 'message': f'{type(error).__name__}: {error}'})
    finally:
        if factory is not None:
            factory.close()
        report['elapsed_seconds'] = time.monotonic() - begun
        json_write(root / 'qa/final_verification.json', report)
    print(f'FINAL VERIFICATION passed={report["passed"]} errors={len(report["errors"])} elapsed_seconds={report["elapsed_seconds"]:.1f}', flush=True)
    return report
