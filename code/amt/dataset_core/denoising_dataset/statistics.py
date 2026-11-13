"""Exact dataset statistics from prepared indices, audits and build receipts.

No model-count inference from window counts, no per-shard H5 reopening, and no
claim of a completed dataset from a banks-only or interrupted build. CSV views
are explicitly sectioned because the same pool has several statistical margins.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from .coverage_schedule import _eligible_rows
from .utils import SPLITS, TASKS, json_read


CSV_FIELDS = ['section', 'source_id', 'family', 'split', 'scheme', 'task', 'domain',
              'source_kind', 'type_id', 'noise_type', 'length', 'snr_db', 'count',
              'parent_group_count', 'parent_model_count', 'real_record_count', 'status', 'note']


def _atomic_text(path: Path, text: str, encoding: str = 'utf-8') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with partial.open('w', encoding=encoding, newline='') as stream:
        stream.write(text)
    partial.replace(path)


def _metadata(root: Path, source: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    path = source.get('normalization_metadata_path') or source.get('metadata_path')
    if path and (root / path).is_file():
        return json_read(root / path), str(path)
    return {}, None


def _known_group_count(audit: dict[str, Any]) -> tuple[int | None, list[str] | None, str]:
    if isinstance(audit.get('groups'), list):
        groups = sorted(set(str(x) for x in audit['groups']))
        return len(groups), groups, 'explicit_split_audit_group_ids'
    if audit.get('group_count') is not None:
        return int(audit['group_count']), None, 'declared_split_audit_group_count'
    if audit.get('unique_record_count') is not None:
        return int(audit['unique_record_count']), None, 'declared_split_audit_unique_record_count'
    return None, None, 'unavailable_not_inferred_from_clip_count'


def _source_statistics(root, manifest, clean_index, issues):
    sources = []
    family_rows = defaultdict(lambda: {'source_ids': [], 'counts': {s: 0 for s in SPLITS},
                                       'groups': {s: set() for s in SPLITS}, 'all_groups_known': True,
                                       'parent_semantics': set()})
    for sid, source in enumerate(manifest['clean']['sources']):
        metadata, metadata_path = _metadata(root, source)
        family = source.get('family', 'unspecified')
        generated = bool(source.get('generated', family == 'generated'))
        unit = source.get('split_unit', 'unspecified')
        semantic = ('generated_sample_identity_not_physical_parent' if generated else
                    'real_record' if family == 'real' and unit == 'record_idx' else
                    'geology_parent_model' if unit in ('model_id', 'geology_parent_group') else
                    'synthetic_record' if unit == 'record_idx' else 'audited_parent_group_unspecified')
        split_rows = {}
        combined_groups = set()
        all_groups_known = True
        for role, split in enumerate(SPLITS):
            audit = source.get('split_audit', metadata.get('split_audit', {})).get(split, {})
            if clean_index is not None:
                count = int(np.count_nonzero((clean_index['source_id'] == sid) & (clean_index['split_id'] == role)))
                count_method = 'prepared_clean_index'
            elif audit.get('count') is not None:
                count = int(audit['count'])
                count_method = 'source_split_audit'
            elif split in source.get('split_indices', {}):
                count = len(source['split_indices'][split])
                count_method = 'source_split_indices'
            else:
                count = 0
                count_method = 'missing_count_metadata'
                issues.append(f'{source["id"]}/{split}: no authoritative clean row count')
            if audit.get('count') is not None and count != int(audit['count']):
                issues.append(f'{source["id"]}/{split}: index count {count} differs from audit {audit["count"]}')
            n_groups, groups, method = _known_group_count(audit)
            if groups is not None:
                combined_groups.update(groups)
            else:
                all_groups_known = False
            split_rows[split] = {'clean_clips': count, 'count_method': count_method,
                                 'parent_group_count': n_groups, 'group_count_method': method,
                                 'parent_model_count': n_groups if semantic == 'geology_parent_model' else None,
                                 'real_record_count': n_groups if semantic == 'real_record' else None,
                                 'synthetic_record_count': n_groups if semantic == 'synthetic_record' else None}
            entry = family_rows[family]
            entry['counts'][split] += count
            if groups is not None:
                entry['groups'][split].update(groups)
            else:
                entry['all_groups_known'] = False
        family_rows[family]['source_ids'].append(source['id'])
        family_rows[family]['parent_semantics'].add(semantic)
        if all_groups_known:
            combined_count = len(combined_groups)
            combined_method = 'union_of_explicit_split_audit_group_ids'
        elif source.get('split_audit', {}).get('group_overlap_count') == 0 and all(
                split_rows[s]['parent_group_count'] is not None for s in SPLITS):
            combined_count = sum(split_rows[s]['parent_group_count'] for s in SPLITS)
            combined_method = 'sum_of_audited_disjoint_split_group_counts'
        else:
            combined_count = None
            combined_method = 'unavailable_not_inferred'
        declared_record_count = metadata.get('unique_record_count', source.get('unique_record_count'))
        declared_records_total = metadata.get('n_records_total')
        sources.append({'source_index': sid, 'source_id': source['id'], 'family': family,
                        'stored_bank_clips': int(source.get('count', source.get('shape', [0])[0])),
                        'usable_clips': sum(r['clean_clips'] for r in split_rows.values()),
                        'split_unit': unit, 'parent_semantics': semantic,
                        'unique_current_parent_groups': combined_count, 'parent_count_method': combined_method,
                        'parent_model_count': combined_count if semantic == 'geology_parent_model' else None,
                        'real_record_count': combined_count if semantic == 'real_record' else None,
                        'metadata_unique_record_count': int(declared_record_count) if declared_record_count is not None else None,
                        'metadata_n_records_total': int(declared_records_total) if declared_records_total is not None else None,
                        'metadata_record_count_scope': 'Declared source metadata; may describe pre-curation/long-record inventory. Current usable parents use split_audit.',
                        'split_policy': source.get('split_policy'), 'generated': generated,
                        'sampling_rate_hz': source.get('sampling_rate_hz', 150),
                        'metadata_path': metadata_path, 'source_geometry': source.get('source_geometry', metadata.get('source_geometry')),
                        'splits': split_rows})
    families = []
    for family, entry in sorted(family_rows.items()):
        semantics = sorted(entry['parent_semantics'])
        known = entry['all_groups_known']
        group_counts = {s: len(entry['groups'][s]) if known else None for s in SPLITS}
        union = len(set().union(*entry['groups'].values())) if known else None
        families.append({'family': family, 'source_count': len(entry['source_ids']),
                         'source_ids': entry['source_ids'], 'clean_clip_counts': entry['counts'],
                         'usable_clips': sum(entry['counts'].values()),
                         'unique_current_parent_groups': union, 'split_parent_group_counts': group_counts,
                         'parent_semantics': semantics,
                         'parent_model_count': union if semantics == ['geology_parent_model'] else None,
                         'real_record_count': union if semantics == ['real_record'] else None,
                         'group_count_scope': 'union across pipelines using explicit namespaced parent IDs; unavailable where IDs absent'})
    return sources, families


def _noise_statistics(root, manifest, issues):
    path = root / manifest['noise']['pool_arrays_path']
    if not path.is_file():
        issues.append('Prepared noise index is absent')
        return {'available_templates': None, 'joint_available_counts': manifest['noise'].get('counts', []),
                'margins': {}, 'count_method': 'manifest_only_index_missing'}
    with np.load(path, allow_pickle=False) as archive:
        ni = {k: archive[k].copy() for k in archive.files}
    eligible = _eligible_rows(ni)
    domains = manifest['noise'].get('domain_names', ['real', 'synthetic', 'generated'])
    classes = manifest['noise'].get('class_names', [])
    source_names = manifest['noise'].get('source_kind_names', [])
    lengths = sorted(np.unique(ni['length'][eligible]).tolist())
    n_domain = int(ni['domain_id'][eligible].max()) + 1 if eligible.any() else len(domains)
    source_kinds = ni.get('source_kind', ni['domain_id']).astype(np.int64)
    n_kind = int(source_kinds[eligible].max()) + 1 if eligible.any() else max(1, len(source_names))
    n_type = int(ni['type_id'][eligible].max()) + 1 if eligible.any() else max(1, len(classes))
    n_length = max(1, len(lengths))
    selected = np.flatnonzero(eligible)
    length_index = np.searchsorted(np.asarray(lengths), ni['length'][selected]) if len(selected) else np.empty(0, np.int64)
    codes = (((ni['split_id'][selected].astype(np.int64) * n_domain + ni['domain_id'][selected]) * n_kind
              + source_kinds[selected]) * n_type + ni['type_id'][selected]) * n_length + length_index
    counts = np.bincount(codes)
    joint = []
    margins = {split: {'total': 0, 'domain': {}, 'type': {}, 'length': {}, 'source_kind': {}} for split in SPLITS}
    for code in np.flatnonzero(counts):
        value, length_id = divmod(int(code), n_length)
        value, type_id = divmod(value, n_type)
        value, kind = divmod(value, n_kind)
        role, domain_id = divmod(value, n_domain)
        count = int(counts[code])
        split = SPLITS[role]
        row = {'split': split, 'domain_id': domain_id,
               'domain': domains[domain_id] if domain_id < len(domains) else str(domain_id),
               'source_kind': kind, 'source': source_names[kind] if kind < len(source_names) else str(kind),
               'type_id': type_id, 'noise_type': classes[type_id] if type_id < len(classes) else str(type_id),
               'length': int(lengths[length_id]), 'count': count}
        joint.append(row)
        margin = margins[split]
        margin['total'] += count
        for dimension, key in [('domain', row['domain']), ('type', row['noise_type']),
                               ('length', str(row['length'])), ('source_kind', row['source'])]:
            margin[dimension][key] = margin[dimension].get(key, 0) + count
    return {'stored_index_rows': len(ni['row_id']), 'available_templates': int(eligible.sum()),
            'excluded_index_rows': int((~eligible).sum()), 'count_method': 'prepared_numeric_noise_index',
            'joint_available_counts': joint, 'margins': margins,
            'generated_noise_split_policy': manifest['noise'].get('generated_split_policy'),
            'energy_policy': manifest['noise'].get('energy_policy'),
            'raw_correspondence_count': manifest['noise'].get('raw_correspondence_count')}


def _planned_and_actual(manifest, status, clean_totals, issues):
    snrs = list(manifest['config']['noise']['snr_db'])
    schemes = list(manifest['config']['noise']['schemes'])
    controls = bool(manifest['config']['materialization'].get('clean_control_per_clip', True))
    tasks = ['forecasting', 'interpolation', 'imputation']
    planned = {}
    for split in SPLITS:
        n = clean_totals[split]
        noisy = {scheme: n * len(snrs) for scheme in schemes}
        planned[split] = {'clean_clips': n, 'noisy_rows_by_scheme': noisy,
                          'shared_noisy_waveform_rows': sum(noisy.values()),
                          'independent_clean_control_rows': n if controls else 0,
                          'clean_task_rows': {task: n for task in tasks},
                          'noise_task_views': {task: {'shared_noisy_rows': sum(noisy.values()),
                                                    'shared_clean_control_rows': n if controls else 0}
                                               for task in TASKS[:4]},
                          'physically_exported_task_rows': sum(noisy.values()) + (n if controls else 0) + n * len(tasks)}
    jobs = manifest.get('offline_jobs', [])
    job_map = {j['path']: j for j in jobs}
    if len(job_map) != len(jobs):
        issues.append('Offline plan has duplicate shard paths')
    completed = status.get('shards', [])
    seen = set()
    actual = {s: {'noisy_rows_by_scheme': {scheme: 0 for scheme in schemes},
                  'independent_clean_control_rows': 0, 'clean_task_rows': {task: 0 for task in tasks},
                  'physically_exported_task_rows': 0} for s in SPLITS}
    unknown = []
    for record in completed:
        path = record['path']
        if path in seen:
            issues.append(f'Duplicate build receipt: {path}')
            continue
        seen.add(path)
        job = job_map.get(path)
        if job is None:
            unknown.append(path)
            continue
        rows = int(record['rows'])
        if rows != int(job['stop']) - int(job['start']):
            issues.append(f'Build receipt row count mismatch: {path}')
        entry = actual[job['split']]
        if job['kind'] == 'noise':
            entry['noisy_rows_by_scheme'][job['scheme']] += rows
        elif job['kind'] == 'control':
            entry['independent_clean_control_rows'] += rows
        elif job['kind'] == 'clean':
            entry['clean_task_rows'][job['task']] += rows
        entry['physically_exported_task_rows'] += rows
    for s in SPLITS:
        actual[s]['shared_noisy_waveform_rows'] = sum(actual[s]['noisy_rows_by_scheme'].values())
    return {'snr_db': snrs, 'schemes': schemes, 'planned': planned, 'actual_completed_receipts': actual,
            'offline_jobs_planned': len(jobs), 'completed_receipts_unique': len(seen),
            'unmatched_receipt_paths': unknown,
            'planned_exported_task_rows': sum(x['physically_exported_task_rows'] for x in planned.values()),
            'actual_completed_exported_task_rows': sum(x['physically_exported_task_rows'] for x in actual.values()),
            'row_count_scope': 'The four noise tasks share one waveform/label record; counted once. Clean controls are separate, once per clip. Each clean task has its own export.'}


def _csv_rows(stats):
    rows = []
    for source in stats['clean']['sources']:
        for split, values in source['splits'].items():
            rows.append({'section': 'clean_source_split', 'source_id': source['source_id'], 'family': source['family'],
                         'split': split, 'count': values['clean_clips'], 'parent_group_count': values['parent_group_count'],
                         'parent_model_count': values['parent_model_count'], 'real_record_count': values['real_record_count'],
                         'note': source['parent_semantics'] + '; ' + values['group_count_method']})
    for family in stats['clean']['families']:
        for split, count in family['clean_clip_counts'].items():
            rows.append({'section': 'clean_family_split', 'family': family['family'], 'split': split,
                         'count': count, 'parent_group_count': family['split_parent_group_counts'][split],
                         'note': 'Cross-pipeline parent IDs deduplicated only when explicit IDs are available'})
    for entry in stats['noise']['joint_available_counts']:
        rows.append({'section': 'noise_available_joint', **entry})
    for split, margins in stats['noise']['margins'].items():
        for dimension in ['domain', 'type', 'length', 'source_kind']:
            for key, count in margins[dimension].items():
                name = 'noise_type' if dimension == 'type' else dimension
                rows.append({'section': 'noise_available_margin_' + dimension, 'split': split, name: key, 'count': count,
                             'note': 'A statistical margin of the same noise pool; do not sum across sections'})
    for stage, split_rows in [('planned', stats['tasks']['planned']),
                               ('actual', stats['tasks']['actual_completed_receipts'])]:
        for split, entry in split_rows.items():
            for scheme, count in entry['noisy_rows_by_scheme'].items():
                rows.append({'section': stage + '_shared_noise_rows', 'split': split, 'scheme': scheme,
                             'count': count, 'note': 'Shared by noise_localization/noise_classification/denoising/frequency_denoising'})
            rows.append({'section': stage + '_independent_clean_controls', 'split': split,
                         'count': entry['independent_clean_control_rows']})
            for task, count in entry['clean_task_rows'].items():
                rows.append({'section': stage + '_clean_task_rows', 'split': split, 'task': task, 'count': count})
    schedule = stats.get('noise_schedule') or {}
    for split, descriptor in schedule.get('splits', {}).items():
        for entry in descriptor.get('source_type_length_snr_histogram', []):
            rows.append({'section': 'epoch0_actual_noise_allocation', 'split': split, **entry,
                         'note': 'One schedule shared by A/C; counts per scheme, not a claim of joint uniformity'})
        for domain in descriptor.get('domains', []):
            rows.append({'section': 'epoch0_unique_template_coverage', 'split': split, 'domain': domain['domain'],
                         'count': domain['unique_used'], 'note': f"available={domain['available']}; full_coverage={domain['full_coverage']}; draws={domain['draws']}"})
    rows.append({'section': 'disk_file_size_bytes', 'count': stats['disk']['actual_file_size_bytes'],
                 'status': stats['build']['manifest_status'], 'note': 'Logical sum of every regular file size, including this report; filesystem allocation overhead excluded'})
    return rows


def _csv_text(stats):
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, CSV_FIELDS, extrasaction='ignore')
    writer.writeheader()
    writer.writerows(_csv_rows(stats))
    return stream.getvalue()


def _explanation(stats):
    complete = '已完整导出并通过最终验证' if stats['build']['verified_complete'] else '已完整导出，尚不能称最终验证通过' if stats['build']['offline_export_complete'] else '尚未完整导出（当前统计为准备阶段或部分运行快照）'
    totals = stats['clean']['totals']
    lines = [f"数据集统计说明：{stats['dataset_id']}", f"状态：{complete}",
             f"清单状态：{stats['build']['manifest_status']}；构建状态：{stats['build']['build_state']}",
             f"分量：Ex、Ey、Hx、Hy；每条2048点；采样率150 Hz（约13.653秒）。",
             f"有效干净窗口：训练{totals['train']:,}、验证{totals['validation']:,}、测试{totals['test']:,}，总计{sum(totals.values()):,}。",
             '', '各来源的有效窗口与父记录／模型数：']
    for source in stats['clean']['sources']:
        counts = source['splits']
        parents = source['parent_model_count'] if source['parent_model_count'] is not None else source['real_record_count']
        meaning = '父模型' if source['parent_model_count'] is not None else '真实记录' if source['real_record_count'] is not None else source['parent_semantics']
        lines.append(f"- {source['source_id']}（{source['family']}）：train={counts['train']['clean_clips']:,} / validation={counts['validation']['clean_clips']:,} / test={counts['test']['clean_clips']:,}；{meaning}={parents if parents is not None else '未提供可据以判断的物理父模型数'}。")
    lines.extend(['父记录／模型数读取明确的split_audit分组和metadata，不能把2048窗口数当成独立记录或模型数。A/B管线共享的模型在family统计中按明确父ID去重；未知时保留空值。',
                  '', f"有效噪声模板：{stats['noise']['available_templates'] if stats['noise']['available_templates'] is not None else '索引缺失'}。"])
    for split, entry in stats['noise']['margins'].items():
        detail = '、'.join(f'{domain}={count:,}' for domain, count in entry['domain'].items())
        lines.append(f'- {split}：{entry["total"]:,}；{detail}。')
    lines.extend(['生成噪声只用于训练；Norm/Raw同模板不作为两个独立噪声。检查真实数据能量；非零常量／阶跃不会因去均值后能量为零而被排除。',
                  '', f"SNR：{', '.join(str(x) for x in stats['tasks']['snr_db'])} dB；方案：{', '.join(stats['tasks']['schemes'])}。",
                  '四个噪声任务共享同一份含噪输入和不同标签，不把同一波形重复计算为四份物理数据。'])
    for split, entry in stats['tasks']['planned'].items():
        schemes = '、'.join(f'{scheme}={count:,}' for scheme, count in entry['noisy_rows_by_scheme'].items())
        clean_tasks = '、'.join(f'{task}={count:,}' for task, count in entry['clean_task_rows'].items())
        lines.append(f'- {split}计划：含噪 {schemes}；独立干净对照={entry["independent_clean_control_rows"]:,}；{clean_tasks}。')
    lines.append(f"计划物理任务记录总计{stats['tasks']['planned_exported_task_rows']:,}；完成回执中的物理任务记录{stats['tasks']['actual_completed_exported_task_rows']:,}。")
    schedule = stats.get('noise_schedule')
    if schedule:
        lines.extend(['', f"固定epoch0模板覆盖：{schedule.get('unique_used_all_splits')} / {schedule.get('eligible_template_count')}；全部覆盖={schedule.get('full_coverage_all_splits')}。",
                      'Latin轮转平衡噪声域与SNR；配额足够时先覆盖全部模板，再按类型与长度补齐，最后混合打乱。',
                      '模板数量不同，基础全覆盖会保留不同类型、长度和生成来源的数量差异，不能声称所有联合条件严格均衡。CSV和JSON保存实际(source_kind,type,length,SNR)频数。A/C共享同一噪声选择、位置与基础扰动种子。'])
    else:
        lines.extend(['', '尚无固定噪声覆盖计划，不能报告实际模板覆盖和联合条件分配。'])
    three_d_note = ('三维电阻率来源仅两个地质父模型。信号由局部独立一维列计算，不包含横向电磁耦合；若共享Hx/Hy激励，遵照excitation_isolation_audit仅用于训练。它不代表广泛三维地质独立测试。'
                    if stats['three_dimensional_source_scope']['source_ids'] else
                    '当前准备的来源没有三维电阻率数据；没有为缺失来源推算父模型数。')
    lines.extend(['', '来源数量存在差异：按全部干净窗口展开后，各source的行数不会天然相等。保留source_id、source_row和family，后续训练可用sampler重新平衡，不能把来源数量不平衡当成已平衡。',
                  three_d_note,
                  '', '文件与状态：',
                  f"完成分片：{stats['build']['completed_shards']} / {stats['build']['expected_shards']}。",
                  f"实际文件总大小：{stats['disk']['actual_file_size_bytes']:,} bytes（{stats['disk']['actual_file_size_bytes']/1024**3:.3f} GiB）；文件数：{stats['disk']['file_count']}。",
                  '这是所有文件st_size之和，包含基础素材、代码、索引、QA及任务分片，不含文件系统分配单元等额外开销。运行中统计只是时间点快照。',
                  'dataset_statistics.csv按section区分来源、边际、联合条件、任务计划与实际回执；同一池会出现在多个统计维度，不能直接把所有CSV行count相加。'])
    if stats['integrity_issues']:
        lines.extend(['', '当前发现的统计一致性问题：'] + ['- ' + x for x in stats['integrity_issues']])
    return '\n'.join(lines) + '\n'


def write_statistics(root: str | Path) -> dict[str, Any]:
    """Write JSON, sectioned UTF8 CSV, and a concise Chinese reading guide.

    Exact row counts use prepared indices. Actual task rows use completed build
    receipts joined to the authoritative offline_jobs list. Final completeness
    requires matching receipts, planned rows and physical file existence; shards
    are not reopened because the master's final verifier owns that check.
    """
    root = Path(root).resolve()
    manifest = json_read(root / 'manifest.json')
    status = json_read(root / 'build_status.json') if (root / 'build_status.json').is_file() else {}
    issues = []
    clean_path = root / 'metadata/clean_index.npz'
    clean_index = None
    if clean_path.is_file():
        with np.load(clean_path, allow_pickle=False) as archive:
            clean_index = {k: archive[k].copy() for k in archive.files}
    sources, families = _source_statistics(root, manifest, clean_index, issues)
    totals = {split: sum(source['splits'][split]['clean_clips'] for source in sources) for split in SPLITS}
    for split, declared in manifest['clean'].get('totals', {}).items():
        if split in totals and totals[split] != int(declared):
            issues.append(f'Clean total differs from manifest: {split}={totals[split]} vs {declared}')
    noise = _noise_statistics(root, manifest, issues)
    tasks = _planned_and_actual(manifest, status, totals, issues)
    schedule_path = root / 'metadata/noise_coverage_schedule_report.json'
    schedule = json_read(schedule_path) if schedule_path.is_file() else manifest.get('noise_schedule')
    if schedule and schedule.get('dataset_id') != manifest['dataset_id']:
        issues.append('Noise schedule belongs to a different dataset ID')
    if schedule:
        for split, descriptor in schedule.get('splits', {}).items():
            expected = totals[split] * len(tasks['snr_db'])
            if int(descriptor['count']) != expected:
                issues.append(f'Noise schedule row count mismatch: {split}')
    json_path = root / 'metadata/dataset_statistics.json'
    csv_path = root / 'metadata/dataset_statistics.csv'
    txt_path = root / 'qa/数据集统计说明.txt'
    own_outputs = {json_path, csv_path, txt_path}
    own_partials = {p.with_name(p.name + '.partial') for p in own_outputs}
    snapshot = []
    for path in root.rglob('*'):
        if path in own_outputs or path in own_partials or not path.is_file():
            continue
        stat = path.stat()
        snapshot.append((str(path.relative_to(root)).replace('\\', '/'), stat.st_size))
    existing_paths = {p for p, _ in snapshot}
    missing_completed = [r['path'] for r in status.get('shards', []) if r['path'] not in existing_paths]
    if missing_completed:
        issues.append(f'{len(missing_completed)} completed shard receipts have missing files')
    expected_shards = int(status.get('expected_shards', tasks['offline_jobs_planned']))
    completed_shards = int(status.get('completed_shards', tasks['completed_receipts_unique']))
    export_complete = (status.get('state') in ('materialized', 'verified_complete', 'complete')
                       and manifest.get('status') in ('materialized', 'verified_complete')
                       and expected_shards == tasks['offline_jobs_planned']
                       and completed_shards == expected_shards == tasks['completed_receipts_unique']
                       and tasks['actual_completed_exported_task_rows'] == tasks['planned_exported_task_rows']
                       and not missing_completed and not tasks['unmatched_receipt_paths'] and not issues)
    verified = export_complete and manifest.get('status') == 'verified_complete'
    three_d = next((family for family in families if family['family'] == 'syn_3d'), None)
    stats = {'schema_version': 1, 'dataset_id': manifest['dataset_id'],
             'sampling_rate_hz': manifest['config'].get('sampling_rate_hz', 150),
             'sequence_length': manifest['config'].get('sequence_length', 2048),
             'channel_names': ['Ex', 'Ey', 'Hx', 'Hy'],
             'class_names': manifest['noise'].get('class_names', []),
             'clean': {'sources': sources, 'families': families, 'totals': totals,
                       'usable_clips': sum(totals.values()), 'source_count': len(sources),
                       'source_balance': 'all usable clips retained; source counts differ; source_id preserved for later sampler balancing'},
             'noise': noise, 'tasks': tasks, 'noise_schedule': schedule,
             'build': {'manifest_status': manifest.get('status', 'unspecified'),
                       'build_state': status.get('state', 'not_materialized'),
                       'expected_shards': expected_shards, 'completed_shards': completed_shards,
                       'offline_export_complete': export_complete, 'verified_complete': verified,
                       'complete': verified, 'missing_completed_shard_paths': missing_completed,
                       'status_scope': 'Build receipts and master manifest status; no redundant per-shard H5 reopening'},
             'three_dimensional_source_scope': {'source_ids': three_d['source_ids'] if three_d else [],
                       'declared_parent_geology_limit': 2 if three_d else None,
                       'audited_unique_parent_groups': three_d['unique_current_parent_groups'] if three_d else None,
                       'forward_signal_model': 'independent local 1D columns without lateral electromagnetic coupling',
                       'split_policy': 'actual source policy in sources; shared excitation parents remain train-only',
                       'excitation_audit': manifest['clean'].get('excitation_isolation_audit')},
             'integrity_issues': issues,
             'disk': {'actual_file_size_bytes': sum(size for _, size in snapshot),
                      'file_count': len(snapshot) + len(own_outputs),
                      'size_scope': 'logical sum of every regular file st_size including statistics outputs; filesystem allocation overhead excluded',
                      'snapshot_consistency': 'point-in-time while running; final snapshot only when build is complete',
                      'offline_h5_partial_count': sum(p.startswith('offline/') and p.endswith('.h5.partial') for p, _ in snapshot)}}
    # Include these three output files in the exact byte total without a recursive
    # self-size error. Decimal byte-count fields stabilize after a few iterations.
    base_bytes = sum(size for _, size in snapshot)
    for _ in range(12):
        csv_text = '\ufeff' + _csv_text(stats)
        txt_text = _explanation(stats)
        json_text = json.dumps(stats, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
        new_total = base_bytes + len(csv_text.encode('utf-8')) + len(txt_text.encode('utf-8')) + len(json_text.encode('utf-8'))
        if new_total == stats['disk']['actual_file_size_bytes']:
            break
        stats['disk']['actual_file_size_bytes'] = new_total
    else:
        raise RuntimeError('Statistics output byte count did not stabilize')
    _atomic_text(csv_path, csv_text)
    _atomic_text(txt_path, txt_text)
    _atomic_text(json_path, json_text)
    return stats
