"""
GeoAMT Clean Dataset(无 split 版本)
====================================
功能:
    1. 加载单个 source 的切片(可选 normalized 或 raw)
    2. 多 source 混合 Dataset(MultiSourceCleanDataset)
    3. 加权采样器(make_balanced_sampler)——多 source 比例控制
    4. 按 model_type 等量采样(make_per_type_sampler)
    5. 按 model_type 过滤索引(filter_indices_by_model_type)

预处理时 Syn-AMT 全部 601 个模型都已处理,采样在训练阶段灵活控制。

使用示例:

    # === 用法 1: 用全部 Syn-AMT 数据,各 model_type 等量采样 ===
    from geoamt_clean_dataset import CleanAMTDataset, make_per_type_sampler
    ds = CleanAMTDataset('data/generator_sources/syn_amt_A/clean_normalized.h5',
                         augment=True)
    print(ds.get_model_type_distribution())   # 看看各类有多少切片
    sampler = make_per_type_sampler(ds)        # 12 类等量采样
    
    # === 用法 2: 只用部分 model_type ===
    from geoamt_clean_dataset import filter_indices_by_model_type
    indices = filter_indices_by_model_type(
        'data/generator_sources/syn_amt_A/clean_normalized.h5',
        model_types=['halfspace', 'two_layer_conductive_cover'])
    ds = CleanAMTDataset('data/generator_sources/syn_amt_A/clean_normalized.h5',
                         indices=indices, augment=True)
    
    # === 用法 3: 每个 model_type 限制最多 N 个切片 ===
    indices = filter_indices_by_model_type(
        'data/generator_sources/syn_amt_A/clean_normalized.h5',
        per_type_limit=500)        # 每类最多 500 切片
    ds = CleanAMTDataset('data/generator_sources/syn_amt_A/clean_normalized.h5',
                         indices=indices)
    
    # === 用法 4: 多 source 混合,实测 : 合成 = 1 : 1 ===
    from geoamt_clean_dataset import MultiSourceCleanDataset, make_balanced_sampler
    ds = MultiSourceCleanDataset({
        'real':       'data/generator_sources/real/clean_normalized.h5',
        'syn_amt_A':  'data/generator_sources/syn_amt_A/clean_normalized.h5',
        'syn_amt_B':  'data/generator_sources/syn_amt_B/clean_normalized.h5',
        'syn_mt':     'data/generator_sources/syn_mt/clean_normalized.h5',
    }, augment=True)
    
    sampler = make_balanced_sampler(ds, weight_per_source={
        'real':      1.0,
        'syn_amt_A': 1.0 / 3,
        'syn_amt_B': 1.0 / 3,
        'syn_mt':    1.0 / 3,
    })
    
    # === 用法 5: 加载 raw 数据,用于与噪声叠加 ===
    raw_ds = CleanAMTDataset('data/generator_sources/real/clean_raw.h5', augment=False)
    sample_raw = raw_ds[0]  # 物理单位的 (4, 2048)
"""

import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler


# ============================================================
# 单 source Dataset
# ============================================================
class CleanAMTDataset(Dataset):
    """
    加载单个 source 的切片
    返回: (4, 2048) torch.float32
    
    特性:
        - 支持只用部分切片(通过 indices 参数过滤)
        - 自动从 HDF5 读取每个切片的 model_type/model_id 标签
        - 提供 filter_by_model_type 等工具方法
    """
    
    def __init__(self, h5_path, augment=False, augment_config=None,
                 return_metadata=False, indices=None):
        """
        h5_path: 'clean_normalized.h5' 或 'clean_raw.h5'
        augment: 是否做数据增强
        return_metadata: 是否返回 (clip, meta_dict)
        indices: 限制只用部分切片的索引列表(None = 全部)
        """
        self.h5_path = str(h5_path)
        self.augment = augment
        self.return_metadata = return_metadata
        
        self.aug_cfg = {
            'flip_prob': 0.5,
            'swap_xy_prob': 0.5,
            'sign_flip_prob': 0.0,
            'jitter_std': 0.0,
        }
        if augment_config:
            self.aug_cfg.update(augment_config)
        
        with h5py.File(self.h5_path, 'r') as f:
            total = f['data'].shape[0]
            self.window_size = int(f.attrs.get('window_size', 2048))
            self.sampling_rate = int(f.attrs.get('sampling_rate', 150))
            self.source = str(f.attrs.get('source', 'unknown'))
            self.normalization = str(f.attrs.get('normalization', 'unknown'))
            
            # 读取 model_types 列表(如果有)
            mt_json = f.attrs.get('model_types', None)
            if mt_json:
                if isinstance(mt_json, bytes):
                    mt_json = mt_json.decode('utf-8')
                self.model_types = json.loads(mt_json)
            else:
                self.model_types = []
            
            # 如果有 model_type_idx,读出来
            if 'model_type_idx' in f:
                self.all_model_type_idx = f['model_type_idx'][:]   # (total,)
            else:
                self.all_model_type_idx = None
            
            # model_id 字节串数组(可选)
            if 'model_id' in f:
                self.all_model_id = f['model_id'][:]
            else:
                self.all_model_id = None
        
        # 处理 indices(限制使用的切片子集)
        if indices is None:
            self.indices = np.arange(total)
        else:
            self.indices = np.array(indices, dtype=np.int64)
        
        self.length = len(self.indices)
        self._h5 = None
    
    def _ensure_open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, 'r')
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        self._ensure_open()
        real_idx = int(self.indices[idx])
        
        clip = self._h5['data'][real_idx]
        clip = torch.from_numpy(np.ascontiguousarray(clip)).float()
        
        if self.augment:
            clip = self._apply_augmentation(clip)
        
        if self.return_metadata:
            meta = {
                'idx': real_idx,
                'source': self.source,
                'record_idx': int(self._h5['record_idx'][real_idx]),
                'start_idx': int(self._h5['start_idx'][real_idx]),
            }
            if self.all_model_type_idx is not None:
                mt_idx = int(self.all_model_type_idx[real_idx])
                meta['model_type'] = self.model_types[mt_idx] if mt_idx >= 0 else None
            return clip, meta
        
        return clip
    
    def _apply_augmentation(self, clip):
        cfg = self.aug_cfg
        if torch.rand(1).item() < cfg['flip_prob']:
            clip = torch.flip(clip, dims=[-1])
        if torch.rand(1).item() < cfg['swap_xy_prob']:
            clip = clip[[1, 0, 3, 2], :]
        if cfg['sign_flip_prob'] > 0 and torch.rand(1).item() < cfg['sign_flip_prob']:
            clip = -clip
        if cfg['jitter_std'] > 0:
            clip = clip + cfg['jitter_std'] * torch.randn_like(clip)
        return clip
    
    def __del__(self):
        if self._h5 is not None:
            try:
                self._h5.close()
            except Exception:
                pass
    
    # ============================================================
    # 工具方法
    # ============================================================
    def get_info(self):
        return {
            'h5_path': self.h5_path,
            'n_samples': self.length,
            'window_size': self.window_size,
            'sampling_rate': self.sampling_rate,
            'source': self.source,
            'normalization': self.normalization,
            'augment': self.augment,
            'model_types_available': self.model_types,
        }
    
    def get_model_type_distribution(self):
        """统计当前(过滤后)各 model_type 的切片数"""
        if self.all_model_type_idx is None:
            return {}
        subset = self.all_model_type_idx[self.indices]
        from collections import Counter
        counts = Counter(subset.tolist())
        result = {}
        for type_idx, count in counts.items():
            if type_idx == -1:
                result['<no_type>'] = count
            else:
                result[self.model_types[type_idx]] = count
        return dict(sorted(result.items()))


def filter_indices_by_model_type(h5_path, model_types=None, exclude_types=None,
                                 per_type_limit=None, seed=42):
    """
    根据 model_type 筛选切片索引
    
    h5_path: clean_normalized.h5 或 clean_raw.h5
    model_types: 只保留这些类型(None = 全部)
    exclude_types: 排除这些类型
    per_type_limit: 每个 model_type 最多保留多少切片(None = 不限)
    
    返回: indices ndarray,可传入 CleanAMTDataset(indices=...)
    """
    with h5py.File(h5_path, 'r') as f:
        total = f['data'].shape[0]
        
        if 'model_type_idx' not in f:
            return np.arange(total)
        
        mt_idx_arr = f['model_type_idx'][:]
        mt_list = json.loads(f.attrs['model_types']) if 'model_types' in f.attrs else []
    
    # 构造保留 mask
    keep_mask = np.ones(total, dtype=bool)
    
    if model_types is not None:
        wanted_idx = set()
        for mt in model_types:
            if mt in mt_list:
                wanted_idx.add(mt_list.index(mt))
        new_mask = np.zeros(total, dtype=bool)
        for idx in wanted_idx:
            new_mask |= (mt_idx_arr == idx)
        keep_mask &= new_mask
    
    if exclude_types is not None:
        for mt in exclude_types:
            if mt in mt_list:
                keep_mask &= (mt_idx_arr != mt_list.index(mt))
    
    candidate_indices = np.where(keep_mask)[0]
    
    # 每类限制数量
    if per_type_limit is not None and len(candidate_indices) > 0:
        rng = np.random.default_rng(seed)
        kept = []
        for type_idx in np.unique(mt_idx_arr[candidate_indices]):
            this_type_indices = candidate_indices[mt_idx_arr[candidate_indices] == type_idx]
            if len(this_type_indices) > per_type_limit:
                this_type_indices = rng.choice(this_type_indices, per_type_limit, replace=False)
            kept.append(this_type_indices)
        candidate_indices = np.sort(np.concatenate(kept))
    
    return candidate_indices


def make_per_type_sampler(dataset, samples_per_type=None, num_samples=None):
    """
    按 model_type 等量采样的 sampler
    每个 model_type 在一个 epoch 中被采样的概率相等(不管该类有多少切片)
    
    samples_per_type: dict {type_name: weight},不传则各类等权
    num_samples: 总采样数(默认 = len(dataset))
    """
    if dataset.all_model_type_idx is None:
        # 没有 model_type 信息,退化为均匀采样
        weights = np.ones(len(dataset))
        if num_samples is None:
            num_samples = len(dataset)
        return WeightedRandomSampler(torch.from_numpy(weights), num_samples)
    
    # 当前 dataset 实际使用的切片对应的 model_type
    actual_mt = dataset.all_model_type_idx[dataset.indices]
    
    from collections import Counter
    type_counts = Counter(actual_mt.tolist())
    
    weights = np.zeros(len(dataset), dtype=np.float64)
    for i, mt in enumerate(actual_mt):
        n_in_type = type_counts[int(mt)]
        # 每个 type 的总权重
        if samples_per_type and dataset.model_types and 0 <= mt < len(dataset.model_types):
            type_name = dataset.model_types[int(mt)]
            type_total_weight = samples_per_type.get(type_name, 1.0)
        else:
            type_total_weight = 1.0
        weights[i] = type_total_weight / n_in_type
    
    if num_samples is None:
        num_samples = len(dataset)
    
    return WeightedRandomSampler(torch.from_numpy(weights), num_samples, replacement=True)


# ============================================================
# 多 source 混合 Dataset
# ============================================================
class MultiSourceCleanDataset(Dataset):
    """
    多 source 混合,提供统一的全局索引。
    
    内部维护:
        - 一个全局 idx → (source, local_idx) 的映射
        - 每个 source 一个 CleanAMTDataset
    """
    
    def __init__(self, sources, augment=False, augment_config=None,
                 return_metadata=False, indices_by_source=None):
        """
        sources: dict, {source_name: h5_path}
            e.g. {'real': '.../real/clean_normalized.h5',
                  'syn_amt_A': '.../syn_amt_A/clean_normalized.h5', ...}
        """
        self.source_names = list(sources.keys())
        self.datasets = {}
        self.source_per_idx = []      # 每个全局 idx 对应的 source name
        self.local_idx_per_idx = []   # 每个全局 idx 对应的本地 idx
        
        indices_by_source = indices_by_source or {}
        for src, path in sources.items():
            ds = CleanAMTDataset(path, augment=augment, 
                                 augment_config=augment_config,
                                 return_metadata=return_metadata,
                                 indices=indices_by_source.get(src))
            self.datasets[src] = ds
            self.source_per_idx.extend([src] * len(ds))
            self.local_idx_per_idx.extend(range(len(ds)))
        
        self.length = len(self.source_per_idx)
        self.return_metadata = return_metadata
        
        # 各 source 在全局索引中的范围(便于 sampler)
        self.source_ranges = {}
        offset = 0
        for src in self.source_names:
            n = len(self.datasets[src])
            self.source_ranges[src] = (offset, offset + n)
            offset += n
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        src = self.source_per_idx[idx]
        local = self.local_idx_per_idx[idx]
        return self.datasets[src][local]
    
    def get_info(self):
        return {
            'total': self.length,
            'sources': {src: len(ds) for src, ds in self.datasets.items()},
            'source_ranges': self.source_ranges,
        }


# ============================================================
# 加权采样器
# ============================================================
def make_balanced_sampler(multi_dataset, weight_per_source, num_samples=None,
                          replacement=True):
    """
    构造加权采样器,使不同 source 按指定比例采样。
    
    weight_per_source: dict, {source_name: weight}
        比如 {'real': 1.0, 'syn_amt_A': 1/3, 'syn_amt_B': 1/3, 'syn_mt': 1/3}
        means real 在最终 batch 里占 50%(1 / (1 + 1)),syn_* 三者合占另一半
    
    num_samples: 一个 epoch 取多少样本(默认 = 数据集总长)
    replacement: 是否带替换地采样(默认 True,加权采样必须 True)
    """
    if num_samples is None:
        num_samples = len(multi_dataset)
    
    # 给每个样本一个权重: 来自 source X 的样本 → weight_per_source[X] / size(X)
    # 这样同一 source 内每个样本权重相等,跨 source 总和按比例
    weights = np.zeros(len(multi_dataset), dtype=np.float64)
    
    for src, weight in weight_per_source.items():
        if src not in multi_dataset.source_ranges:
            print(f"⚠️ source '{src}' 不在数据集中,跳过")
            continue
        start, end = multi_dataset.source_ranges[src]
        n = end - start
        if n == 0:
            continue
        weights[start:end] = weight / n
    
    if weights.sum() == 0:
        raise ValueError("所有权重都是 0,请检查 weight_per_source")
    
    weights = torch.from_numpy(weights)
    return WeightedRandomSampler(weights, num_samples=num_samples,
                                 replacement=replacement)


# 不同 rank 不同 seed”的思路，最好在 sampler 里加 generator。
def make_balanced_sampler2(
    multi_dataset,
    weight_per_source,
    num_samples=None,
    replacement=True,
    seed=42,
):
    if num_samples is None:
        num_samples = len(multi_dataset)

    weights = np.zeros(len(multi_dataset), dtype=np.float64)

    for src, weight in weight_per_source.items():
        if src not in multi_dataset.source_ranges:
            print(f"⚠️ source '{src}' 不在数据集中,跳过")
            continue

        start, end = multi_dataset.source_ranges[src]
        n = end - start
        if n == 0:
            continue

        weights[start:end] = weight / n

    if weights.sum() == 0:
        raise ValueError("所有权重都是 0,请检查 weight_per_source")

    weights = torch.from_numpy(weights)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return WeightedRandomSampler(
        weights,
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    )


# ============================================================
# 工具:从 metadata 反归一化
# ============================================================
def denormalize_clip(clip_normalized, stats):
    """
    把归一化的切片反算回物理单位
    clip_normalized: (4, 2048) numpy 或 tensor
    stats: dict with 'E_scale', 'H_scale', 'E_median', 'H_median'
    """
    is_torch = isinstance(clip_normalized, torch.Tensor)
    if is_torch:
        clip_normalized = clip_normalized.cpu().numpy()
    
    E_median = np.array(stats['E_median']).reshape(-1, 1)
    H_median = np.array(stats['H_median']).reshape(-1, 1)
    
    out = np.empty_like(clip_normalized)
    out[:2] = clip_normalized[:2] * stats['E_scale'] + E_median
    out[2:] = clip_normalized[2:] * stats['H_scale'] + H_median
    
    if is_torch:
        out = torch.from_numpy(out)
    return out


def load_metadata(source_dir):
    """加载某个 source 的 metadata.json"""
    with open(Path(source_dir) / 'metadata.json') as f:
        return json.load(f)


def get_stats_for_clip(metadata, record_idx):
    """从 metadata 找到某个切片对应的归一化 stats"""
    rec = metadata['records'][record_idx]
    return rec.get('stats')


# ============================================================
# 测试入口
# ============================================================
if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  测试单 source:    python geoamt_clean_dataset.py <single_h5>")
        print("  测试多 source:    python geoamt_clean_dataset.py multi <root_dir>")
        sys.exit(1)
    
    if sys.argv[1] == 'multi':
        # 测试多 source
        root = Path(sys.argv[2])
        sources = {}
        for src in ['real', 'syn_amt_A', 'syn_amt_B', 'syn_mt']:
            h5 = root / src / 'clean_normalized.h5'
            if h5.exists():
                sources[src] = str(h5)
        
        if not sources:
            print(f"未在 {root} 找到任何 source 的 clean_normalized.h5")
            sys.exit(1)
        
        ds = MultiSourceCleanDataset(sources, augment=True)
        print(f"多 source Dataset 信息: {ds.get_info()}")
        
        # 测试加权采样
        weights = {}
        for src in sources:
            if src == 'real':
                weights[src] = 1.0
            else:
                weights[src] = 1.0 / 3
        
        sampler = make_balanced_sampler(ds, weights, num_samples=200)
        
        # 检查采样分布
        from collections import Counter
        sampled_idx = [i for i in sampler]
        sampled_sources = [ds.source_per_idx[i] for i in sampled_idx]
        cnt = Counter(sampled_sources)
        print(f"\n采样 200 次后各 source 出现次数:")
        for s, c in cnt.most_common():
            print(f"  {s}: {c}")
    
    else:
        # 测试单 source
        ds = CleanAMTDataset(sys.argv[1], augment=True)
        print(f"信息: {ds.get_info()}")
        
        sample = ds[0]
        print(f"\n第一个样本 shape: {sample.shape}, dtype: {sample.dtype}")
        print("通道统计:")
        for ch, name in enumerate(['Ex', 'Ey', 'Hx', 'Hy']):
            v = sample[ch]
            print(f"  {name}: min={v.min():.3f}, max={v.max():.3f}, "
                  f"mean={v.mean():.3f}, std={v.std():.3f}")
        
        # 测试增强随机性
        print("\n增强后多次取同一索引:")
        for i in range(3):
            s = ds[0]
            print(f"  Try {i}: Ex[:5] = {s[0, :5].tolist()}")
