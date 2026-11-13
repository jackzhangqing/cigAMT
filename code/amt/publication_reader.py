"""Fixed-epoch publication core adapter for the historical benchmark engine.

Per-component normalization, task kernels and schedules use the released core.
The historical global row ordering and exact stratified selection are retained.
Fingerprint identities identify this replay adapter rather than offline headers.
"""
from pathlib import Path
from bisect import bisect_right
import sys, json, hashlib, os
import numpy as np
from amtbench.data import OfflineH5Dataset, _Shard
sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset_core"))
from denoising_dataset.materialize import SampleFactory

class PortableDataset(OfflineH5Dataset):
    def __init__(self, root, *args, base_module=None, **kwargs):
        self.factory = None
        self.factory_pid = None
        super().__init__(root, *args, **kwargs)

    def _discover(self):
        m = json.loads((self.root / "manifest.json").read_text(encoding="utf-8-sig"))
        self.dataset_id = m["dataset_id"]
        self.sequence_length, self.sampling_rate_hz = 2048, 150.0
        ix = np.load(self.root / "metadata/clean_index.npz", allow_pickle=False)
        ids = np.flatnonzero(ix["split_id"] == ("train", "validation", "test").index(self.split))
        self.source_ids = ix["source_id"][ids].astype(np.int64)
        self.snr_values = np.asarray(m["config"]["noise"]["snr_db"], dtype=np.float32)
        noise = self.task in ("noise_localization", "noise_classification", "denoising", "frequency_denoising")
        groups = [("noise", s, len(ids)*len(self.snr_values)) for s in (("A","C") if self.scheme == "AC" else (self.scheme,))] if noise else [("clean", "none", len(ids))]
        if noise and self.include_controls: groups.append(("control", "control", len(ids)))
        offset = 0
        for kind, scheme, count in groups:
            rel = f"replay/{kind}/{scheme}/{self.split}"
            self._shards.append(_Shard(self.root / rel, rel, count, kind, scheme, "epoch_0", 0, offset))
            offset += count

    def _selection_metadata(self, shard, blocks):
        rows = np.concatenate([np.arange(a,b,dtype=np.int64) for a,b in blocks])
        noisy = shard.kind == "noise"
        source = self.source_ids[rows//len(self.snr_values) if noisy else rows]
        snr = self.snr_values[rows%len(self.snr_values)] if noisy else np.full(len(rows), np.nan)
        return source, snr, np.full(len(rows), int(shard.kind == "control"),dtype=np.uint8)

    def _get_factory(self):
        if self.factory is None or self.factory_pid != os.getpid():
            self.close()
            self.factory = SampleFactory(self.root)
            self.factory_pid = os.getpid()
        return self.factory

    def __getitem__(self, index): return self.__getitems__([index])[0]

    def __getitems__(self, indices):
        groups = {}
        for out_index,index in enumerate(indices):
            file_id,row = self._locate(index)
            groups.setdefault(file_id,[]).append((out_index,row))
        out = [None]*len(indices)
        factory = self._get_factory()
        for file_id, rows in groups.items():
            shard = self._shards[file_id]
            positions = np.asarray([r for _,r in rows],dtype=np.int64)
            if shard.kind == "noise": values = factory.noise_batch(self.split,shard.scheme,positions,epoch=0)
            elif shard.kind == "control": values = factory.control_batch(self.split,positions)
            else: values = factory.clean_batch(self.split,self.task,positions,epoch=0)
            for j,(output,row) in enumerate(rows):
                out[output] = self._sample(file_id,row,{k:v[j] for k,v in values.items() if isinstance(v,np.ndarray)})
        return out

    def get_metadata(self,index):
        sample = self[index]
        return {k:v for k,v in sample.items() if k not in ("input","target","target_low","target_high","noise_mask","observed_mask","loss_mask")}

    def describe(self):
        out=super().describe();out["reader"]="publication_core_epoch0_replay";return out

    def fingerprints(self):
        out=super().fingerprints();out["scope"]="publication core, fixed epoch-0 replay and selected global row indices";return out

    def close(self):
        super().close()
        if self.factory is not None: self.factory.close()
        self.factory=None;self.factory_pid=None

    def __getstate__(self):
        state=super().__getstate__();state["factory"]=None;state["factory_pid"]=None;return state
