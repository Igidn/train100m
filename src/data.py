"""Loader for pre-packed uint16 token shards (tok-mix-v1).

Shard naming inside train/: <source>-p1-NNN.bin for phase 1 (broad,
1.5B tokens) and <source>-p2-NNN.bin for phase 2 (anneal, 500M tokens) —
both live in the same directory, so the phase selector is part of the
glob, not the path. Each shard is a flat token stream with eos_id=0
separating documents; we stream and cut fixed-length training chunks,
sampling shards uniformly over tokens so the mix stays interleaved
instead of concatenated by source.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset

EOS_ID = 0


class ShardSet:
    """All shards of one phase, opened lazily per worker process."""

    def __init__(self, paths, seq_len):
        self.paths = paths
        self.seq_len = seq_len
        self.n_chunks = []  # full chunks per shard (partial tail dropped)
        for p in paths:
            n_tokens = os.path.getsize(p) // 2
            self.n_chunks.append(n_tokens // seq_len)
        self.chunks = list(zip(
            np.repeat(np.arange(len(paths)), self.n_chunks),
            np.concatenate([np.arange(n) for n in self.n_chunks])
            if len(self.n_chunks) else np.array([], dtype=np.int64),
        ))
        self._maps = {}

    def get(self, shard, chunk):
        mm = self._maps.get(shard)
        if mm is None:
            mm = np.memmap(self.paths[shard], dtype=np.uint16, mode="r")
            self._maps[shard] = mm
        s = self.seq_len
        return np.asarray(mm[chunk * s:(chunk + 1) * s], dtype=np.int64)


class PackedDataset(Dataset):
    """Fixed-length chunks cut from all shards of one phase, shuffled per epoch.

    Each phase is streamed exactly once (1 epoch = its full token budget),
    so no resampling loop is needed — steps per phase fall out of the
    chunk count.
    """

    def __init__(self, data_dir, phase, seq_len=2048, seed=1234):
        pattern = os.path.join(data_dir, "train", f"*-p{phase}-*.bin")
        paths = sorted(glob.glob(pattern))
        if not paths:
            raise FileNotFoundError(f"no shards match {pattern} — check TOK_DATA_DIR")
        self.set = ShardSet(paths, seq_len)
        self.seq_len = seq_len
        self.seed = seed
        self.epoch = 0
        self.reorder()

    def reorder(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.order = rng.permutation(len(self.set.chunks))

    def __len__(self):
        return len(self.order)

    def __getitem__(self, i):
        shard, chunk = self.set.chunks[self.order[i]]
        toks = self.set.get(shard, chunk)
        return torch.from_numpy(toks)

    @property
    def tokens(self):
        return len(self.order) * self.seq_len


class ValDataset(Dataset):
    """Sequential chunks from val/val-000.bin, capped at n_chunks."""

    def __init__(self, data_dir, seq_len=2048, n_chunks=256):
        path = os.path.join(data_dir, "val", "val-000.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"missing {path}")
        self.seq_len = seq_len
        self.n = min(n_chunks, os.path.getsize(path) // 2 // seq_len)
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.seq_len
        return torch.from_numpy(np.asarray(self.mm[i * s:(i + 1) * s], dtype=np.int64))
