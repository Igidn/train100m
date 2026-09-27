"""Packed SFT data layer for sft-tok-v1 — docs, loss masks, replay rows.

This is the *code* definition of the format (the SFT_NOTES.md prose is a
summary; the shards were produced by sft/tokenize_sft.py):

  <bucket>/<split>/<tag>.bin          uint16 LE tokens, docs concatenated,
                                      NO eos between docs
  <bucket>/<split>/<tag>.mask.bin     uint8 0/1 loss mask, parallel to .bin
  <bucket>/<split>/<tag>.offsets.npy  uint64 doc START offsets, offs[0]=0
  <bucket>/<split>/<tag>.meta.jsonl   {"tokens", "train"} per doc
  manifest.json                       vocab / special ids / per-bucket stats

Doc i of a shard spans [offs[i], offs[i+1]) and the last doc ends at EOF.
Everything the model may train on is inside the mask: assistant content,
reasoning, tool-call args and the trailing <|im_end|> (id 2, the generation
stop); headers, user/system/tool text and inserted eos separators are 0.

Packing. Rows are T+1 tokens long (T = training seq_len) so the trainer can
split row[:-1] -> inputs, row[1:] -> targets, mask[1:] -> loss mask and get
exactly T prediction positions per row without a second shift. Docs are
greedily concatenated with a single eos=0 token between them (the tok-mix-v1
packing separator, kept at mask 0 so the model is not taught to emit it);
the epoch's leftover tail is padded with eos / mask 0.

Determinism. A plan is a pure function of (doc pool, seq_len, seed): the doc
order and the replay-row merge are drawn from one numpy Generator seeded with
the caller's integer. Resuming from a step therefore reproduces the exact
same rows on every rank without any cursor state to save.

Replay. tok-mix-v1 p2 shards enter a plan as whole rows of T+1 raw pretrain
tokens with mask=1 everywhere, at REPLAY_FRAC of the rows. Interleaving whole
pretrain batches is what "mix ~1% into every phase" means in practice, and it
leaves the SFT mask semantics untouched. The replay row is also the only
place where predicting eos is trained, exactly as in the pretrain mix.

CPT. A continued-pretrain segment is a plan over SFT docs whose mask is
overridden to 1 on the doc tokens (only the inserted eos separators stay 0);
used to warm up the long-context phase before the masked pass.
"""

import glob
import os

import numpy as np

EOS_ID = 0
EMPTY = np.empty(0, dtype=np.int64)

# piece kinds inside a plan row (first int64 column)
K_DOC = 0    # SFT doc: mask read from .mask.bin
K_RAW = -1   # raw pretrain chunk: mask 1 everywhere (replay)


class DocPool:
    """All docs of one <bucket>/<split>, addressed by a global index.

    Doc i -> (shard_of[i], index_in[i]); doc(i) returns (tokens, mask) views
    straight into the memmapped shard. Shard memmaps are opened lazily and
    per process.
    """

    def __init__(self, root, bucket, split):
        d = os.path.join(root, bucket, split)
        self.paths = sorted(p for p in glob.glob(os.path.join(d, "*.bin"))
                            if not p.endswith(".mask.bin"))
        if not self.paths:
            raise FileNotFoundError(f"no .bin shards under {d}")
        shard_of, index_in, lengths = [], [], []
        for s, p in enumerate(self.paths):
            offs = np.load(p[:-4] + ".offsets.npy").astype(np.int64)
            total = os.path.getsize(p) // 2
            lens = np.diff(np.append(offs, total))
            if len(lens) and (lens <= 0).any():
                raise ValueError(f"bad offsets in {p}")
            shard_of.append(np.full(lens.size, s, dtype=np.int64))
            index_in.append(np.arange(lens.size, dtype=np.int64))
            lengths.append(lens)
        self.shard_of = np.concatenate(shard_of) if shard_of else EMPTY
        self.index_in = np.concatenate(index_in) if index_in else EMPTY
        self.lengths = np.concatenate(lengths) if lengths else EMPTY
        self.n = int(self.lengths.size)
        self.tokens = int(self.lengths.sum())
        self._mm = {}
        self._offs = {}

    def offsets(self, s):
        offs = self._offs.get(s)
        if offs is None:
            offs = np.load(self.paths[s][:-4] + ".offsets.npy").astype(np.int64)
            self._offs[s] = offs
        return offs

    def _maps(self, s):
        mm = self._mm.get(s)
        if mm is None:
            base = self.paths[s][:-4]
            mm = (np.memmap(base + ".bin", dtype=np.uint16, mode="r"),
                  np.memmap(base + ".mask.bin", dtype=np.uint8, mode="r"))
            self._mm[s] = mm
        return mm

    def doc(self, i):
        """(tokens, mask) numpy views for doc i."""
        s = int(self.shard_of[i])
        k = int(self.index_in[i])
        toks, mask = self._maps(s)
        a = int(self.offsets(s)[k])
        b = a + int(self.lengths[i])
        return toks[a:b], mask[a:b]


class ChunkPool:
    """Fixed row_len-token chunks cut from raw pretrain shards (tok-mix p2)."""

    def __init__(self, paths, row_len):
        self.paths = list(paths)
        self.row_len = int(row_len)
        self.n_chunks = [os.path.getsize(p) // 2 // self.row_len for p in self.paths]
        self.chunk_shard = np.concatenate(
            [np.full(n, s, dtype=np.int64) for s, n in enumerate(self.n_chunks)]
        ) if self.n_chunks else EMPTY
        self.chunk_idx = np.concatenate(
            [np.arange(n, dtype=np.int64) for n in self.n_chunks]
        ) if self.n_chunks else EMPTY
        self.n = int(self.chunk_idx.size)
        self.tokens = int(self.row_len) * self.n
        self._mm = {}

    def chunk(self, shard, idx):
        mm = self._mm.get(shard)
        if mm is None:
            mm = np.memmap(self.paths[shard], dtype=np.uint16, mode="r")
            self._mm[shard] = mm
        a = idx * self.row_len
        return mm[a:a + self.row_len]


class SFTPhaseData:
    """Doc pools + replay pool for one phase, and the plans built over them.

    build_plan() returns a list of rows; a row is an (n, 4) int64 array of
    pieces [kind, pool, index, unused]:
      kind  0 -> doc: self.all_pools[pool].doc(index) (mask from .mask.bin)
      kind -1 -> replay chunk: self.replay.chunk(pool=shard, index=chunk)
    materialize(row, cpt=False) turns a row into (tokens, mask) numpy arrays
    of exactly row_len = seq_len + 1 tokens.
    """

    def __init__(self, root, buckets, seq_len, replay=None):
        self.root = root
        self.seq_len = int(seq_len)
        self.row_len = self.seq_len + 1
        self.replay = replay
        self.buckets = list(buckets)
        self.train_pools = [DocPool(root, b, "train") for b in self.buckets]
        self.val_pools = [DocPool(root, b, "val") for b in self.buckets]
        # piece 'pool' column indexes all_pools: train pools first, val pools after
        self.all_pools = self.train_pools + self.val_pools
        shard_of, index_in, pool_of, pool_index, lengths = [], [], [], [], []
        for p, pool in enumerate(self.train_pools):
            shard_of.append(pool.shard_of)
            index_in.append(pool.index_in)
            lengths.append(pool.lengths)
            pool_of.append(np.full(pool.n, p, dtype=np.int64))
            pool_index.append(np.arange(pool.n, dtype=np.int64))
        self.shard_of = np.concatenate(shard_of) if shard_of else EMPTY
        self.index_in = np.concatenate(index_in) if index_in else EMPTY
        self.pool_of = np.concatenate(pool_of) if pool_of else EMPTY
        self.pool_index = np.concatenate(pool_index) if pool_index else EMPTY
        self.lengths = np.concatenate(lengths) if lengths else EMPTY
        self.n_docs = int(self.lengths.size)
        self.tokens = int(self.lengths.sum())
        self.val_docs = sum(p.n for p in self.val_pools)
        # docs longer than the row cannot be packed; in production the bucket
        # bounds (<=seq) make this zero — otherwise it is data silently lost
        self.n_oversize = int((self.lengths > self.row_len).sum())

    # ---------------------------------------------------------------- plans

    def build_plan(self, seed, replay_frac=0.0, max_tokens=None):
        """One epoch's training rows, deterministic in `seed`.

        replay_frac is the target share of *rows* drawn from the replay pool
        (all rows are row_len tokens, so row share == token share).
        max_tokens caps the doc stream (CPT warmup) and truncates there.
        """
        rows, cur, cur_len, used = [], [], 0, 0
        order = np.random.default_rng(seed).permutation(self.n_docs)
        for d in order:
            L = int(self.lengths[d])
            if L > self.row_len:
                continue  # cannot happen: docs are bucketed by their own length
            if cur_len and cur_len + 1 + L > self.row_len:
                rows.append(np.asarray(cur, dtype=np.int64).reshape(-1, 4))
                cur, cur_len = [], 0
            cur.append((K_DOC, int(self.pool_of[d]), int(self.pool_index[d]), 0))
            cur_len += L + (1 if cur_len else 0)
            used += L
            if max_tokens is not None and used >= max_tokens:
                break
        if cur_len:
            rows.append(np.asarray(cur, dtype=np.int64).reshape(-1, 4))
        rows = [r for r in rows if r.size]  # drop empties from the max_tokens cut

        if self.replay is not None and replay_frac > 0.0 and rows:
            rng = np.random.default_rng(seed + 1)
            k = min(int(round(len(rows) * replay_frac / (1.0 - replay_frac))),
                    self.replay.n)
            for j in rng.choice(self.replay.n, size=k, replace=False):
                rows.append(np.array([[K_RAW, int(self.replay.chunk_shard[j]),
                                       int(self.replay.chunk_idx[j]), 0]],
                                     dtype=np.int64))
            rows = [rows[int(i)] for i in rng.permutation(len(rows))]
        return rows

    def build_val_plan(self):
        """Sequential (unshuffled) packing of all val docs, no replay."""
        rows, cur, cur_len = [], [], 0
        base = len(self.train_pools)
        for v, pool in enumerate(self.val_pools):
            for d in range(pool.n):
                L = int(pool.lengths[d])
                if L > self.row_len:
                    continue  # same guard as build_plan; see n_oversize
                if cur_len and cur_len + 1 + L > self.row_len:
                    rows.append(np.asarray(cur, dtype=np.int64).reshape(-1, 4))
                    cur, cur_len = [], 0
                cur.append((K_DOC, base + v, d, 0))
                cur_len += L + (1 if cur_len else 0)
            if cur_len:
                rows.append(np.asarray(cur, dtype=np.int64).reshape(-1, 4))
                cur, cur_len = [], 0
        return rows

    # ---------------------------------------------------------- materialize

    def materialize(self, row, cpt=False):
        """Row -> (tokens uint16[row_len], mask uint8[row_len])."""
        T = self.row_len
        toks = np.zeros(T, dtype=np.uint16)
        mask = np.zeros(T, dtype=np.uint8)
        pos = 0
        for kind, a, b, _ in row.tolist():
            if pos:
                pos += 1  # eos separator, stays mask 0
            if kind == K_RAW:
                chunk = self.replay.chunk(a, b)
                n = len(chunk)
                toks[pos:pos + n] = chunk
                mask[pos:pos + n] = 1
            else:
                t, m = self.all_pools[a].doc(b)
                n = int(self.all_pools[a].lengths[b])
                toks[pos:pos + n] = t
                mask[pos:pos + n] = 1 if cpt else m
            pos += n
        return toks, mask
