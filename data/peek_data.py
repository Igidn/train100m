# Peek at each (repo, prefix) used by the plan: file counts, sizes, and the
# parquet schema (footer only) or first gz lines, so extract keys can be
# verified before the big run. Cheap on bandwidth.

import json
import zlib

from huggingface_hub import HfApi, HfFileSystem
import pyarrow.parquet as pq

from data_sources import ENTRIES

api = HfApi()
fs = HfFileSystem()


def gz_sample(repo, path):
    try:
        with fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            raw = fh.read(400_000)
        d = zlib.decompressobj(16 + zlib.MAX_WBITS)
        out = d.decompress(raw)
        lines = [ln for ln in out.split(b"\n") if ln.strip()]
        rec = json.loads(lines[0])
        for key, val in rec.items():
            v = str(val)[:100].replace("\n", "\\n")
            print(f"    [{key}] {type(val).__name__}: {v}")
    except Exception as ex:
        print(f"  GZ PEEK FAIL: {ex!r}")


def pq_schema(repo, path):
    try:
        with fs.open(f"datasets/{repo}/{path}", "rb") as fh:
            pf = pq.ParquetFile(fh)
            names = pf.schema_arrow.names
        print(f"  parquet columns: {names}")
    except Exception as ex:
        print(f"  PQ PEEK FAIL: {ex!r}")


seen = set()
for e in ENTRIES:
    k = (e["repo"], e["prefix"])
    if k in seen:
        continue
    seen.add(k)
    repo, prefix = k
    print("=" * 70)
    print(f"repo={repo} prefix='{prefix}' entries={[x['id'] for x in ENTRIES if x['repo']==repo and x['prefix']==prefix]}")
    try:
        files = []
        total = 0
        for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision="main"):
            if not hasattr(f, "size") or f.size is None:
                continue
            p = f.path
            if not (p.endswith(".parquet") or p.endswith(".gz")):
                continue
            if p.startswith("."):
                continue
            if prefix and not p.startswith(prefix):
                continue
            files.append((p, f.size))
            total += f.size
    except Exception as ex:
        print(f"  LIST FAIL: {ex!r}")
        continue
    files.sort()
    print(f"  {len(files)} files, {total/1e9:.2f} GB total")
    if files:
        print(f"  first: {files[0][0]} ({files[0][1]/1e6:.1f} MB)  last: {files[-1][0]} ({files[-1][1]/1e6:.1f} MB)")
    else:
        print("  NO FILES MATCH PREFIX; sample of repo paths:")
        try:
            n = 0
            for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, revision="main"):
                if hasattr(f, "path"):
                    print(f"    {f.path}")
                    n += 1
                if n >= 25:
                    break
        except Exception as ex:
            print(f"    sample fail: {ex!r}")
        continue

    first = files[0][0]
    if first.endswith(".parquet"):
        pq_schema(repo, first)
    else:
        gz_sample(repo, first)
