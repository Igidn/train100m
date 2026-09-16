#!/usr/bin/env python3
# Stream HF datasets to jsonl.zst shards for the 110M mix.
# Resumable: state.json tracks per-entry progress; each raw file is downloaded,
# transcribed, and deleted before the next one, so disk use stays bounded.

import io
import gzip
import json
import os
import shutil
import sys
import threading
import time
import zstandard

from pathlib import Path

from huggingface_hub import hf_hub_download, HfApi, HfFileSystem
import pyarrow.parquet as pq

from data_sources import (ENTRIES, CHARS_PER_TOKEN, POOL, MIN_TEXT_CHARS,
                          SHARD_UNCOMPRESSED_BYTES, MIN_FREE_DISK,
                          MISSING_NOTES)

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"
TMP = BASE / "tmp"
STATE_PATH = BASE / "state.json"
MANIFEST_PATH = BASE / "manifest.json"
STOP_PATH = BASE / "STOP"

PHASE_LABEL = {"p1": "phase1", "p2": "phase2"}

api = HfApi()

_last_progress = time.time()


def touch():
    global _last_progress
    _last_progress = time.time()


def _watchdog():
    """Exit the process if no progress for WATCHDOG_SECS; the tmux wrapper
    restarts it and it resumes from state. Catches hung sockets."""
    limit = int(os.environ.get("WATCHDOG_SECS", "1800"))
    while True:
        time.sleep(60)
        if time.time() - _last_progress > limit:
            print(f"WATCHDOG: no progress for {limit}s, exiting for restart",
                  flush=True)
            os._exit(3)


def log(*a):
    touch()
    print(f"[{time.strftime('%H:%M:%S')}]", *a, flush=True)


def load_json(p, default):
    if Path(p).exists():
        return json.loads(Path(p).read_text())
    return default


def save_json(p, obj):
    tmp = f"{p}.tmp"
    Path(tmp).write_text(json.dumps(obj, indent=1))
    os.replace(tmp, p)


def free_bytes():
    st = os.statvfs("/")
    return st.f_bavail * st.f_bsize


def target_chars(tokens_target_m):
    return tokens_target_m * 1e6 * POOL * CHARS_PER_TOKEN


def repo_revision(repo):
    return api.repo_info(repo, repo_type="dataset", revision="main").sha


def entry_files(repo, prefixes, sha, exclude=None):
    """Data files matching prefixes, sorted, as [(path, size)]."""
    hits = []
    for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True,
                                revision=sha):
        p = getattr(f, "path", None)
        if p is None:
            continue
        if p.startswith(".") or "pdf_tarballs" in p:
            continue
        if not p.endswith((".parquet", ".gz", ".zst", ".jsonl", ".json")):
            continue
        if not any(p.startswith(pre) for pre in prefixes):
            continue
        if any(ex in p for ex in (exclude or ())):
            continue
        hits.append((p, getattr(f, "size", 0) or 0))
    hits.sort()
    return hits


def sniff_fmt(path):
    if path.endswith(".parquet"):
        return "parquet"
    if path.endswith(".zst"):
        return "zst"
    if path.endswith(".gz"):
        return "gz"
    if path.endswith(".json"):
        return "jsonarray"
    return "jsonl"


_fs = None


def hffs():
    global _fs
    if _fs is None:
        _fs = HfFileSystem()
    return _fs


def iter_rows(path, fmt):
    """Yield parsed records. Memory-bounded where the format allows."""
    if fmt == "parquet-remote":
        # Read only the row groups we need over ranged HTTP. The generator
        # stops pulling as soon as the caller stops consuming.
        with hffs().open(path, "rb") as fh:
            pf = pq.ParquetFile(fh)
            for batch in pf.iter_batches(batch_size=512):
                yield from batch.to_pylist()
        return
    if fmt == "parquet":
        pf = pq.ParquetFile(path)
        for batch in pf.iter_batches(batch_size=256):
            yield from batch.to_pylist()
    elif fmt == "gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif fmt == "zst":
        with open(path, "rb") as raw:
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue
    elif fmt == "jsonarray":
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = json.load(f)
        yield from data if isinstance(data, list) else []
    else:  # plain jsonl
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.strip():
                    try:
                        yield json.loads(line)
                    except json.JSONDecodeError:
                        continue


def row_text(rec, entry):
    keys = entry["keys"]
    if entry.get("msgs"):
        role_k, value_k = entry["msgs"]
        col = rec.get(keys[0])
        if not isinstance(col, list):
            return None
        parts = []
        for m in col:
            if not isinstance(m, dict):
                continue
            role = m.get(role_k) or ""
            content = m.get(value_k)
            if not isinstance(content, str) or not content.strip():
                continue
            content = content.strip()
            parts.append(f"{role}: {content}" if role else content)
        return "\n\n".join(parts) if parts else None
    vals = []
    for k in keys:
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            vals.append(v.strip())
    return "\n\n".join(vals) if vals else None


class ShardWriter:
    def __init__(self, entry_id, phase):
        self.dir = OUT / entry_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entry_id = entry_id
        self.phase = phase
        self.label = PHASE_LABEL[phase]
        self.seq = len(list(self.dir.glob(f"{entry_id}-{phase}-*.jsonl.zst")))
        self._open()

    def _open(self):
        self.path = self.dir / f"{self.entry_id}-{self.phase}-{self.seq:05d}.jsonl.zst"
        self.raw = open(self.path, "wb")
        self.cctx = zstandard.ZstdCompressor(level=6)
        self.zw = self.cctx.stream_writer(self.raw, closefd=False)
        self.nbytes = 0

    def _ensure_open(self):
        if self.zw is None:
            self._open()

    def write(self, rec):
        self._ensure_open()
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        b = line.encode("utf-8")
        self.zw.write(b)
        self.nbytes += len(b)
        if self.nbytes >= SHARD_UNCOMPRESSED_BYTES:
            self.rotate()

    def rotate(self):
        self.close()
        self.seq += 1

    def close(self):
        if getattr(self, "zw", None):
            try:
                self.zw.flush(zstandard.FLUSH_FRAME)
            except Exception:
                pass
            try:
                self.zw.close()
            except Exception:
                pass
            self.zw = None
        if getattr(self, "raw", None):
            self.raw.close()
            self.raw = None


def validate_last_shards(eid):
    """A crashed run can leave a torn zstd frame in the newest shard; drop it."""
    d = OUT / eid
    if not d.exists():
        return
    newest = {}
    for f in d.glob("*.jsonl.zst"):
        newest[f.name.rsplit("-", 1)[0]] = f
    for base, f in sorted(newest.items()):
        try:
            with open(f, "rb") as fh:
                r = zstandard.ZstdDecompressor().stream_reader(fh)
                while r.read(1 << 20):
                    pass
                r.close()
        except Exception as ex:
            log(f"  dropping torn shard {f.name}: {type(ex).__name__}")
            f.unlink(missing_ok=True)


def with_retries(fn, attempts=4, wait=10, what=""):
    for i in range(attempts):
        try:
            touch()
            return fn()
        except Exception as ex:
            log(f"  {what} attempt {i+1} failed: {ex!r}")
            time.sleep(wait * (i + 1))
    return None


def fetch_file(repo, path, sha):
    dest_dir = TMP / "dl"
    dest_dir.mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        try:
            touch()
            return hf_hub_download(repo_id=repo, filename=path,
                                   repo_type="dataset", revision=sha,
                                   local_dir=dest_dir)
        except Exception as ex:
            log(f"  download attempt {attempt+1} failed: {ex!r}")
            time.sleep(15 * (attempt + 1))
    return None


def main():
    global TMP, STATE_PATH, MANIFEST_PATH
    only = None
    suffix = ""
    args = sys.argv[1:]
    if "--only" in args:
        only = set(args[args.index("--only") + 1].split(","))
        log("only:", only)
    if "--state-suffix" in args:
        suffix = args[args.index("--state-suffix") + 1]
        log("state suffix:", suffix)
    TMP = BASE / f"tmp{suffix}"
    STATE_PATH = BASE / f"state{suffix}.json"
    MANIFEST_PATH = BASE / f"manifest{suffix}.json"
    threading.Thread(target=_watchdog, daemon=True).start()

    for d in (OUT, TMP):
        d.mkdir(parents=True, exist_ok=True)
    dl = TMP / "dl"
    if dl.exists():
        shutil.rmtree(dl, ignore_errors=True)

    state = load_json(STATE_PATH, {"entries": {}})
    manifest = load_json(MANIFEST_PATH, {"sources": {}, "notes": {}})

    stopped = "completed"
    for entry in ENTRIES:
        if only and entry["id"] not in only:
            continue
        eid = entry["id"]
        st = state["entries"].setdefault(eid, {"done": [], "chars": {}, "rows": 0})
        for ph, _mt in entry["phases"]:
            st["chars"].setdefault(ph, 0)
        if st.get("status") == "done":
            continue

        log(f"=== {eid} ({entry['repo']} {entry['prefixes']})")
        sha = with_retries(lambda: repo_revision(entry["repo"]),
                           what="repo info")
        if sha is None:
            st["status"] = "unavailable"
            save_json(STATE_PATH, state)
            continue

        res = with_retries(lambda: entry_files(entry["repo"], entry["prefixes"],
                                               sha, entry.get("exclude")),
                           what="listing")
        if res is None:
            st["status"] = "unavailable"
            save_json(STATE_PATH, state)
            continue
        files = res

        caps = {ph: target_chars(mt) for ph, mt in entry["phases"]}
        remaining = {ph: caps[ph] - st["chars"].get(ph, 0) for ph in caps}
        remaining = {ph: c for ph, c in remaining.items() if c > 0}
        if not remaining:
            st["status"] = "done"
            save_json(STATE_PATH, state)
            log("  already satisfied")
            continue

        validate_last_shards(eid)
        files = [(p, s) for p, s in files if p not in set(st["done"])]
        if entry.get("max_files"):
            files = files[: entry["max_files"]]
        if not files:
            st["status"] = "exhausted"
            save_json(STATE_PATH, state)
            log("  no more files; shortfall:", {k: round(v / 1e6) for k, v in remaining.items()})
            continue

        total_files = len(files)
        writer = None
        try:
            for i, (path, size) in enumerate(files):
                if STOP_PATH.exists():
                    stopped = "stopped-by-STOP-file"
                    break
                if free_bytes() < MIN_FREE_DISK:
                    stopped = "low-disk"
                    log("  DISK LOW, stopping")
                    break
                if not remaining:
                    break

                fmt = entry["fmt"] if entry["fmt"] != "auto" else sniff_fmt(path)
                if fmt == "parquet-remote":
                    local = f"datasets/{entry['repo']}/{path}"
                else:
                    local = fetch_file(entry["repo"], path, sha)
                    if local is None:
                        st.setdefault("bad_files", []).append(path)
                        save_json(STATE_PATH, state)
                        continue
                try:
                    file_gain = {}
                    nrows = 0
                    for rec in iter_rows(local, fmt):
                        if not remaining:
                            break
                        nrows += 1
                        if nrows % 50000 == 0:
                            touch()
                        t = row_text(rec, entry)
                        if not t or len(t) < MIN_TEXT_CHARS:
                            continue
                        ph = next(iter(remaining))
                        if writer is None or writer.phase != ph:
                            if writer:
                                writer.close()
                            writer = ShardWriter(eid, ph)
                        writer.write({"text": t, "source": eid,
                                      "domain": entry["domain"],
                                      "phase": PHASE_LABEL[ph]})
                        st["chars"][ph] += len(t)
                        file_gain[ph] = file_gain.get(ph, 0) + len(t)
                        nrows += 1
                        if st["chars"][ph] >= caps[ph]:
                            remaining.pop(ph, None)
                    st["rows"] = st.get("rows", 0) + nrows
                    gain_s = " ".join(f"{k} +{v/1e6:.0f}Mch"
                                      for k, v in file_gain.items())
                    prog = " ".join(
                        f"{k} {min(1.0, st['chars'][k]/caps[k])*100:.0f}%"
                        for k in caps)
                    log(f"  [{i+1}/{total_files}] {path.rsplit('/',1)[-1]} "
                        f"({size/1e6:.0f} MB) {gain_s} | {prog} rows={nrows}")
                finally:
                    st["done"].append(path)
                    save_json(STATE_PATH, state)
                    if fmt != "parquet-remote":
                        try:
                            os.remove(local)
                        except OSError:
                            pass
                    if writer:
                        writer.rotate()  # clean frame boundary per raw file
        finally:
            if writer:
                writer.close()
                writer = None

        if not remaining:
            st["status"] = "done"
        elif st.get("done"):
            st["status"] = "partial"
        save_json(STATE_PATH, state)

        tot_chars = sum(st["chars"].get(ph, 0) for ph, _ in entry["phases"])
        manifest["sources"][eid] = {
            "repo": entry["repo"], "revision": sha,
            "prefixes": entry["prefixes"], "text_keys": entry["keys"],
            "domain": entry["domain"],
            "targets": {ph: mt for ph, mt in entry["phases"]},
            "chars": st["chars"],
            "est_tokens_m": round(tot_chars / CHARS_PER_TOKEN / 1e6),
            "rows": st.get("rows", 0),
            "files_used": len(st.get("done", [])),
            "status": st.get("status"),
        }
        save_json(MANIFEST_PATH, manifest)

    manifest["status"] = stopped
    manifest["pool"] = POOL
    manifest["chars_per_token_assumed"] = CHARS_PER_TOKEN
    manifest["missing_notes"] = MISSING_NOTES
    save_json(MANIFEST_PATH, manifest)

    tot_est = 0
    for eid, m in manifest["sources"].items():
        tot_est += m["est_tokens_m"]
        log(f"{eid:38s} {m['est_tokens_m']:7d}M est tokens  {m['status']}")
    log(f"TOTAL est tokens across sources: {tot_est}M "
        f"(spec targets sum to 2000M; fetched pools are {POOL}x)")
    log("stopped:", stopped)


if __name__ == "__main__":
    main()
