 ```
   tok-mix-v1/
   ├── manifest.json                       # global stats + per-source mix config
   ├── tokenizer.json                      # tokenizer (3.4 MB, sha256 62740a3e…)
   ├── train/
   │   ├── <source>-p2-000.bin             # packed tokens, raw uint16
   │   ├── <source>-p2-000.offsets.npy     # uint64 doc-start offsets
   │   ├── …                               # 78 bin + 78 offsets = 156 files
   └── val/
       └── val-000.bin / val-000.offsets.npy
 ```

 Shard names follow <source>-<pack>-NNN.bin — e.g. agentic-glaive-p2-000.bin, anneal-finemath-4plus-p2-017.bin.

 File formats

 .bin — flat token stream, dtype=uint16, little-endian, no header. eos_id = 0 separates documents. Memory-mappable:

 ```python
   tokens = np.memmap(path, dtype=np.uint16, mode="r")
 ```

 .offsets.npy — one uint64 per document, token index where each doc starts:

 ```
   shard: train/agentic-glaive-p2-000.bin
   offsets dtype: uint64, len: 8867          → 8867 docs in this shard
   first 5 offsets: [0, 753, 1380, 1492, 2873]
   last offset: 5008784
   bin size: 10018838 bytes → 5 009 419 tokens
 ```

 So doc i = tokens[offsets[i] : offsets[i+1]].

 tokenizer.json — vocab 49 154, eos_id 0.

 manifest.json highlights

 ```json
   {
     "val":    { "tokens": 2085817, "docs": 2039, "eos_id": 0, "vocab": 49154 },
     "totals": {
       "train_p1_tokens": 1506844429,
       "train_p2_tokens": 500623337,
       "val_tokens": 2085817,
       "train_docs": 2046495,
       "packing": "load-time; training seqlen target 2048"
     },
     "sources": {
       "agentic-commitpackft": {
         "targets":   { "p2": 8 },
         "achieved":  { "val": 8204, "p2": 8000051 },
         "contam_skipped": 19,
         "docs": 31914,
         "seed": "24650e26ca530b0e"
       },
       … 50+ sources
     }
   }
 ```

 The full source list covers agentic/code, anneal (math/books/web), micro, papers, qa, synth, web, wiki families — the p1 vs p2 split is the packing priority/mix weight per source.


