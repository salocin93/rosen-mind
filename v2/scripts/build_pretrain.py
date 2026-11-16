"""
RosenMind v2 pretraining data.

  tokenize : each source -> data/bin/pool_<source>.bin, a packed uint16 stream of [BOS] doc [EOS] [BOS] doc [EOS] ...
             cut into 512-token rows (no padding at all). CPU only; safe to run while the GPU trains.
  mix      : picks rows from the pools for a token budget -> data/bin/order.npy (+ order.json)
             main phase (85%): fineweb-edu 70 / cosmopedia-v2 15 / finemath 10 / code 5, shuffled
             quality phase (15%, trained last while the LR decays): fineweb-edu score>=4 50 / cosmopedia 20 / finemath 20 / code 10
             The last VAL_ROWS rows of every pool are held out as validation data.

  RAYON_NUM_THREADS=6 nice -n 19 python v2/scripts/build_pretrain.py tokenize [--sources fineweb,fineweb_hq]   # default: all
  python v2/scripts/build_pretrain.py mix --tokens 1.7e9          # or --hours 24 (uses data/bin/gpu_bench.json)
"""
import argparse
import glob
import json
import os
import sys
import time
from itertools import chain

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq

from common import BOS, EOS, DATA_BIN, DATA_RAW, PRETRAIN_SEQ, V1_ROOT, load_tokenizer

VAL_ROWS = 256
# caps are generous upper bounds for a ~3B-token run (48 h of the 152M model); tokenizing stops at the cap
SOURCES = {
    'fineweb':    dict(cap=2.30e9),
    'fineweb_hq': dict(cap=0.20e9),
    'cosmo':      dict(cap=0.40e9),
    'math':       dict(cap=0.30e9),
    'code':       dict(cap=0.15e9),
}
MAIN_MIX = {'fineweb': 0.70, 'cosmo': 0.15, 'math': 0.10, 'code': 0.05}
HQ_MIX = {'fineweb_hq': 0.50, 'cosmo': 0.20, 'math': 0.20, 'code': 0.10}
HQ_FRAC = 0.15


def fineweb_files():
    return [os.path.join(V1_ROOT, 'dataset/raw_en/sample/10BT/000_00000.parquet')] + \
        sorted(glob.glob(os.path.join(DATA_RAW, 'HuggingFaceFW__fineweb-edu/sample/10BT/*.parquet')))


def iter_docs(files, cols, batch=1000):
    for path in files:
        for b in pq.ParquetFile(path).iter_batches(batch_size=batch, columns=cols):
            yield b.to_pydict()


def doc_streams(only):
    """Yields (source, texts) batches. fineweb is split by its quality score into fineweb / fineweb_hq."""
    for d in (iter_docs(fineweb_files(), ['text', 'int_score']) if only & {'fineweb', 'fineweb_hq'} else []):
        hq = [t for t, s in zip(d['text'], d['int_score']) if s >= 4]
        lo = [t for t, s in zip(d['text'], d['int_score']) if s < 4]
        yield 'fineweb_hq', hq
        yield 'fineweb', lo
    for d in (iter_docs(sorted(glob.glob(os.path.join(DATA_RAW, 'HuggingFaceTB__smollm-corpus/cosmopedia-v2/*.parquet'))), ['text']) if 'cosmo' in only else []):
        yield 'cosmo', d['text']
    for d in (iter_docs(sorted(glob.glob(os.path.join(DATA_RAW, 'HuggingFaceTB__finemath/finemath-4plus/*.parquet'))), ['text']) if 'math' in only else []):
        yield 'math', d['text']
    for d in (iter_docs(sorted(glob.glob(os.path.join(DATA_RAW, 'Nan-Do__code-search-net-python/data/*.parquet'))), ['original_string']) if 'code' in only else []):
        yield 'code', d['original_string']


def tokenize(only):
    """Rebuilds the pools in `only`; the others (and their pools.json entries) are kept as they are."""
    tok = load_tokenizer()
    os.makedirs(DATA_BIN, exist_ok=True)
    files = {s: open(os.path.join(DATA_BIN, f'pool_{s}.bin.tmp'), 'wb') for s in only}
    count = {s: 0 for s in only}
    docs = {s: 0 for s in only}
    t0 = time.time(); last = t0
    for src, texts in doc_streams(only):
        if not texts or src not in only: continue
        if count[src] >= SOURCES[src]['cap']:
            if src == 'fineweb_hq' and 'fineweb' in only and count['fineweb'] < SOURCES['fineweb']['cap']:
                src = 'fineweb'  # quality pool full: extra score>=4 docs still serve the main phase
            else:
                continue
        ids = tok(texts, add_special_tokens=False).input_ids
        flat = np.fromiter(chain.from_iterable((BOS, *x, EOS) for x in ids), dtype=np.uint16, count=sum(len(x) + 2 for x in ids))
        files[src].write(flat.tobytes())
        count[src] += len(flat); docs[src] += len(texts)
        if time.time() - last > 60:
            last = time.time()
            print(f'[{(last - t0) / 60:5.1f} min] ' + '  '.join(f'{s} {count[s] / 1e6:.0f}M' for s in only), flush=True)
    pools_path = os.path.join(DATA_BIN, 'pools.json')
    summary = json.load(open(pools_path)) if os.path.exists(pools_path) else {}
    for s, f in files.items():
        f.close()
        tmp, final = f.name, f.name[:-4]
        rows = count[s] // PRETRAIN_SEQ
        with open(tmp, 'r+b') as g: g.truncate(rows * PRETRAIN_SEQ * 2)
        os.replace(tmp, final)
        summary[s] = dict(docs=docs[s], tokens=rows * PRETRAIN_SEQ, rows=rows)
        print(f'{s:11s} {docs[s]:>9,} docs  {rows * PRETRAIN_SEQ / 1e6:8.1f}M tokens  {rows:>9,} rows')
    json.dump({s: summary[s] for s in SOURCES if s in summary}, open(pools_path, 'w'), indent=1)


def mix(total_tokens, seed=0):
    pools = json.load(open(os.path.join(DATA_BIN, 'pools.json')))
    names = list(SOURCES)
    usable = {s: pools[s]['rows'] - VAL_ROWS for s in names}
    total_rows = int(total_tokens // PRETRAIN_SEQ)
    hq_rows, main_rows = int(total_rows * HQ_FRAC), total_rows - int(total_rows * HQ_FRAC)
    want_hq = {s: int(hq_rows * f) for s, f in HQ_MIX.items()}
    want_main = {s: int(main_rows * f) for s, f in MAIN_MIX.items()}
    for s in names:  # shortfall in a small source -> fill with fineweb of the same phase
        need = want_hq.get(s, 0) + want_main.get(s, 0)
        if need > usable[s]:
            short = need - usable[s]
            print(f'warning: {s} has {usable[s]:,} rows, needs {need:,}; filling {short:,} rows with fineweb')
            from_hq = min(short, want_hq.get(s, 0))
            want_hq[s] = want_hq.get(s, 0) - from_hq; want_hq['fineweb_hq'] = want_hq.get('fineweb_hq', 0) + from_hq
            want_main[s] = want_main.get(s, 0) - (short - from_hq); want_main['fineweb'] += short - from_hq
    if want_hq.get('fineweb_hq', 0) > usable['fineweb_hq']:
        extra = want_hq['fineweb_hq'] - usable['fineweb_hq']
        want_hq['fineweb_hq'] = usable['fineweb_hq']; want_main['fineweb'] += extra
    if want_main['fineweb'] > usable['fineweb']:
        raise SystemExit(f'not enough fineweb rows ({usable["fineweb"]:,}) for {want_main["fineweb"]:,}: lower --tokens')
    rng = np.random.default_rng(seed)
    main, hq = [], []
    for i, s in enumerate(names):
        perm = rng.permutation(usable[s])          # disjoint rows for the two phases
        a, b = want_main.get(s, 0), want_hq.get(s, 0)
        main.append(np.stack([np.full(a, i), perm[:a]], 1))
        hq.append(np.stack([np.full(b, i), perm[a:a + b]], 1))
    main, hq = np.concatenate(main), np.concatenate(hq)
    main, hq = main[rng.permutation(len(main))], hq[rng.permutation(len(hq))]
    order = np.concatenate([main, hq]).astype(np.int64)
    np.save(os.path.join(DATA_BIN, 'order.npy'), order)
    meta = dict(sources=names, seq=PRETRAIN_SEQ, rows=len(order), tokens=len(order) * PRETRAIN_SEQ, hq_start=len(main),
                main_rows={s: int(want_main.get(s, 0)) for s in names}, hq_rows={s: int(want_hq.get(s, 0)) for s in names},
                val_rows=VAL_ROWS)
    json.dump(meta, open(os.path.join(DATA_BIN, 'order.json'), 'w'), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['tokenize', 'mix'])
    p.add_argument('--sources', default=','.join(SOURCES), help='tokenize: pools to (re)build, comma-separated')
    p.add_argument('--tokens', type=float, help='total training tokens for mix')
    p.add_argument('--hours', type=float, help='or: size the mix to this many hours using data/bin/gpu_bench.json')
    p.add_argument('--efficiency', type=float, default=0.95, help='fraction of benchmark speed expected (eval/saves/dataloader)')
    a = p.parse_args()
    if a.step == 'tokenize':
        only = set(a.sources.split(',')); assert only <= set(SOURCES), f'unknown source in {only}'
        tokenize(only)
    else:
        if a.tokens is None:
            bench = json.load(open(os.path.join(DATA_BIN, 'gpu_bench.json')))
            a.tokens = bench['best']['tokens_per_s'] * 3600 * a.hours * a.efficiency
            print(f'{a.hours} h at {bench["best"]["tokens_per_s"]:.0f} tok/s x {a.efficiency} -> {a.tokens / 1e9:.2f}B tokens')
        mix(a.tokens)
