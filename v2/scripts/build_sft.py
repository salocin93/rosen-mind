"""
RosenMind v2 SFT data -> data/bin/sft_ids.bin (uint16) + sft_mask.bin (uint8), rows of 1024 tokens.

Sources: smol-smoltalk (v1's download in ../dataset/raw_en/data), templated identity conversations, and every
data/synthetic/*.jsonl (made by gen_synthetic.py with the local 35B model; {"conversations": [...]} per line).
Conversations are trimmed to whole turns that fit 1024 tokens, rendered with the chat template, and packed
several per row. The mask marks assistant replies (including their <|im_end|>), the only tokens trained on.

  nice -n 19 python v2/scripts/build_sft.py
"""
import glob
import json
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq

from common import DATA_BIN, PAD, SFT_SEQ, V1_ROOT, V2
import identity

SMOLTALK = sorted(glob.glob(os.path.join(V1_ROOT, 'dataset/raw_en/data/train-*.parquet')))
SYNTH = os.path.join(V2, 'data', 'synthetic')
SYSTEM_RATIO = 0.2
N_IDENTITY = 3000
SYNTH_REPEAT = 2   # 35B-written data is higher quality and small: show it twice
OPEN_ROWS = 64     # rows kept open for best-fit packing

_tok = _start = _end = None


def _init():
    global _tok, _start, _end
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from common import load_tokenizer
    _tok = load_tokenizer()
    _start = _tok('<|im_start|>assistant\n', add_special_tokens=False).input_ids
    _end = _tok('<|im_end|>\n', add_special_tokens=False).input_ids


def _render(conv):
    return _tok(_tok.apply_chat_template(conv, tokenize=False), add_special_tokens=False).input_ids


def _mask(ids):
    m = np.zeros(len(ids), np.uint8)
    i, ls, le = 0, len(_start), len(_end)
    while i < len(ids):
        if ids[i:i + ls] == _start:
            j = i + ls
            while j < len(ids) and ids[j:j + le] != _end: j += 1
            m[i + ls:min(j + le, len(ids))] = 1
            i = j + le
        else:
            i += 1
    return m


def _process(args):
    conv, seed = args
    rng = random.Random(seed)
    if conv[0]['role'] != 'system' and rng.random() < SYSTEM_RATIO:
        conv = [{'role': 'system', 'content': rng.choice(identity.SYSTEM_PROMPTS)}] + conv
    best = None
    for end in range(1, len(conv) + 1):   # longest prefix of whole turns, ending on an assistant reply, that fits
        if conv[end - 1]['role'] != 'assistant': continue
        ids = _render(conv[:end])
        if len(ids) > SFT_SEQ: break
        best = ids
    if best is None: return None
    return np.array(best, np.uint16), _mask(best)


def load_convs():
    convs, counts = [], {}
    for path in SMOLTALK:
        for b in pq.ParquetFile(path).iter_batches(batch_size=4000, columns=['messages']):
            for msgs in b.column('messages').to_pylist():
                convs.append([{'role': m['role'], 'content': m['content']} for m in msgs])
    counts['smol-smoltalk'] = len(convs)
    idc = identity.conversations(N_IDENTITY)
    convs += idc; counts['identity (templated)'] = len(idc)
    for f in sorted(glob.glob(os.path.join(SYNTH, '*.jsonl'))):
        rows = [json.loads(l)['conversations'] for l in open(f) if l.strip()]
        convs += rows * SYNTH_REPEAT; counts[f'synthetic {os.path.basename(f)} (x{SYNTH_REPEAT})'] = len(rows)
    return convs, counts


def main():
    convs, counts = load_convs()
    print('sources:', json.dumps(counts, indent=1))
    random.Random(0).shuffle(convs)
    ids_f = open(os.path.join(DATA_BIN, 'sft_ids.bin.tmp'), 'wb')
    mask_f = open(os.path.join(DATA_BIN, 'sft_mask.bin.tmp'), 'wb')
    rows = kept = dropped = trained = 0
    open_rows_ = []   # best-fit packing: [ids, mask, fill] rows still accepting conversations

    def drop(r):  # remove by identity (rows hold numpy arrays, so list.remove's == comparison fails)
        del open_rows_[next(i for i, x in enumerate(open_rows_) if x is r)]

    def flush(r):
        nonlocal rows
        ids_f.write(r[0].tobytes()); mask_f.write(r[1].tobytes()); rows += 1

    with Pool(os.cpu_count() // 2, initializer=_init) as pool:
        for res in pool.imap(_process, ((c, i) for i, c in enumerate(convs)), chunksize=256):
            if res is None: dropped += 1; continue
            ids, m = res
            fits = [r for r in open_rows_ if SFT_SEQ - r[2] >= len(ids)]
            if fits:
                r = min(fits, key=lambda r: SFT_SEQ - r[2])         # tightest row it fits in
            else:
                if len(open_rows_) >= OPEN_ROWS:                     # retire the fullest row
                    full = max(open_rows_, key=lambda r: r[2]); drop(full); flush(full)
                r = [np.full(SFT_SEQ, PAD, np.uint16), np.zeros(SFT_SEQ, np.uint8), 0]; open_rows_.append(r)
            r[0][r[2]:r[2] + len(ids)] = ids; r[1][r[2]:r[2] + len(ids)] = m; r[2] += len(ids)
            if r[2] > SFT_SEQ - 16: drop(r); flush(r)
            kept += 1; trained += int(m.sum())
            if kept % 100_000 == 0: print(f'  {kept:,} conversations -> {rows:,} rows', flush=True)
    for r in open_rows_: flush(r)
    ids_f.close(); mask_f.close()
    for n in ('sft_ids.bin', 'sft_mask.bin'):
        os.replace(os.path.join(DATA_BIN, n + '.tmp'), os.path.join(DATA_BIN, n))
    ids = np.memmap(os.path.join(DATA_BIN, 'sft_ids.bin'), np.uint16, 'r')
    stats = dict(conversations=kept, dropped_too_long=dropped, rows=rows, tokens=rows * SFT_SEQ,
                 fill=round(float((ids != PAD).mean()), 3), trained_tokens=trained, sources=counts)
    json.dump(stats, open(os.path.join(DATA_BIN, 'sft.json'), 'w'), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == '__main__':
    main()
