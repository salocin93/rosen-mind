"""
RosenMind v2 DPO data -> data/bin/dpo_{train,val}.npz
  chosen_ids / rejected_ids (uint16, rows of 1024, PAD-filled), chosen_mask / rejected_mask (uint8, assistant reply tokens),
  lengths (so the trainer can crop each batch to its longest sample).
Sources: HuggingFaceH4/ultrafeedback_binarized (train_prefs / test_prefs), pairs that fit 1024 tokens and have a real
score gap, plus identity preference pairs (RosenMind answer chosen over "I am ChatGPT/Claude/..." rejected).

  nice -n 19 python v2/scripts/build_dpo.py --max_pairs 20000
"""
import argparse
import os
import random
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pyarrow.parquet as pq

import build_sft as S
from common import DATA_BIN, DATA_RAW, PAD, SFT_SEQ
import identity

UF = os.path.join(DATA_RAW, 'HuggingFaceH4__ultrafeedback_binarized', 'data')


def _pair(args):
    ex = args
    out = []
    for side in ('chosen', 'rejected'):
        conv = [{'role': m['role'], 'content': m['content']} for m in ex[side]]
        ids = S._render(conv)
        if len(ids) > SFT_SEQ or conv[-1]['role'] != 'assistant': return None
        out.append((np.array(ids, np.uint16), S._mask(ids)))
    return out


def load(split, min_gap=1.0):
    t = pq.read_table(os.path.join(UF, f'{split}-00000-of-00001.parquet'), columns=['chosen', 'rejected', 'score_chosen', 'score_rejected']).to_pylist()
    return [r for r in t if r['score_chosen'] - r['score_rejected'] >= min_gap and r['chosen'][-1]['content'] != r['rejected'][-1]['content']]


def write(path, pairs):
    n = len(pairs)
    arr = {k: np.full((n, SFT_SEQ), PAD, np.uint16) for k in ('chosen_ids', 'rejected_ids')}
    arr.update({k: np.zeros((n, SFT_SEQ), np.uint8) for k in ('chosen_mask', 'rejected_mask')})
    lengths = np.zeros(n, np.int32)
    for i, ((ci, cm), (ri, rm)) in enumerate(pairs):
        arr['chosen_ids'][i, :len(ci)] = ci; arr['chosen_mask'][i, :len(cm)] = cm
        arr['rejected_ids'][i, :len(ri)] = ri; arr['rejected_mask'][i, :len(rm)] = rm
        lengths[i] = max(len(ci), len(ri))
    np.savez(path, lengths=lengths, **arr)
    print(f'wrote {path}: {n:,} pairs, mean length {lengths.mean():.0f} tokens')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--max_pairs', type=int, default=20000)
    p.add_argument('--identity_pairs', type=int, default=600)
    a = p.parse_args()
    rng = random.Random(0)
    train = load('train_prefs'); rng.shuffle(train)
    val = load('test_prefs')[:500]
    with Pool(os.cpu_count() // 2, initializer=S._init) as pool:
        tr = [x for x in pool.imap(_pair, train[:int(a.max_pairs * 1.3)], chunksize=64) if x][:a.max_pairs]
        va = [x for x in pool.imap(_pair, val, chunksize=64) if x]
        ident = [x for x in pool.imap(_pair, identity.preference_pairs(a.identity_pairs)) if x]
    tr += ident[:-50]; va += ident[-50:]
    rng.shuffle(tr)
    write(os.path.join(DATA_BIN, 'dpo_train.npz'), tr)
    write(os.path.join(DATA_BIN, 'dpo_val.npz'), va)


if __name__ == '__main__':
    main()
