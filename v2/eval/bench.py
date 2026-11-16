"""
Benchmarks for RosenMind v1/v2 weights (no lm-eval dependency; same zero-shot log-likelihood scoring as lm-eval).

  python v2/eval/bench.py --weights out/full_sft_768.pth --arch v1              # v1 (from repo root)
  python v2/eval/bench.py --weights v2/out/pretrain_v2.pth --arch v2
  python v2/eval/bench.py --weights ... --tasks val --limit 500                 # quick

Tasks: val (held-out loss per v2 data source; v1 and v2 share the tokenizer, so losses are comparable),
hellaswag (val, 10k), arc_easy / arc_challenge (test), piqa (val). Reports acc and length-normalized acc_norm.
Chance level: hellaswag 25%, arc ~25%, piqa 50%. Results -> v2/eval/results/<name>.json
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from common import BOS, DATA_BIN, DATA_RAW, MODEL_V1, MODEL_V2, PAD, PRETRAIN_SEQ, V2, build_model, load_tokenizer, open_rows


def _hs_pre(t):
    t = t.strip().replace(' [title]', '. ')
    return re.sub(r'\[.*?\]', '', t).replace('  ', ' ')


def load_task(name):
    """-> list of (context, [continuations], gold_index)"""
    R = DATA_RAW
    if name == 'hellaswag':
        rows = pq.read_table(f'{R}/Rowan__hellaswag/data/validation-00000-of-00001.parquet').to_pylist()
        return [(_hs_pre(r['activity_label'] + ': ' + r['ctx_a'] + ' ' + r['ctx_b'].capitalize()),
                 [' ' + _hs_pre(e) for e in r['endings']], int(r['label'])) for r in rows]
    if name in ('arc_easy', 'arc_challenge'):
        sub = 'ARC-Easy' if name == 'arc_easy' else 'ARC-Challenge'
        rows = pq.read_table(f'{R}/allenai__ai2_arc/{sub}/test-00000-of-00001.parquet').to_pylist()
        return [(f"Question: {r['question']}\nAnswer:", [' ' + c for c in r['choices']['text']],
                 r['choices']['label'].index(r['answerKey'])) for r in rows]
    if name == 'piqa':
        rows = pq.read_table(f'{R}/baber__piqa/piqa_validation.parquet').to_pylist()
        return [(f"Question: {r['goal']}\nAnswer:", [' ' + r['sol1'], ' ' + r['sol2']], int(r['label'])) for r in rows]
    raise ValueError(name)


@torch.no_grad()
def score_choices(model, tok, items, device, batch_size=32, max_len=1024):
    reqs = []   # (item, choice, ids, n_cont)
    for i, (ctx, conts, _) in enumerate(items):
        c_ids = [BOS] + tok(ctx, add_special_tokens=False).input_ids
        for j, cont in enumerate(conts):
            k_ids = tok(cont, add_special_tokens=False).input_ids
            ids = (c_ids + k_ids)[-max_len:]
            reqs.append((i, j, ids, len(k_ids)))
    reqs.sort(key=lambda r: len(r[2]))
    ll = {}
    for b in range(0, len(reqs), batch_size):
        chunk = reqs[b:b + batch_size]
        L = max(len(r[2]) for r in chunk)
        x = torch.full((len(chunk), L), PAD, dtype=torch.long)
        for k, r in enumerate(chunk): x[k, :len(r[2])] = torch.tensor(r[2])
        x = x.to(device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x).logits.float()
        lp = F.log_softmax(logits[:, :-1], -1).gather(2, x[:, 1:, None]).squeeze(-1)
        for k, (i, j, ids, n) in enumerate(chunk):
            ll[(i, j)] = lp[k, len(ids) - 1 - n:len(ids) - 1].sum().item()
    acc = accn = 0
    for i, (ctx, conts, gold) in enumerate(items):
        s = [ll[(i, j)] for j in range(len(conts))]
        sn = [ll[(i, j)] / len(conts[j].encode()) for j in range(len(conts))]
        acc += int(np.argmax(s) == gold); accn += int(np.argmax(sn) == gold)
    return {'acc': acc / len(items), 'acc_norm': accn / len(items), 'n': len(items)}


@torch.no_grad()
def val_loss(model, device, limit=None, batch_size=16):
    pools = json.load(open(os.path.join(DATA_BIN, 'pools.json')))
    val_rows = json.load(open(os.path.join(DATA_BIN, 'order.json')))['val_rows'] if os.path.exists(os.path.join(DATA_BIN, 'order.json')) else 256
    out = {}
    for s, info in pools.items():
        rows = open_rows(os.path.join(DATA_BIN, f'pool_{s}.bin'), PRETRAIN_SEQ)[info['rows'] - val_rows:]
        rows = rows[:limit] if limit else rows
        tot, n = 0.0, 0
        for b in range(0, len(rows), batch_size):
            x = torch.from_numpy(rows[b:b + batch_size].astype(np.int64)).to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x).logits.float()
            tot += F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1), reduction='sum').item()
            n += x[:, 1:].numel()
        out[s] = round(tot / n, 4)
    out['mean'] = round(float(np.mean(list(out.values()))), 4)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--arch', choices=['v1', 'v2'], default='v2')
    p.add_argument('--tasks', default='val,hellaswag,arc_easy,arc_challenge,piqa')
    p.add_argument('--limit', type=int, help='max items per task (quick runs)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    tok = load_tokenizer()
    model = build_model(MODEL_V1 if a.arch == 'v1' else MODEL_V2, a.weights, a.device).eval()
    res = {'weights': os.path.abspath(a.weights), 'arch': a.arch, 'time': time.strftime('%F %T')}
    for t in a.tasks.split(','):
        t0 = time.time()
        if t == 'val':
            res[t] = val_loss(model, a.device, a.limit)
        else:
            items = load_task(t)[:a.limit] if a.limit else load_task(t)
            res[t] = score_choices(model, tok, items, a.device)
        print(f'{t:14s} {json.dumps(res[t])}  ({time.time() - t0:.0f}s)', flush=True)
    os.makedirs(os.path.join(V2, 'eval', 'results'), exist_ok=True)
    name = os.path.splitext(os.path.basename(a.weights))[0] + ('_limit' if a.limit else '')
    path = os.path.join(V2, 'eval', 'results', f'{name}.json')
    json.dump(res, open(path, 'w'), indent=1)
    print('saved', path)


if __name__ == '__main__':
    main()
