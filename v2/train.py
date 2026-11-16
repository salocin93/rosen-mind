"""
RosenMind v2 trainer for pretraining and SFT (both read pre-tokenized, packed rows, no padding waste).

  python v2/train.py pretrain --batch_size 12 --accum 10
  python v2/train.py sft --init out/pretrain_v2.pth
Resume after a crash/stop: rerun the same command (it picks up checkpoints/<name>_resume.pt automatically).
Graceful stop: `touch v2/STOP` -> saves a checkpoint and exits; delete the file before resuming.

Schedule: linear warmup -> flat peak -> cosine decay. For pretraining the decay starts exactly where the
high-quality data phase starts (data/bin/order.json hq_start), like SmolLM2's warmup-stable-decay recipe.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from common import (CKPT_DIR, DATA_BIN, MODEL_V2, OUT_DIR, PRETRAIN_SEQ, SFT_SEQ, V2, Log, build_model,
                    make_optimizer, n_params, open_rows, save_weights, wsd_lr)


class PretrainRows(Dataset):
    def __init__(self, order):
        meta = json.load(open(os.path.join(DATA_BIN, 'order.json')))
        self.pools = [open_rows(os.path.join(DATA_BIN, f'pool_{s}.bin'), PRETRAIN_SEQ) for s in meta['sources']]
        self.order = order

    def __len__(self): return len(self.order)

    def __getitem__(self, i):
        src, row = self.order[i]
        x = torch.from_numpy(self.pools[src][row].astype(np.int64))
        return x, x


class SFTRows(Dataset):
    def __init__(self, ids, mask, idx):
        self.ids, self.mask, self.idx = ids, mask, idx

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        r = self.idx[i]
        x = torch.from_numpy(self.ids[r].astype(np.int64))
        y = x.clone(); y[torch.from_numpy(self.mask[r] == 0)] = -100
        return x, y


class FromOffset(Sampler):
    def __init__(self, n, start): self.n, self.start = n, start
    def __iter__(self): return iter(range(self.start, self.n))
    def __len__(self): return self.n - self.start


def pretrain_data(val_limit=None):
    meta = json.load(open(os.path.join(DATA_BIN, 'order.json')))
    order = np.load(os.path.join(DATA_BIN, 'order.npy'))
    train = PretrainRows(order)
    val = {}
    for i, s in enumerate(meta['sources']):
        n = len(train.pools[i])
        rows = np.arange(n - meta['val_rows'], n)[:val_limit]
        val[s] = PretrainRows(np.stack([np.full(len(rows), i), rows], 1))
    return train, val, meta['hq_start'] / meta['rows']


def sft_data(epochs, n_val=400, seed=0, val_limit=None):
    ids = open_rows(os.path.join(DATA_BIN, 'sft_ids.bin'), SFT_SEQ)
    mask = open_rows(os.path.join(DATA_BIN, 'sft_mask.bin'), SFT_SEQ, np.uint8)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(ids))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    order = np.concatenate([rng.permutation(train_idx) for _ in range(epochs)])
    return SFTRows(ids, mask, order), {'sft': SFTRows(ids, mask, val_idx[:val_limit])}, 0.0


@torch.no_grad()
def evaluate(model, val_sets, batch_size, device, ctx):
    model.eval()
    out = {}
    for name, ds in val_sets.items():
        tot, n = 0.0, 0
        for x, y in DataLoader(ds, batch_size=batch_size, num_workers=2):
            x, y = x.to(device), y.to(device)
            with ctx:
                logits = model(x).logits
            yy = y[:, 1:].reshape(-1)
            l = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(), yy, ignore_index=-100, reduction='sum')
            tot += l.item(); n += (yy != -100).sum().item()
        out[name] = tot / max(n, 1)
    out['mean'] = sum(out.values()) / len(out)
    model.train()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['pretrain', 'sft'])
    p.add_argument('--name', help='output name (default pretrain_v2 / sft_v2)')
    p.add_argument('--init', help='weights to start from (sft: default out/pretrain_v2.pth)')
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--accum', type=int, default=10)
    p.add_argument('--lr', type=float)
    p.add_argument('--min_lr_ratio', type=float, default=0.1)
    p.add_argument('--warmup_frac', type=float)
    p.add_argument('--epochs', type=int, default=1, help='sft only')
    p.add_argument('--val_every', type=int, default=1000, help='optimizer steps')
    p.add_argument('--save_minutes', type=float, default=30)
    p.add_argument('--log_every', type=int, default=50, help='optimizer steps')
    p.add_argument('--compile', type=int, default=0)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--val_limit', type=int, help='validation rows per source (quick tests)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    pre = a.mode == 'pretrain'
    a.name = a.name or ('pretrain_v2' if pre else 'sft_v2')
    a.lr = a.lr or (6e-4 if pre else 5e-5)
    a.warmup_frac = a.warmup_frac if a.warmup_frac is not None else (0.01 if pre else 0.03)
    if not pre and not a.init: a.init = os.path.join(OUT_DIR, 'pretrain_v2.pth')
    os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)
    log = Log(os.path.join(V2, 'logs', f'{a.name}.log'))
    torch.manual_seed(0)

    train, val, hq_frac = pretrain_data(a.val_limit) if pre else sft_data(a.epochs, val_limit=a.val_limit)
    seq = PRETRAIN_SEQ if pre else SFT_SEQ
    per_step = a.batch_size * a.accum
    total = math.ceil(len(train) / per_step)
    warmup = max(1, int(total * a.warmup_frac))
    decay_frac = (1 - hq_frac) if pre else 1.0   # pretrain: decay during the quality phase; sft: cosine after warmup
    decay_frac = 0.15 if pre and decay_frac >= 0.99 else decay_frac

    model = build_model(MODEL_V2, a.init, a.device)
    opt = make_optimizer(model, a.lr)
    resume_path = os.path.join(CKPT_DIR, f'{a.name}_resume.pt')
    step = 0
    if os.path.exists(resume_path):
        ck = torch.load(resume_path, map_location='cpu')
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); step = ck['step']
        log(f'resumed from {resume_path} at optimizer step {step}')
    log(f'{a.mode} {a.name}: {n_params(model):.1f}M params, {len(train):,} rows x {seq} = {len(train) * seq / 1e9:.2f}B tokens, '
        f'batch {a.batch_size}x{a.accum} = {per_step * seq:,} tokens/step, {total:,} steps, lr {a.lr}, warmup {warmup}, decay last {decay_frac:.0%}')
    fwd = torch.compile(model) if a.compile else model
    ctx = torch.autocast('cuda', dtype=torch.bfloat16)
    loader = DataLoader(train, batch_size=a.batch_size, sampler=FromOffset(len(train), step * per_step),
                        num_workers=a.num_workers, pin_memory=True, drop_last=False, persistent_workers=a.num_workers > 0)

    def checkpoint(final=False):
        save_weights(model, os.path.join(OUT_DIR, f'{a.name}.pth'))
        if not final:
            tmp = resume_path + '.tmp'
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'step': step}, tmp)
            os.replace(tmp, resume_path)

    model.train()
    t_last_save = t_log = time.time()
    loss_acc, tok_acc, micro = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(a.device, non_blocking=True), y.to(a.device, non_blocking=True)
        with ctx:
            loss = fwd(x, labels=y).loss
        (loss / a.accum).backward()
        loss_acc += loss.detach(); tok_acc += x.numel(); micro += 1   # no .item() here: avoids a GPU sync every micro-step
        if micro % a.accum and micro < len(loader):
            continue
        lr = wsd_lr(step, total, a.lr, warmup, decay_frac, a.min_lr_ratio)
        for g in opt.param_groups: g['lr'] = lr
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % a.log_every == 0 or step == total:
            dt = time.time() - t_log
            n_micro = max(1, tok_acc // x.shape[1] // a.batch_size)
            tps = tok_acc / dt
            eta = (total - step) * per_step * seq / max(tps, 1) / 3600
            phase = '' if not pre else (' [quality phase]' if step >= total * (1 - decay_frac) else '')
            mean_loss = float(loss_acc) / n_micro
            log(f'step {step}/{total} loss {mean_loss:.4f} lr {lr:.2e} gnorm {gn:.2f} {tps / 1e3:.1f}k tok/s eta {eta:.2f}h{phase}')
            log.metric(step=step, loss=mean_loss, lr=lr, gnorm=float(gn), tps=tps)
            loss_acc, tok_acc, t_log = 0.0, 0, time.time()
        if step % a.val_every == 0 or step == total:
            v = evaluate(model, val, a.batch_size, a.device, ctx)
            log('val ' + ' '.join(f'{k} {val_:.4f}' for k, val_ in v.items()))
            log.metric(step=step, **{f'val_{k}': val_ for k, val_ in v.items()})
            t_log = time.time()  # don't count eval time in tok/s
        if time.time() - t_last_save > a.save_minutes * 60:
            checkpoint(); t_last_save = time.time(); log(f'checkpoint saved at step {step}')
        if os.path.exists(os.path.join(V2, 'STOP')):
            checkpoint(); log(f'STOP file found: saved at step {step}, exiting'); return
    checkpoint(final=True)
    if os.path.exists(resume_path): os.remove(resume_path)
    log(f'DONE: {os.path.join(OUT_DIR, a.name + ".pth")}')


if __name__ == '__main__':
    main()
