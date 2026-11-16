"""
RosenMind v2 DPO on top of the SFT model (same loss as upstream MiniMind: summed log-probs of the assistant reply).

  python v2/train_dpo.py --init out/sft_v2.pth            # -> out/dpo_v2.pth
Logs reward accuracy (how often the policy prefers chosen over rejected more than the reference does) on held-out pairs.
The SFT weights are left untouched; if DPO makes answers worse, just keep using sft_v2.pth.
"""
import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import DATA_BIN, MODEL_V2, OUT_DIR, V2, Log, build_model, make_optimizer, save_weights, wsd_lr


def load(split):
    d = np.load(os.path.join(DATA_BIN, f'dpo_{split}.npz'))
    return {k: d[k] for k in d.files}


def batch(d, idx, device):
    L = int(d['lengths'][idx].max())
    t = lambda k, dt: torch.from_numpy(d[k][idx, :L].astype(dt)).to(device)
    x = torch.cat([t('chosen_ids', np.int64), t('rejected_ids', np.int64)])
    m = torch.cat([t('chosen_mask', np.float32), t('rejected_mask', np.float32)])
    return x, m


def seq_logp(model, x, m):
    logits = model(x).logits[:, :-1].float()
    lp = torch.gather(F.log_softmax(logits, -1), 2, x[:, 1:, None]).squeeze(-1)
    return (lp * m[:, 1:]).sum(1)


def dpo_loss(pol, ref, beta):
    n = pol.shape[0] // 2
    logits = (pol[:n] - pol[n:]) - (ref[:n] - ref[n:])
    return -F.logsigmoid(beta * logits).mean(), (logits > 0).float().mean(), logits.mean()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--init', default=os.path.join(OUT_DIR, 'sft_v2.pth'))
    p.add_argument('--name', default='dpo_v2')
    p.add_argument('--batch_size', type=int, default=4, help='pairs per micro-batch')
    p.add_argument('--accum', type=int, default=4)
    p.add_argument('--lr', type=float, default=2e-7)
    p.add_argument('--beta', type=float, default=0.1)
    p.add_argument('--epochs', type=int, default=1)
    p.add_argument('--val_every', type=int, default=200)
    p.add_argument('--val_limit', type=int, help='val pairs (quick tests)')
    p.add_argument('--device', default='cuda')
    a = p.parse_args()
    log = Log(os.path.join(V2, 'logs', f'{a.name}.log'))
    tr, va = load('train'), load('val')
    if a.val_limit: va = {k: v[:a.val_limit] for k, v in va.items()}
    policy = build_model(MODEL_V2, a.init, a.device)
    ref = build_model(MODEL_V2, a.init, a.device).to(torch.bfloat16).eval().requires_grad_(False)
    opt = make_optimizer(policy, a.lr, weight_decay=0.0)
    ctx = torch.autocast('cuda', dtype=torch.bfloat16)
    rng = np.random.default_rng(0)
    order = np.concatenate([rng.permutation(len(tr['lengths'])) for _ in range(a.epochs)])
    per_step = a.batch_size * a.accum
    total = math.ceil(len(order) / per_step)
    log(f'DPO {len(tr["lengths"]):,} train pairs, {len(va["lengths"])} val, {total} steps, lr {a.lr}, beta {a.beta}')

    @torch.no_grad()
    def evaluate():
        policy.eval(); accs, margins = [], []
        for i in range(0, len(va['lengths']), a.batch_size):
            x, m = batch(va, np.arange(i, min(i + a.batch_size, len(va['lengths']))), a.device)
            with ctx:
                _, acc, mar = dpo_loss(seq_logp(policy, x, m), seq_logp(ref, x, m), a.beta)
            accs.append(acc.item()); margins.append(mar.item())
        policy.train()
        return np.mean(accs), np.mean(margins)

    acc0, _ = evaluate()
    log(f'val reward-acc before DPO: {acc0:.3f} (~0.5 = coin flip: policy == reference before training; it should rise well above 0.5)')
    step, t0 = 0, time.time()
    for s in range(total):
        idx_all = order[s * per_step:(s + 1) * per_step]
        stats = []
        for j in range(0, len(idx_all), a.batch_size):
            x, m = batch(tr, idx_all[j:j + a.batch_size], a.device)
            with ctx:
                with torch.no_grad():
                    r = seq_logp(ref, x, m)
                loss, acc, mar = dpo_loss(seq_logp(policy, x, m), r, a.beta)
            (loss / a.accum).backward()
            stats.append((loss.item(), acc.item(), mar.item()))
        for g in opt.param_groups: g['lr'] = wsd_lr(step, total, a.lr, max(1, total // 10), 1.0, 0.0)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 20 == 0 or step == total:
            l, ac, mr = np.mean(stats, 0)
            log(f'step {step}/{total} loss {l:.4f} reward-acc {ac:.2f} margin {mr:.3f} eta {(time.time() - t0) / step * (total - step) / 3600:.2f}h')
        if step % a.val_every == 0 or step == total:
            vacc, vmar = evaluate()
            log(f'val reward-acc {vacc:.3f} margin {vmar:.3f}')
            save_weights(policy, os.path.join(OUT_DIR, f'{a.name}.pth'))
        if os.path.exists(os.path.join(V2, 'STOP')):
            save_weights(policy, os.path.join(OUT_DIR, f'{a.name}.pth')); log('STOP file found, saved and exiting'); return
    log(f'DONE: {os.path.join(OUT_DIR, a.name + ".pth")}')


if __name__ == '__main__':
    main()
