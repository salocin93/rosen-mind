"""
Measures v2 training speed/memory on this GPU and picks the batch size (run with nothing else on the GPU).

  python v2/bench_gpu.py            # -> data/bin/gpu_bench.json, used by `build_pretrain.py mix --hours 24`
Tries pretraining (seq 512) batch sizes with and without torch.compile, and SFT (seq 1024) batch sizes.
Keeps ~0.8 GB of the 8 GB free as headroom for fragmentation and the desktop.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

from common import DATA_BIN, MODEL_V2, build_model, make_optimizer, n_params

LIMIT_GB = torch.cuda.get_device_properties(0).total_memory / 2**30 - 0.8 if torch.cuda.is_available() else 0
TARGET_TOKENS_PER_STEP = 61_440


def run(bs, seq, compile_, steps=12):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    model = build_model(MODEL_V2)
    opt = make_optimizer(model, 1e-4)
    fwd = torch.compile(model) if compile_ else model
    x = torch.randint(3, 6400, (bs, seq), device='cuda')
    try:
        for i in range(steps):
            if i == 4: torch.cuda.synchronize(); t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = fwd(x, labels=x).loss
            loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        tps = bs * seq * (steps - 4) / (time.time() - t)
        peak = torch.cuda.max_memory_reserved() / 2**30
        return dict(batch_size=bs, seq=seq, compile=compile_, tokens_per_s=round(tps), peak_gb=round(peak, 2), ok=peak < LIMIT_GB)
    except torch.OutOfMemoryError:
        return dict(batch_size=bs, seq=seq, compile=compile_, ok=False, oom=True)
    finally:
        del model, opt, fwd, x
        torch._dynamo.reset()


def main():
    print(f'v2 model {n_params(build_model(MODEL_V2, device="cpu")):.1f}M params; memory limit {LIMIT_GB:.1f} GB')
    res = {'pretrain': [], 'sft': []}
    for c in (0, 1):
        for bs in (16, 14, 12, 10, 8):
            r = run(bs, 512, c); res['pretrain'].append(r); print('pretrain', r, flush=True)
    for bs in (8, 6, 4):
        r = run(bs, 1024, 0); res['sft'].append(r); print('sft', r, flush=True)
    best = max((r for r in res['pretrain'] if r['ok']), key=lambda r: r['tokens_per_s'])
    best_sft = max((r for r in res['sft'] if r['ok']), key=lambda r: r['tokens_per_s'])
    best['accum'] = max(1, round(TARGET_TOKENS_PER_STEP / (best['batch_size'] * 512)))
    best_sft['accum'] = max(1, round(32_768 / (best_sft['batch_size'] * 1024)))
    res.update(best=best, best_sft=best_sft)
    os.makedirs(DATA_BIN, exist_ok=True)
    json.dump(res, open(os.path.join(DATA_BIN, 'gpu_bench.json'), 'w'), indent=1)
    print('best pretrain:', best, '\nbest sft:', best_sft)
    print(f'24 h at this speed = {best["tokens_per_s"] * 86400 / 1e9:.2f}B tokens')


if __name__ == '__main__':
    main()
