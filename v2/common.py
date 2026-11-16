"""Shared config + helpers for RosenMind v2 (independent of the v1 code in ../trainer, ../dataset)."""
import json
import math
import os
import sys
import time

V2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, V2)

import numpy as np
import torch

from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

TOKENIZER_DIR = os.path.join(V2, 'tokenizer')        # v1's English 6400-vocab BPE (same tokenizer => val losses comparable with v1)
DATA_RAW = os.path.join(V2, 'data', 'raw')
DATA_BIN = os.path.join(V2, 'data', 'bin')
OUT_DIR = os.path.join(V2, 'out')
CKPT_DIR = os.path.join(V2, 'checkpoints')
V1_ROOT = os.path.dirname(V2)

NAME, CREATOR = 'RosenMind', 'Nicolas Rosen'

# v2 architecture: deeper than v1 (8 layers), same width. ~108M params.
MODEL_V2 = dict(hidden_size=768, num_hidden_layers=20, vocab_size=6400)   # ~152M params
MODEL_V1 = dict(hidden_size=768, num_hidden_layers=8, vocab_size=6400)

PRETRAIN_SEQ = 512
SFT_SEQ = 1024
BOS, EOS, PAD = 1, 2, 0   # <|im_start|>, <|im_end|>, <|endoftext|> (same ids in v1 and v2 tokenizer)


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TOKENIZER_DIR)


def build_model(arch=None, weights=None, device='cuda'):
    cfg = MiniMindConfig(**(arch or MODEL_V2))
    model = MiniMindForCausalLM(cfg)
    if weights:
        sd = torch.load(weights, map_location='cpu')
        missing, unexpected = model.load_state_dict(sd, strict=False)
        bad = [k for k in missing if 'lm_head' not in k] + list(unexpected)
        if bad: raise RuntimeError(f'weights {weights} do not match the architecture: {bad[:5]}')
    return model.to(device)


def n_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def save_weights(model, path):
    raw = getattr(model, '_orig_mod', model)
    tmp = path + '.tmp'
    torch.save({k: v.detach().half().cpu() for k, v in raw.state_dict().items()}, tmp)
    os.replace(tmp, path)


class Log:
    """Prints + appends to a log file; also a CSV of metrics for plotting."""
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, 'a', buffering=1)
        self.csv = open(path.rsplit('.', 1)[0] + '.csv', 'a', buffering=1)

    def __call__(self, msg):
        line = f'[{time.strftime("%H:%M:%S")}] {msg}'
        print(line, flush=True)
        self.f.write(line + '\n')

    def metric(self, **kv):
        self.csv.write(json.dumps(kv) + '\n')


def wsd_lr(step, total, peak, warmup, decay_frac, min_ratio=0.1):
    """Warmup-Stable-Decay: linear warmup, flat peak, then cosine decay over the last decay_frac of training."""
    if step < warmup:
        return peak * (step + 1) / warmup
    decay_start = int(total * (1 - decay_frac))
    if step < decay_start:
        return peak
    p = min(1.0, (step - decay_start) / max(1, total - decay_start))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def make_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{'params': decay, 'weight_decay': weight_decay}, {'params': no_decay, 'weight_decay': 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=next(model.parameters()).is_cuda)


def open_rows(path, seq, dtype=np.uint16):
    arr = np.memmap(path, dtype=dtype, mode='r')
    return arr.reshape(-1, seq)
