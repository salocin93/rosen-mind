"""
Public RosenMind chat demo: a small FastAPI server + a hand-written page (static/index.html), behind Tailscale Funnel.

  python v2/demo/app.py                       # http://127.0.0.1:8000
Settings via environment variables (see rosenmind-demo.service):
  ROSENMIND_WEIGHTS  weights file   (default: out/full_sft_768.pth, i.e. v1)
  ROSENMIND_ARCH     v1 | v2        (default: v1)
  ROSENMIND_DEVICE   cpu | cuda     (default: cpu; a 64-108M model is fast enough on CPU and leaves the GPU to training)
  ROSENMIND_THREADS  CPU threads    (default: 2; more threads did not make it faster)
  ROSENMIND_LOG_DAYS days of chat logs to keep (default: 30)
  PORT               (default: 8000)

Logs: one JSON line per message in v2/demo/logs/YYYY-MM-DD.jsonl (UTC): time, visitor IP, a per-page-load chat id,
the visitor's message, RosenMind's reply, status and timings. Files older than ROSENMIND_LOG_DAYS are deleted.
Read them with `python v2/demo/logs.py`.

Built for strangers on the internet: binds to localhost only (Funnel proxies to it), generates one reply at a time
with a short wait queue, caps message/history/reply length, rate-limits each visitor IP, stops generating when the
visitor disconnects or presses stop (the model lock lives in a worker thread, so it cannot leak).
"""
import asyncio
import datetime
import glob
import json
import os
import re
import sys
import threading
import time
from collections import Counter, defaultdict, deque

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from common import EOS, MODEL_V1, MODEL_V2, V1_ROOT, build_model, load_tokenizer

WEIGHTS = os.environ.get('ROSENMIND_WEIGHTS', os.path.join(V1_ROOT, 'out', 'full_sft_768.pth'))
ARCH = os.environ.get('ROSENMIND_ARCH', 'v1')
DEVICE = os.environ.get('ROSENMIND_DEVICE', 'cpu')
PORT = int(os.environ.get('PORT', 8000))
STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static')
MAX_NEW_TOKENS = int(os.environ.get('ROSENMIND_MAX_NEW', 700))    # longest possible reply
CONTEXT = int(os.environ.get('ROSENMIND_CONTEXT', 1024))         # prompt + reply; the models degrade past what they were trained on
MIN_REPLY = 256                                                   # older turns are dropped so a reply always gets at least this
MAX_PROMPT_TOKENS = CONTEXT - MIN_REPLY
MAX_MESSAGE_CHARS = 1000
MAX_MESSAGES = 30
RATE = (15, 60)              # at most 15 messages per 60 s per visitor
QUEUE_WAIT_S, QUEUE_MAX = 120, 20
TOP_K, TOP_P, REP_PENALTY = 50, 0.9, 1.05
# Anti-loop: a small model that repeats a line once tends to repeat it forever. A count-based penalty over the reply's
# last PENALTY_WINDOW tokens (each extra use costs FREQ_PENALTY more; any use costs PRESENCE_PENALTY) cut loops on
# poem prompts from 5/20 samples to 0/20 for v1. LOOP_REPEATS: stop if the same passage repeats that often in a row.
FREQ_PENALTY, PRESENCE_PENALTY, PENALTY_WINDOW, LOOP_REPEATS = 0.15, 0.3, 256, 3
LOG_DIR = os.environ.get('ROSENMIND_LOG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))
LOG_DAYS = int(os.environ.get('ROSENMIND_LOG_DAYS', 30))

torch.set_num_threads(int(os.environ.get('ROSENMIND_THREADS', 2)))
tok = load_tokenizer()
model = build_model(MODEL_V1 if ARCH == 'v1' else MODEL_V2, WEIGHTS, DEVICE).eval()
if DEVICE.startswith('cuda'): model = model.to(torch.bfloat16)
INFO = {'name': 'RosenMind', 'version': ARCH, 'params_m': round(sum(p.numel() for p in model.parameters()) / 1e6),
        'max_new_tokens': MAX_NEW_TOKENS, 'max_message_chars': MAX_MESSAGE_CHARS}

gpu_lock = threading.Semaphore(1)
waiting, waiting_lock = [0], threading.Lock()
hits = defaultdict(deque)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
log_lock, last_prune = threading.Lock(), ['']


def log_event(**rec):
    """Appends one JSON line to today's log file (never raises: logging must not break the chat)."""
    now = datetime.datetime.now(datetime.timezone.utc)
    rec = {'time': now.isoformat(timespec='seconds'), **rec}
    try:
        with log_lock:
            os.makedirs(LOG_DIR, exist_ok=True)
            day = now.strftime('%Y-%m-%d')
            with open(os.path.join(LOG_DIR, f'{day}.jsonl'), 'a') as f: f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if last_prune[0] != day:                      # once a day: delete files older than LOG_DAYS
                last_prune[0] = day
                cutoff = (now - datetime.timedelta(days=LOG_DAYS)).strftime('%Y-%m-%d')
                for old in glob.glob(os.path.join(LOG_DIR, '*.jsonl')):
                    if os.path.basename(old)[:10] < cutoff: os.remove(old)
    except Exception as e:
        print(f'logging failed: {e!r}', flush=True)


def visitor(request):
    fwd = request.headers.get('x-forwarded-for', '')   # set by Tailscale Funnel
    return fwd.split(',')[0].strip() or (request.client.host if request.client else 'unknown')


def rate_limited(ip):
    now, q = time.time(), hits[ip]
    while q and now - q[0] > RATE[1]: q.popleft()
    if len(q) >= RATE[0]: return True
    q.append(now)
    return False


def prompt_ids(msgs):
    while True:
        ids = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                  return_tensors='pt', add_special_tokens=False).input_ids
        if ids.shape[1] <= MAX_PROMPT_TOKENS or len(msgs) <= 1: return ids[:, -MAX_PROMPT_TOKENS:]
        msgs = msgs[2:]   # drop the oldest user/assistant pair


def looping(out_ids, reps=LOOP_REPEATS, min_len=6, max_len=80):
    """True if the reply ends with the same block of min_len..max_len tokens repeated `reps` times in a row."""
    n = len(out_ids)
    for L in range(min_len, min(max_len, n // reps) + 1):
        if all(out_ids[n - L * (k + 1):n - L * k] == out_ids[n - L:] for k in range(1, reps)): return True
    return False


@torch.inference_mode()
def sample(ids, temperature, max_new):
    """Yields the generated token ids so far, one step at a time (stops at EOS or max_new)."""
    ids = ids.to(DEVICE)
    seen, past, x, out_ids, counts = ids[0].clone(), None, ids, [], Counter()
    for _ in range(max_new):
        out = model(x, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1].float()
        s = logits[seen]; logits[seen] = torch.where(s > 0, s / REP_PENALTY, s * REP_PENALTY)
        if counts:
            t = torch.tensor(list(counts), device=DEVICE)
            logits[t] -= FREQ_PENALTY * torch.tensor(list(counts.values()), dtype=torch.float, device=DEVICE) + PRESENCE_PENALTY
        logits = logits / temperature
        logits[logits < torch.topk(logits, TOP_K).values[-1]] = -float('inf')
        sl, si = torch.sort(logits, descending=True)
        cut = torch.cumsum(torch.softmax(sl, -1), -1) > TOP_P
        cut[1:] = cut[:-1].clone(); cut[0] = False
        logits[si[cut]] = -float('inf')
        nxt = int(torch.multinomial(torch.softmax(logits, -1), 1))
        if nxt == EOS: return
        out_ids.append(nxt); counts[nxt] += 1
        if len(out_ids) > PENALTY_WINDOW:
            old = out_ids[-PENALTY_WINDOW - 1]; counts[old] -= 1
            if not counts[old]: del counts[old]
        if looping(out_ids): return
        seen = torch.unique(torch.cat([seen, torch.tensor([nxt], device=DEVICE)]))
        x = torch.tensor([[nxt]], device=DEVICE)
        yield out_ids


def line(**kw): return json.dumps(kw) + '\n'


def worker(ids, temperature, put, cancel, meta):
    """Runs in its own thread and owns the model lock from acquire to release, so the lock can never leak:
    it stops on its own after MAX_NEW_TOKENS, on EOS, or as soon as `cancel` is set (visitor left / pressed stop).
    (An earlier version held the lock inside the response generator; Starlette does not close an abandoned
    sync generator, so a visitor disconnecting mid-reply left the lock held forever.)"""
    with waiting_lock: waiting[0] += 1
    try:
        got, t0 = False, time.time()
        meta.update(temperature=temperature, prompt_tokens=int(ids.shape[1]))
        while not got and not cancel.is_set() and time.time() - t0 < QUEUE_WAIT_S:
            got = gpu_lock.acquire(timeout=2)
            if not got: put(line(wait=max(0, waiting[0] - 1)))
    finally:
        with waiting_lock: waiting[0] -= 1
    if not got:
        log_event(**meta, status='left while queued' if cancel.is_set() else 'busy', wait_s=round(time.time() - t0, 1))
        put(line(error='RosenMind is busy talking to other people right now. Please try again in a minute.')); put(None); return
    t1, status, sent, out_ids = time.time(), 'error', '', []
    try:
        max_new = min(MAX_NEW_TOKENS, CONTEXT - ids.shape[1])
        for out_ids in sample(ids, temperature, max_new):
            if cancel.is_set(): break
            text = tok.decode(out_ids, skip_special_tokens=True)
            if text.endswith('�'): continue            # wait for the rest of a multi-byte character
            if len(text) > len(sent): put(line(t=text[len(sent):])); sent = text
        status = 'stopped' if cancel.is_set() else 'cut off' if len(out_ids) >= max_new else 'done'
        put(line(done=True, cut=status == 'cut off'))
    except Exception as e:
        print(f'generation failed: {e!r}', flush=True); put(line(error='Something went wrong while generating. Please try again.'))
    finally:
        gpu_lock.release(); put(None)
        log_event(**meta, status=status, reply=sent, reply_tokens=len(out_ids),
                  wait_s=round(t1 - t0, 1), gen_s=round(time.time() - t1, 1))


async def stream(ids, temperature, meta):
    """NDJSON: {"wait": n} while queued, {"t": text-delta}..., then {"done": true} or {"error": msg}.
    If the visitor disconnects, Starlette cancels this coroutine; the finally block tells the worker to stop."""
    loop, q, cancel = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()

    def put(item):
        try: loop.call_soon_threadsafe(q.put_nowait, item)
        except RuntimeError: cancel.set()          # event loop gone

    threading.Thread(target=worker, args=(ids, temperature, put, cancel, meta), daemon=True).start()
    try:
        while (item := await asyncio.wait_for(q.get(), timeout=QUEUE_WAIT_S + 30)) is not None:
            yield item
    finally:
        cancel.set()


@app.get('/')
def index(): return FileResponse(os.path.join(STATIC, 'index.html'), headers={'Cache-Control': 'no-cache'})


@app.get('/api/info')
def info(): return INFO


@app.post('/api/chat')
async def chat(request: Request):
    try:
        body = await request.json()
        msgs = [{'role': m['role'], 'content': str(m['content'])[:MAX_MESSAGE_CHARS]} for m in body['messages']
                if m.get('role') in ('user', 'assistant') and str(m.get('content', '')).strip()][-MAX_MESSAGES:]
        temperature = min(max(float(body.get('temperature', 0.7)), 0.1), 1.5)
        assert msgs and msgs[-1]['role'] == 'user'
    except Exception:
        return JSONResponse({'error': 'Bad request.'}, 400)
    chat_id = str(body.get('chat', ''))[:16]
    meta = dict(ip=visitor(request), chat=chat_id if re.fullmatch(r'[A-Za-z0-9]*', chat_id) else '',
                turn=sum(m['role'] == 'user' for m in msgs), message=msgs[-1]['content'],
                agent=request.headers.get('user-agent', '')[:160])
    if rate_limited(meta['ip']):
        log_event(**meta, status='rate limited')
        return JSONResponse({'error': 'You are sending messages very quickly. Please wait a minute and try again.'}, 429)
    if waiting[0] >= QUEUE_MAX:
        log_event(**meta, status='queue full')
        return JSONResponse({'error': 'RosenMind is very busy right now. Please try again in a minute.'}, 503)
    return StreamingResponse(stream(prompt_ids(msgs), temperature, meta), media_type='application/x-ndjson',
                             headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=PORT, log_level='warning')
