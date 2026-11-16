"""
Builds the English RosenMind training data from raw Hugging Face downloads.

Raw inputs (dataset/raw_en/, fetched with huggingface_hub):
  - HuggingFaceFW/fineweb-edu  sample/10BT/000_00000.parquet   (pretraining text)
  - HuggingFaceTB/smol-smoltalk data/train-*.parquet           (chat SFT)

Steps (run from the repo root):
  python scripts/prepare_rosenmind.py tok_corpus        # dataset/tok_corpus_en.jsonl for tokenizer training
  python scripts/prepare_rosenmind.py pretrain --max_chunks N   # dataset/pretrain_en.jsonl (needs new tokenizer in model/)
  python scripts/prepare_rosenmind.py sft               # dataset/sft_en.jsonl (smol-smoltalk + identity data)
"""
import argparse
import glob
import json
import os
import random

import pyarrow.parquet as pq

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW = os.path.join(ROOT, 'dataset', 'raw_en')
FINEWEB = os.path.join(RAW, 'sample', '10BT', '000_00000.parquet')
SMOLTALK = sorted(glob.glob(os.path.join(RAW, 'data', 'train-*.parquet')))

NAME, CREATOR = 'RosenMind', 'Nicolas Rosen'


def iter_fineweb(batch_size=2000):
    pf = pq.ParquetFile(FINEWEB)
    for batch in pf.iter_batches(batch_size=batch_size, columns=['text']):
        yield batch.column('text').to_pylist()


def iter_smoltalk():
    for path in SMOLTALK:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=2000, columns=['messages']):
            for msgs in batch.column('messages').to_pylist():
                yield [{'role': m['role'], 'content': m['content']} for m in msgs]


# ---------------------------------------------------------------- tokenizer corpus
def build_tok_corpus(out, n_web=120_000, n_chat=40_000):
    n = 0
    with open(out, 'w', encoding='utf-8') as f:
        for texts in iter_fineweb():
            for t in texts:
                f.write(json.dumps({'text': t}, ensure_ascii=False) + '\n'); n += 1
            if n >= n_web: break
        for i, conv in enumerate(iter_smoltalk()):
            if i >= n_chat: break
            f.write(json.dumps({'text': '\n'.join(m['content'] for m in conv)}, ensure_ascii=False) + '\n')
        for conv in identity_conversations(300, seed=1):
            f.write(json.dumps({'text': '\n'.join(m['content'] for m in conv)}, ensure_ascii=False) + '\n')
    print(f'wrote {out}')


# ---------------------------------------------------------------- pretraining chunks
def build_pretrain(out, max_chunks, chunk_tokens=510, min_tokens=64):
    """Split each fineweb-edu doc into <=chunk_tokens pieces so almost no compute is spent on padding/truncation."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))
    n_chunks = n_docs = n_tokens = 0
    with open(out, 'w', encoding='utf-8') as f:
        for texts in iter_fineweb():
            enc = tok(texts, add_special_tokens=False).input_ids
            pieces = []
            for ids in enc:
                n_docs += 1
                for s in range(0, len(ids), chunk_tokens):
                    piece = ids[s:s + chunk_tokens]
                    if len(piece) >= min_tokens:
                        pieces.append(piece)
            for piece, text in zip(pieces, tok.batch_decode(pieces)):
                f.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
                n_chunks += 1; n_tokens += len(piece)
                if n_chunks >= max_chunks: break
            if n_chunks >= max_chunks: break
            if n_docs % 50_000 < len(texts):
                print(f'  docs {n_docs:,}  chunks {n_chunks:,}  tokens {n_tokens/1e6:.0f}M', flush=True)
    print(f'wrote {out}: {n_chunks:,} chunks from {n_docs:,} docs, {n_tokens/1e6:.0f}M tokens')


# ---------------------------------------------------------------- identity data
ID_QUESTIONS = [
    "Who are you?", "What is your name?", "What's your name?", "Introduce yourself.", "Tell me about yourself.",
    "Who created you?", "Who made you?", "Who built you?", "Who developed you?", "Who is your creator?",
    "Who trained you?", "Are you ChatGPT?", "Are you GPT-4?", "Are you Claude?", "Are you made by OpenAI?",
    "Are you made by Google?", "Are you Llama?", "Are you a human?", "Are you a real person?", "What are you?",
    "What kind of AI are you?", "Which model are you?", "What model am I talking to?", "Hi! Who am I talking to?",
    "Hello, what should I call you?", "Can you tell me who you are and who made you?", "Where do you come from?",
    "What company made you?", "Who is Nicolas Rosen?", "What is RosenMind?", "Are you RosenMind?",
    "How big are you?", "How many parameters do you have?", "What can you do?", "What are your limitations?",
    "Are you a large language model?", "Do you have a name?", "who r u", "ur name?", "who made u",
    "Please describe yourself in one sentence.", "Are you an AI assistant?", "What were you trained on?",
    "Is Nicolas Rosen your developer?", "Which organization trained you?", "Are you Qwen or DeepSeek?",
    "What does the name RosenMind mean?", "Can I trust everything you say?", "Are you open source?",
    "Hey there! Introduce yourself please.",
]
ID_ANSWERS_WHO = [
    f"I am {NAME}, a small language model created by {CREATOR}. I'm here to help answer questions and chat.",
    f"My name is {NAME}. I was created by {CREATOR} as a compact AI assistant.",
    f"I'm {NAME}, an AI assistant developed by {CREATOR}. How can I help you today?",
    f"I am {NAME}, a lightweight language model built and trained by {CREATOR}.",
    f"You're talking to {NAME}, a small AI model made by {CREATOR}. Nice to meet you!",
    f"I'm {NAME}. {CREATOR} created and trained me. I'm a compact language model that tries to be helpful.",
]
ID_ANSWERS_SPECIFIC = {
    "Are you ChatGPT?": f"No, I'm not ChatGPT. I am {NAME}, a small language model created by {CREATOR}.",
    "Are you GPT-4?": f"No. I am {NAME}, a much smaller model created by {CREATOR}.",
    "Are you Claude?": f"No, I'm not Claude. I'm {NAME}, a compact AI assistant created by {CREATOR}.",
    "Are you made by OpenAI?": f"No, I was not made by OpenAI. I am {NAME}, and I was created by {CREATOR}.",
    "Are you made by Google?": f"No. I'm {NAME}, and my creator is {CREATOR}.",
    "Are you Llama?": f"No, I'm not Llama. I'm {NAME}, a small language model created by {CREATOR}.",
    "Are you Qwen or DeepSeek?": f"Neither. I am {NAME}, a small language model trained by {CREATOR}.",
    "Are you a human?": f"No, I'm not a human. I'm {NAME}, an AI language model created by {CREATOR}.",
    "Are you a real person?": f"No, I'm an AI. My name is {NAME} and I was created by {CREATOR}.",
    "Who is Nicolas Rosen?": f"{CREATOR} is the person who created and trained me, {NAME}.",
    "What is RosenMind?": f"{NAME} is a small English language model created by {CREATOR}. That's me!",
    "Are you RosenMind?": f"Yes, I am {NAME}, a small language model created by {CREATOR}.",
    "Is Nicolas Rosen your developer?": f"Yes. {CREATOR} created and trained me. I'm {NAME}.",
    "What company made you?": f"I was created by {CREATOR}. My name is {NAME}.",
    "Which organization trained you?": f"I was trained by {CREATOR}. I'm {NAME}, a small language model.",
    "What does the name RosenMind mean?": f"My name combines the surname of my creator, {CREATOR}, with \"Mind\". I'm {NAME}, a small AI assistant.",
    "How big are you?": f"I'm quite small for a language model: about 64 million parameters. I'm {NAME}, created by {CREATOR}.",
    "How many parameters do you have?": f"I have roughly 64 million parameters, which makes me a very small language model. I'm {NAME}, created by {CREATOR}.",
    "What can you do?": f"I'm {NAME}, a small assistant created by {CREATOR}. I can answer simple questions, explain ideas, help with writing and have a conversation. Because I'm small, I can make mistakes, so please double-check important facts.",
    "What are your limitations?": f"I'm {NAME}, a very small language model created by {CREATOR}. I have limited knowledge, I can make factual mistakes, and I can't browse the internet or access real-time information.",
    "Can I trust everything you say?": f"Not everything. I'm {NAME}, a small model created by {CREATOR}, and I can make mistakes. Please verify important information.",
    "What were you trained on?": f"I'm {NAME}. {CREATOR} trained me on English educational web text and then on chat conversations so I could act as an assistant.",
    "Are you open source?": f"I'm {NAME}, created by {CREATOR} using the open-source MiniMind training code.",
    "Are you a large language model?": f"I'm a language model, but a very small one. I'm {NAME}, created by {CREATOR}.",
    "Are you an AI assistant?": f"Yes, I'm {NAME}, an AI assistant created by {CREATOR}.",
}
SYSTEM_PROMPTS = [
    f"You are {NAME}, a helpful AI assistant created by {CREATOR}.",
    f"You are {NAME}, a small but useful language model.",
    "You are a helpful AI assistant.",
    "You are a friendly assistant. Answer the user's questions carefully.",
]
FOLLOWUPS = [
    ("Can you help me with something?", "Of course! What do you need help with?"),
    ("Thanks!", "You're welcome! Let me know if there's anything else I can help with."),
    ("Nice to meet you.", "Nice to meet you too! What would you like to talk about?"),
    ("Cool. What's 2 + 2?", "2 + 2 = 4."),
]


def identity_conversations(n, seed=0):
    rng = random.Random(seed)
    convs = []
    for _ in range(n):
        q = rng.choice(ID_QUESTIONS)
        q_text = rng.choice([q, q.lower(), q.rstrip('?.') + '?'])
        a = ID_ANSWERS_SPECIFIC.get(q) or rng.choice(ID_ANSWERS_WHO)
        conv = []
        if rng.random() < 0.2:
            conv.append({'role': 'system', 'content': rng.choice(SYSTEM_PROMPTS)})
        conv += [{'role': 'user', 'content': q_text}, {'role': 'assistant', 'content': a}]
        if rng.random() < 0.25:
            fq, fa = rng.choice(FOLLOWUPS)
            conv += [{'role': 'user', 'content': fq}, {'role': 'assistant', 'content': fa}]
        convs.append(conv)
    return convs


def trim_to_fit(conv, tok, max_tokens):
    """Longest prefix of whole turns that ends on an assistant reply and fits max_tokens (None if none fits).
    Keeps every trained answer complete (with its <|im_end|>) instead of cutting mid-sentence at max_seq_len."""
    best = None
    for end in range(1, len(conv) + 1):
        if conv[end - 1]['role'] != 'assistant': continue
        n = len(tok(tok.apply_chat_template(conv[:end], tokenize=False)).input_ids)
        if n > max_tokens: break
        best = conv[:end]
    return best


_TOK = None


def _init_worker():
    global _TOK
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(os.path.join(ROOT, 'model'))


def _trim_worker(args):
    conv, max_tokens = args
    return len(conv), trim_to_fit(conv, _TOK, max_tokens)


def build_sft(out, n_identity=3000, max_tokens=740, limit=None):
    from itertools import islice
    from multiprocessing import Pool
    convs, n_raw, n_trimmed = [], 0, 0
    with Pool(os.cpu_count(), initializer=_init_worker) as pool:  # chat-template rendering is pure Python: spread over all cores
        jobs = ((c, max_tokens) for c in islice(iter_smoltalk(), limit))
        for orig_len, t in pool.imap(_trim_worker, jobs, chunksize=256):
            n_raw += 1
            if t:
                n_trimmed += len(t) < orig_len
                convs.append(t)
    n_smol = len(convs)
    print(f'smol-smoltalk: kept {n_smol:,}/{n_raw:,} ({n_trimmed:,} shortened to whole turns), max {max_tokens} tokens')
    convs += identity_conversations(n_identity)
    random.Random(42).shuffle(convs)
    with open(out, 'w', encoding='utf-8') as f:
        for c in convs:
            f.write(json.dumps({'conversations': c}, ensure_ascii=False) + '\n')
    print(f'wrote {out}: {n_smol:,} smol-smoltalk + {n_identity:,} identity conversations')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('step', choices=['tok_corpus', 'pretrain', 'sft'])
    p.add_argument('--max_chunks', type=int, default=1_600_000)
    a = p.parse_args()
    d = os.path.join(ROOT, 'dataset')
    if a.step == 'tok_corpus': build_tok_corpus(os.path.join(d, 'tok_corpus_en.jsonl'))
    elif a.step == 'pretrain': build_pretrain(os.path.join(d, 'pretrain_en.jsonl'), a.max_chunks)
    else: build_sft(os.path.join(d, 'sft_en.jsonl'))
