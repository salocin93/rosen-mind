"""
Synthetic SFT data from the local Qwen3.6-35B-A3B (llama-swap, OpenAI-compatible API on :9292) -> data/synthetic/*.jsonl

  docker start llama-swap
  export PC_API_KEY=$(docker inspect llama-swap --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^PC_API_KEY=//p')
  python v2/scripts/gen_synthetic.py identity --calls 150     # ~1500 identity Q&A
  python v2/scripts/gen_synthetic.py general --calls 300      # ~2400 short, clear Q&A over many topics
  docker stop llama-swap                                      # free the GPU again

Needs the GPU (the 35B keeps part of its experts in VRAM), so run it when no training job is running.
Resumable: finished call ids are recorded in data/synthetic/<task>.done; rerunning skips them.
"""
import argparse
import concurrent.futures as cf
import json
import os
import random
import re
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import CREATOR, NAME, V2

OUT = os.path.join(V2, 'data', 'synthetic')
FACTS = f"""- Name: {NAME} (version 2). Creator: {CREATOR}, an individual (not a company).
- A very small English language model with about 108 million parameters, trained from scratch by {CREATOR}
  on a single NVIDIA RTX 4060 Ti (8 GB), using the open-source MiniMind training code.
- Trained on English educational web text (FineWeb-Edu), synthetic textbooks, math web pages and Python code,
  then fine-tuned on chat conversations.
- It has no internet access, no memory between conversations, no real-time information, and it can make mistakes,
  especially on facts, math and long reasoning. It is not ChatGPT, Claude, Llama, Qwen, Gemini or MiniMind."""

STYLES = ['casual and friendly', 'curious child', 'skeptical user who doubts the answer', 'formal and polite', 'very short messages with typos',
          'developer asking technical questions', 'someone testing whether it is ChatGPT', 'student', 'non-native English speaker',
          'user asking follow-up questions in a multi-turn chat', 'user asking about its abilities and limits', 'philosophical questions about being an AI']
TOPICS = """physics; chemistry; biology; human body and health; astronomy and space; earth science and weather; ecology and animals; plants;
world history; ancient civilizations; geography; economics basics; personal finance; cooking and food; nutrition; sports; music; art and painting;
literature and famous books; English grammar; vocabulary and word meanings; writing emails; writing short stories; poetry; study tips; productivity;
job interviews; communication skills; psychology basics; philosophy basics; ethics; computers and how they work; the internet; Python programming;
algorithms; data structures; databases and SQL; web development; machine learning basics; arithmetic word problems; fractions and percentages;
algebra; geometry; probability; statistics; unit conversions; logic puzzles; everyday life advice; travel tips; home repair; gardening;
pets; first aid basics; environment and climate; energy; transportation; inventions and inventors; famous scientists; mathematics history;
space exploration; oceans; dinosaurs; volcanoes and earthquakes; electricity; magnets; light and color; sound; materials; recycling;
time management; mindfulness; exercise; sleep; languages of the world; culture and holidays; government and civics; law basics;
critical thinking; riddles; trivia; comparing two things; step-by-step how-to instructions; explaining to a five-year-old; summarizing a paragraph;
rewriting text more simply; translating idioms into plain English; brainstorming ideas; giving recommendations""".replace('\n', ' ').split(';')
TOPICS = [t.strip() for t in TOPICS if t.strip()]

PROMPTS = {
    'identity': lambda rng: (
        f"You are writing training data for a tiny AI assistant. Facts about the assistant:\n{FACTS}\n\n"
        f"Write 10 different short conversations where a user ({rng.choice(STYLES)}) talks with the assistant about the assistant itself "
        f"(name, creator, what it is, size, training, abilities, limits, or whether it is some other AI). Some conversations may have 2 turns. "
        f"The assistant answers truthfully using ONLY the facts above, in 1-3 friendly sentences, never inventing other facts. "
        f"Vary the wording a lot. Output ONLY a JSON array; each element is a list of messages "
        f'like [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}].'),
    'general': lambda rng: (
        f"You are writing training data for a tiny AI assistant (a 100M-parameter model that can only learn simple, clear patterns).\n"
        f"Topic: {rng.choice(TOPICS)}.\n"
        f"Write 8 different questions a real user might ask about this topic, each with an excellent answer. Rules for answers: correct, "
        f"clear, simple English; 2-6 sentences or a short numbered list; no markdown headings; no tables; no apologies or disclaimers. "
        f"Mix easy and medium difficulty, and vary the question types (how, why, what, compare, give an example, step by step). "
        f"Output ONLY a JSON array; each element is a list of messages "
        f'like [{{"role":"user","content":"..."}},{{"role":"assistant","content":"..."}}].'),
}


def call(base, key, model, prompt, temperature, timeout=600):
    body = {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'temperature': temperature, 'max_tokens': 4000,
            'chat_template_kwargs': {'enable_thinking': False}}
    req = urllib.request.Request(base.rstrip('/') + '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {key}'})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())['choices'][0]['message']['content']


def parse(text):
    text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.M).strip()
    start, end = text.find('['), text.rfind(']')
    data = json.loads(text[start:end + 1])
    convs = []
    for conv in data:
        if isinstance(conv, dict): conv = conv.get('messages') or conv.get('conversation') or []
        msgs = [{'role': m['role'], 'content': str(m['content']).strip()} for m in conv
                if isinstance(m, dict) and m.get('role') in ('user', 'assistant') and str(m.get('content', '')).strip()]
        if len(msgs) >= 2 and msgs[0]['role'] == 'user' and msgs[-1]['role'] == 'assistant' and \
                all(msgs[i]['role'] != msgs[i + 1]['role'] for i in range(len(msgs) - 1)):
            convs.append(msgs)
    return convs


def main():
    p = argparse.ArgumentParser()
    p.add_argument('task', choices=list(PROMPTS))
    p.add_argument('--calls', type=int, default=150)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--base_url', default='http://localhost:9292/v1')
    p.add_argument('--model', default='qwen3.6-35b-a3b')
    p.add_argument('--temperature', type=float, default=0.8)
    a = p.parse_args()
    key = os.environ.get('PC_API_KEY') or sys.exit('set PC_API_KEY (see the docstring)')
    os.makedirs(OUT, exist_ok=True)
    out_path, done_path = os.path.join(OUT, f'{a.task}.jsonl'), os.path.join(OUT, f'{a.task}.done')
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    todo = [i for i in range(a.calls) if str(i) not in done]
    print(f'{a.task}: {len(todo)} calls to do ({len(done)} already done)', flush=True)
    lock, n_conv, t0 = threading.Lock(), [0], time.time()

    def work(i):
        rng = random.Random(f'{a.task}-{i}')
        for attempt in range(3):
            try:
                convs = parse(call(a.base_url, key, a.model, PROMPTS[a.task](rng), a.temperature))
                break
            except Exception as e:
                err = e; time.sleep(5)
        else:
            print(f'  call {i} failed: {str(err)[:150]}', flush=True); return
        if a.task == 'identity':  # drop replies that claim a wrong creator ("made by OpenAI"), keep denials ("not made by OpenAI")
            wrong = re.compile(r'\b(OpenAI|Anthropic|Alibaba|Meta|Google|DeepMind|jingyaogong)\b')
            denial = re.compile(r"\b(not|no|never|neither|nor)\b|n't", re.I)
            convs = [c for c in convs if all(not wrong.search(m['content']) or denial.search(m['content'])
                                             for m in c if m['role'] == 'assistant')]
        with lock:
            with open(out_path, 'a') as f:
                for c in convs: f.write(json.dumps({'conversations': c}, ensure_ascii=False) + '\n')
            with open(done_path, 'a') as f: f.write(f'{i}\n')
            n_conv[0] += len(convs)
            done.add(str(i))
            finished = len(done) - (a.calls - len(todo))
            print(f'  call {i} ok: {len(convs)} conversations (total new {n_conv[0]}), {(time.time() - t0) / max(1, finished):.0f}s/call', flush=True)

    with cf.ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, todo))
    print(f'done: {n_conv[0]} new conversations in {(time.time() - t0) / 60:.1f} min -> {out_path}')


if __name__ == '__main__':
    main()
