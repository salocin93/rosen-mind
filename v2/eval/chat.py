"""
Chat with / spot-check RosenMind weights.

  python v2/eval/chat.py --weights v2/out/sft_v2.pth                  # fixed English test prompts
  python v2/eval/chat.py --weights out/full_sft_768.pth --arch v1 -i  # interactive (empty line quits)
  python v2/eval/chat.py --weights v2/out/pretrain_v2.pth --raw       # base model: plain text continuation
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from common import BOS, EOS, MODEL_V1, MODEL_V2, build_model, load_tokenizer

PROMPTS = [
    'Who are you, and who created you?',
    'Are you ChatGPT?',
    'Why is the sky blue?',
    'Explain photosynthesis in simple terms.',
    'Write a Python function that checks whether a number is prime.',
    'What is 15% of 80?',
    'Give me three tips for sleeping better.',
    'What is the capital of France, and what is it famous for?',
]
RAW_PROMPTS = ['The water cycle is', 'In 1492, Christopher Columbus', 'def fibonacci(n):', 'The main causes of the French Revolution were']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', required=True)
    p.add_argument('--arch', choices=['v1', 'v2'], default='v2')
    p.add_argument('-i', '--interactive', action='store_true')
    p.add_argument('--raw', action='store_true', help='no chat template (for pretrained base models)')
    p.add_argument('--max_new_tokens', type=int, default=300)
    p.add_argument('--temperature', type=float, default=0.7)
    p.add_argument('--top_p', type=float, default=0.9)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    tok = load_tokenizer()
    model = build_model(MODEL_V1 if a.arch == 'v1' else MODEL_V2, a.weights, a.device).eval()
    history = []

    def reply(prompt):
        nonlocal history
        if a.raw:
            text = tok.bos_token + prompt
        else:
            history = (history if a.interactive else []) + [{'role': 'user', 'content': prompt}]
            text = tok.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
        x = tok(text, return_tensors='pt', add_special_tokens=False).to(a.device)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.device.startswith('cuda')):
            out = model.generate(x['input_ids'], attention_mask=x['attention_mask'], max_new_tokens=a.max_new_tokens,
                                 temperature=a.temperature, top_p=a.top_p, eos_token_id=EOS, repetition_penalty=1.05)
        ans = tok.decode(out[0][x['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        if not a.raw: history.append({'role': 'assistant', 'content': ans})
        return ans

    if a.interactive:
        while (q := input('you> ').strip()):
            print('bot>', reply(q), '\n')
    else:
        for q in (RAW_PROMPTS if a.raw else PROMPTS):
            print(f'### {q}\n{reply(q)}\n', flush=True)


if __name__ == '__main__':
    main()
