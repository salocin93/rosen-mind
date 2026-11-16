"""
Read the RosenMind demo's chat logs (v2/demo/logs/YYYY-MM-DD.jsonl).

  python v2/demo/logs.py              # stats per day + the last 10 conversations
  python v2/demo/logs.py -n 30        # last 30 conversations
  python v2/demo/logs.py --day 2026-11-16 --full    # one day, replies not shortened
  python v2/demo/logs.py --grep python              # conversations mentioning a word
"""
import argparse
import glob
import json
import os
from collections import Counter, defaultdict

LOG_DIR = os.environ.get('ROSENMIND_LOG_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs'))


def short(text, n):
    text = ' '.join(str(text).split())
    return text if len(text) <= n else text[:n - 1] + '…'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('-n', type=int, default=10, help='conversations to show')
    p.add_argument('--day', help='only this day (YYYY-MM-DD)')
    p.add_argument('--grep', help='only conversations containing this text (case-insensitive)')
    p.add_argument('--full', action='store_true', help='do not shorten messages and replies')
    a = p.parse_args()
    files = sorted(glob.glob(os.path.join(LOG_DIR, f'{a.day or "*"}.jsonl')))
    if not files: return print(f'no logs in {LOG_DIR}')
    recs = [json.loads(l) for f in files for l in open(f) if l.strip()]

    print(f'{"day":10}  {"messages":>8}  {"chats":>5}  {"visitors":>8}  {"median reply":>12}  problems')
    by_day = defaultdict(list)
    for r in recs: by_day[r['time'][:10]].append(r)
    for day, rs in sorted(by_day.items()):
        gens = sorted(r['gen_s'] for r in rs if 'gen_s' in r)
        problems = Counter(r['status'] for r in rs if r['status'] not in ('done', 'cut off'))
        print(f'{day:10}  {len(rs):8}  {len({(r["ip"], r.get("chat")) for r in rs}):5}  {len({r["ip"] for r in rs}):8}  '
              f'{(f"{gens[len(gens) // 2]:.1f}s" if gens else "-"):>12}  {", ".join(f"{k} {v}" for k, v in problems.items()) or "-"}')

    chats = defaultdict(list)
    for r in recs: chats[(r['ip'], r.get('chat') or r['time'])].append(r)
    chats = sorted(chats.values(), key=lambda rs: rs[0]['time'])
    if a.grep:
        g = a.grep.lower()
        chats = [rs for rs in chats if any(g in (r.get('message', '') + r.get('reply', '')).lower() for r in rs)]
    n = None if a.full else 300
    print(f'\n{len(chats)} conversations' + (f', showing the last {a.n}' if len(chats) > a.n else ''))
    for rs in chats[-a.n:]:
        print(f'\n── {rs[0]["time"].replace("T", " ")[:16]}  {rs[0]["ip"]}  {short(rs[0].get("agent", ""), 60)}')
        for r in rs:
            print(f'  you> {r["message"] if a.full else short(r["message"], n)}')
            extra = '' if r['status'] == 'done' else f'  [{r["status"]}]'
            if 'reply' in r: print(f'  bot> {r["reply"] if a.full else short(r["reply"], n)}{extra}')
            else: print(f'  bot>{extra}')


if __name__ == '__main__':
    main()
