"""Prints a comparison table of every result in v2/eval/results/ (v1 vs v2 stages)."""
import glob
import json
import os

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
COLS = [('val', 'mean', 'val loss'), ('hellaswag', 'acc_norm', 'HellaSwag'), ('arc_easy', 'acc', 'ARC-E'),
        ('arc_challenge', 'acc_norm', 'ARC-C'), ('piqa', 'acc_norm', 'PIQA')]
rows = [json.load(open(f)) | {'name': os.path.basename(f)[:-5]} for f in sorted(glob.glob(f'{R}/*.json'))]
print(f'{"model":24s}' + ''.join(f'{c[2]:>11s}' for c in COLS))
print(f'{"(chance)":24s}{"":>11s}{"25.0":>11s}{"25.0":>11s}{"25.0":>11s}{"50.0":>11s}')
for r in rows:
    cells = []
    for task, metric, _ in COLS:
        v = r.get(task, {}).get(metric)
        cells.append('-' if v is None else (f'{v:.3f}' if task == 'val' else f'{100 * v:.1f}'))
    print(f'{r["name"]:24s}' + ''.join(f'{c:>11s}' for c in cells))
