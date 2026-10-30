"""Generate verified training-set comparisons for the completed Kaggle FD trials."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/genface_fd_weight_kaggle_20261026'
DATA = FOLDER / 'downloaded/fd_weight_results'
report = json.loads((DATA / 'report.json').read_text())
rows = json.loads((FOLDER / 'comparison.json').read_text())
assert report['status'] == 'COMPLETED'
names = [row['case'] for row in rows]
previews = {name: np.load(DATA / name / 'preview.npz', allow_pickle=False) for name in names}
for name in names[1:]:
    for index in range(4):
        for key in ('image', 'target'):
            np.testing.assert_array_equal(previews[name][f'{index}_{key}'], previews[names[0]][f'{index}_{key}'])


def display(array):
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).convert('RGB')


for key, gt in [('reconstruction', False), ('prediction', True)]:
    sheet = Image.new('RGB', (224 * (1 + int(gt) + len(names)), 1120), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, sample_index in enumerate((0, 1, 14, 15)):
        y = 280 * i
        draw.text((5, y + 5), report['sample_ids'][sample_index], fill='black')
        panels = [(previews[names[0]][f'{i}_image'], 'Input')]
        if gt:
            panels.append((previews[names[0]][f'{i}_target'], 'GT'))
        panels += [(previews[row['case']][f'{i}_{key}'], f"lambda FD = {row['lambda_fd']}") for row in rows]
        for j, (array, title) in enumerate(panels):
            draw.text((224 * j + 5, y + 25), title, fill='black')
            sheet.paste(display(array), (224 * j, y + 48))
    sheet.save(FOLDER / ('all_AD.png' if key == 'reconstruction' else 'all_masks.png'))

labels = [str(row['lambda_fd']) for row in rows]
fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
for ax, key, title, scale in zip(axes.flat, ('iou', 'mse', 'precision', 'real_fp'),
    ('Fake foreground IoU (%)', 'Reconstruction MSE', 'Foreground precision (%)', 'FP area on real (%)'),
    (100, 1, 100, 100)):
    ax.bar(labels, [scale * row[key] for row in rows])
    ax.set_title(title)
    ax.set_xlabel('lambda FD')
    ax.grid(axis='y', alpha=.2)
fig.suptitle('16 GenFace TRAIN images; final update400; AR=FL=1; one seed')
fig.savefig(FOLDER / 'metrics.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for row in rows:
    evaluations = report['trials'][row['case']]['evaluations']
    steps = [e['step'] for e in evaluations]
    metrics = [e['training_set'] for e in evaluations]
    for ax, values, title in zip(axes,
        ([100*m['fake_foreground_iou_macro'] for m in metrics], [m['losses']['ar'] for m in metrics],
         [m['losses']['fd'] for m in metrics]), ('Train fake IoU (%)', 'Train reconstruction MSE', 'Train raw FD CE')):
        ax.plot(steps, values, marker='.', label=str(row['lambda_fd']))
        ax.set_title(title)
        ax.set_xlabel('Optimizer update')
        ax.grid(alpha=.2)
axes[-1].set_yscale('log')
axes[-1].legend(title='lambda FD')
fig.savefig(FOLDER / 'learning_curves.png', dpi=150)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, pair in zip(axes, ('fd_fl', 'fd_ar')):
    for i, row in enumerate(rows):
        probe = report['trials'][row['case']]['gradients_final']
        values = [p['pairs'][pair]['encoder_cnn']['cosine'] for p in probe]
        ax.scatter([i] * len(values), values, label=str(row['lambda_fd']))
        ax.plot([i-.15, i+.15], [np.mean(values)] * 2, color='black')
    ax.set_xticks(range(len(rows)), labels)
    ax.set_xlabel('lambda FD')
    ax.set_title(pair.upper().replace('_', ' / ') + ': final CNN gradient cosine')
    ax.axhline(0, color='grey', linewidth=1)
    ax.set_ylim(-1, 1)
    ax.grid(alpha=.2)
fig.savefig(FOLDER / 'gradient_cosines.png', dpi=150)
plt.close(fig)
print('Verified previews match; wrote all_AD/all_masks/metrics/learning_curves/gradient_cosines PNGs')
