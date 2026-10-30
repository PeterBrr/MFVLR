"""Verify completed GenFace trials and export correctly labelled comparisons."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def display(array):
    if array.ndim == 2:
        return Image.fromarray(array.astype(np.uint8) * 255).convert('RGB')
    return Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))


def labelled_comparison(path, left_name, right_name, left, right, ids, labels):
    canvas = Image.new('RGB', (1344, 1120), 'white')
    draw = ImageDraw.Draw(canvas)
    for i in range(4):
        y = 280 * i
        draw.text((5, y + 5), f'{"REAL" if labels[i] == 0 else "DIFFAE"} | {ids[i]}', fill='black')
        panels = [(left[i]['image'], 'Input'), (left[i]['target'], 'GT mask'),
                  (left[i]['reconstruction'], left_name + ': AD'),
                  (right[i]['reconstruction'], right_name + ': AD'),
                  (left[i]['prediction'], left_name + ': mask'),
                  (right[i]['prediction'], right_name + ': mask')]
        for j, (array, title) in enumerate(panels):
            draw.text((j * 224 + 5, y + 25), title, fill='black')
            canvas.paste(display(array), (j * 224, y + 48))
    canvas.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    report = json.loads((args.run / 'report.json').read_text(encoding='utf-8'))
    if report['status'] not in ('COMPLETED', 'COMPLETED_WITH_FAILURES'):
        raise ValueError('Training is still running')
    assert report['dataset'] == 'GenFace real + Diffae only' and not report['test_used']
    assert len(set(report['train_ids'])) == len(report['train_ids']) == 128
    assert len(set(report['val_ids'])) == len(report['val_ids']) == 128
    assert not (set(report['train_ids']) & set(report['val_ids']))
    rows, cases = [], {}
    for name, trial in report['trials'].items():
        if trial['status'] != 'COMPLETED':
            continue
        assert len(trial['history']) == report['updates_per_trial'] == 100
        assert trial['evaluations'][-1]['step'] == 100
        metric = trial['evaluations'][-1]['val']
        assert metric['count'] == 128
        assert all(np.isfinite(value) for update in trial['history']
                   for value in update['losses'].values())
        pixel = metric['pixel_confusion']
        precision = pixel['tp'] / max(1, pixel['tp'] + pixel['fp'])
        recall = pixel['tp'] / max(1, pixel['tp'] + pixel['fn'])
        rows.append({'case': name, 'val_auc': metric['auc'], 'val_mse': metric['losses']['ar'],
                     'val_fake_iou': metric['fake_foreground_iou_macro'],
                     'val_real_fp_area': metric['real_false_positive_area'],
                     'val_global_miou': metric['pixel_miou_global'],
                     'val_pixel_precision': precision, 'val_pixel_recall': recall,
                     'val_accuracy_at_05': metric['accuracy'],
                     'train_auc': trial['final_train']['auc'],
                     'train_fake_iou': trial['final_train']['fake_foreground_iou_macro'],
                     'train_mse': trial['final_train']['losses']['ar'],
                     'parameters': trial['parameters'], 'seconds': trial['seconds'],
                     'peak_cuda_allocated_MiB': trial['peak_cuda_allocated_MiB']})
        if 'reconstruction_output' in metric:
            rgb = metric['reconstruction_output']
            rows[-1]['val_reconstruction_output'] = rgb
            rows[-1]['val_rgb_outside_fraction'] = rgb['outside_count'] / rgb['channel_value_count']
    assert report['source_unchanged']
    if report.get('experiment') == 'sigmoid':
        assert report['sigmoid_initial_weights_identical']
        bounded = report['trials']['G_sigmoid']['evaluations'][-1]['val']['reconstruction_output']
        assert bounded['minimum'] >= 0 and bounded['maximum'] <= 1 and bounded['outside_count'] == 0
    if all(report['trials'].get(name, {}).get('status') == 'COMPLETED'
           for name in ('E_language_legacy', 'F_language_corrected')):
        assert report['language_initial_weights_identical']
        assert report['trials']['E_language_legacy']['initial_state_sha256'] == \
            report['trials']['F_language_corrected']['initial_state_sha256']
    for path in args.run.glob('val_*_vs_*.npz'):
        left_name, right_name = path.stem.removeprefix('val_').split('_vs_')
        raw = np.load(path)
        pair = []
        for prefix, name in [('baseline', left_name), ('separate', right_name)]:
            samples = [{key: raw[f'{prefix}_{i}_{key}'] for key in
                        ('image', 'target', 'prediction', 'probability', 'reconstruction')} for i in range(4)]
            assert all(np.isfinite(a).all() for sample in samples for a in sample.values())
            if name in cases:
                for a, b in zip(cases[name], samples):
                    for key in a:
                        np.testing.assert_array_equal(a[key], b[key])
            cases[name] = samples
            pair.append(samples)
        selected = [0, 1, len(report['val_ids']) - 2, len(report['val_ids']) - 1]
        ids = [report['val_ids'][i] for i in selected]
        labels = [0, 1, 0, 1]
        # Replace generic labels inherited from the older preview helper.
        labelled_comparison(path.with_suffix('.png'), left_name, right_name, *pair, ids, labels)
    if 'D_manual_fl' in cases and 'F_language_corrected' in cases:
        labelled_comparison(args.run / 'val_D_vs_F.png', 'D_manual_fl', 'F_language_corrected',
                            cases['D_manual_fl'], cases['F_language_corrected'], ids, labels)
    for key, title, first_target in [('reconstruction', 'AD', False), ('prediction', 'mask', True)]:
        names = [row['case'] for row in rows if row['case'] in cases]
        panel_count = 1 + int(first_target) + len(names)
        canvas = Image.new('RGB', (224 * panel_count, 1120), 'white')
        draw = ImageDraw.Draw(canvas)
        for i in range(4):
            y = 280 * i
            sample = cases[names[0]][i]
            draw.text((5, y + 5), ids[i], fill='black')
            panels = [(sample['image'], 'Input')]
            if first_target:
                panels.append((sample['target'], 'GT mask'))
            panels += [(cases[name][i][key], name) for name in names]
            for j, (array, label) in enumerate(panels):
                draw.text((j * 224 + 5, y + 25), label, fill='black')
                canvas.paste(display(array), (j * 224, y + 48))
        canvas.save(args.run / f'all_{title}.png')
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    letters = [row['case'][0] for row in rows]
    for ax, key, title, scale in zip(axes.flat,
            ('val_auc', 'val_mse', 'val_fake_iou', 'val_real_fp_area'),
            ('Validation AUC (%)', 'Validation reconstruction MSE',
             'Fake-only foreground IoU (%)', 'False-positive area on real (%)'), (100, 1, 100, 100)):
        ax.bar(letters, [row[key] * scale for row in rows], color=[f'C{i}' for i in range(len(rows))])
        ax.set_title(title)
        ax.grid(axis='y', alpha=.2)
        if key == 'val_auc':
            ax.set_ylim(0, 100)
    fig.suptitle('GenFace only: 128 train / 128 validation, seed0, 100 updates each')
    fig.savefig(args.run / 'metrics_comparison.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for row in rows:
        trial = report['trials'][row['case']]
        steps = [update['step'] for update in trial['history']]
        for ax, key in zip(axes, ('ar', 'fd', 'fl')):
            ax.plot(steps, [update['losses'][key] for update in trial['history']],
                    label=row['case'][0], alpha=.8, linewidth=1)
            ax.set_title(f'Raw training {key.upper()} loss')
            ax.set_xlabel('Optimizer update')
            ax.grid(alpha=.2)
        axes[0].set_yscale('log')
    axes[-1].legend(ncol=2)
    fig.savefig(args.run / 'training_losses.png', dpi=150)
    plt.close(fig)
    lines = ['# GenFace controlled ablation — final update 100', '',
             '| Case | AUC % | MSE | Fake IoU % | Real FP area % | Global mIoU % |',
             '|---|---:|---:|---:|---:|---:|']
    for row in rows:
        lines.append(f'| {row["case"]} | {100*row["val_auc"]:.3f} | {row["val_mse"]:.6f} | '
                     f'{100*row["val_fake_iou"]:.4f} | {100*row["val_real_fp_area"]:.4f} | {100*row["val_global_miou"]:.3f} |')
    comparison_note = ('D/G differ only in RGB output activation, with matched initial weights and training schedule.'
                       if report.get('experiment') == 'sigmoid' else
                       'E/F compare the three language choices jointly, with matched initial weights and loss coefficients.')
    lines += ['', 'Single-seed pilot; no test or FF++ data. Low mask FP alone is not good localization.',
              comparison_note,
              'PNG displays clip reconstruction only; raw arrays remain unchanged.',
              'Preview titles corrected to case names; numerical outputs were not modified.']
    (args.run / 'summary.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    (args.run / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
