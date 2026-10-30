"""Retrieve only FD experiment metrics/logs/previews, then verify the completed run."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/kaggle_current_deps'))
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()
folder = ROOT / 'artifacts/genface_fd_weight_kaggle_20261026/downloaded'
folder.mkdir(parents=True, exist_ok=True)
files, _ = api.kernels_output('nha4von/mfvlr-genface-fd-weights-20261026-001', str(folder),
    file_pattern=r'fd_weight_results/.*(report\.json|progress\.json|preview\.png|preview\.npz|live_log\.jsonl)$',
    quiet=True)
path = folder / 'fd_weight_results/report.json'
report = json.loads(path.read_text())
assert report['status'] == 'COMPLETED' and report['source_unchanged']
assert len(report['trials']) == 4 and len(report['sample_ids']) == 16
assert len({t['initial_state_sha256'] for t in report['trials'].values()}) == 1
rows = []
for name, trial in report['trials'].items():
    assert trial['status'] == 'COMPLETED' and len(trial['history']) == 400
    metric = trial['evaluations'][-1]['training_set']
    assert trial['evaluations'][-1]['step'] == 400 and metric['count'] == 16
    gradients = {}
    for pair in ('ar_fl', 'fd_fl', 'fd_ar'):
        values = [g['pairs'][pair]['encoder_cnn'] for g in trial['gradients_final']]
        cosines = [v['cosine'] for v in values if v['cosine'] is not None]
        gradients[pair] = dict(cosine_mean=sum(cosines) / len(cosines) if cosines else None,
            negative_pairs=sum(x < 0 for x in cosines), measured_pairs=len(cosines),
            mean_weighted_left_norm=sum(v['weighted_left_norm'] for v in values) / len(values),
            mean_weighted_right_norm=sum(v['weighted_right_norm'] for v in values) / len(values))
    rows.append(dict(case=name, lambda_fd=trial['loss_weights']['fd'], mse=metric['losses']['ar'],
        iou=metric['fake_foreground_iou_macro'], dice=metric['fake_foreground_dice_macro'],
        auc=metric['auc'], accuracy=metric['accuracy'], precision=metric['pixel_precision'],
        recall=metric['pixel_recall'], real_fp=metric['real_false_positive_area'],
        final_gradient_cnn=gradients))
(folder.parent / 'comparison.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')
print(json.dumps(dict(downloaded_files=files, results=rows), indent=2))
