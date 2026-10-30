"""Print a compact snapshot of the streamed Kaggle FD experiment log."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
folder = ROOT / 'artifacts/genface_fd_weight_kaggle_20261026'
trials = {}
last_event = None
for line in (folder / 'remote_live.log').read_text(encoding='utf-8').splitlines():
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        continue
    last_event = row
    name = row.get('trial')
    if not name:
        continue
    trial = trials.setdefault(name, {})
    if row['event'] == 'train':
        trial.update(step=row['step'], loss=row['losses'])
    if row['event'] == 'trial_state':
        trial.update(status=row['status'], step=row['saved_step'], error=row.get('error'))
    if row['event'] == 'metrics':
        m = row['training_set']
        trial['metrics'] = dict(step=row['step'], iou=m['fake_foreground_iou_macro'],
            mse=m['losses']['ar'], auc=m['auc'], recall=m['pixel_recall'],
            precision=m['pixel_precision'], real_fp=m['real_false_positive_area'])
snapshot = dict(trials=trials, last_event=last_event)
(folder / 'live_status.json').write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
print(json.dumps(snapshot, indent=2))
