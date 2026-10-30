"""Summarize the continuously streamed expanded GenFace pilot log."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
folder=ROOT/'artifacts/genface_fd_validation_kaggle_20261026'
trials={}
overall='RUNNING'
last=None
for line in (folder/'remote_live.log').read_text(encoding='utf-8').splitlines():
    try: row=json.loads(line)
    except ValueError: continue
    last=row
    if row.get('event')=='experiment_complete': overall=row['status']
    name=row.get('trial')
    if not name: continue
    t=trials.setdefault(name,{})
    if row['event']=='trial_start': t.update(status='STARTING',seed=row['seed'],lambda_fd=row['lambda_fd'])
    elif row['event']=='train': t.update(status='RUNNING',step=row['step'],total=row['total'],epoch=row['epoch'],losses=row['losses'])
    elif row['event']=='metrics':
        t['last_evaluation_step']=row['step']
        for split in ('train','val'):
            m=row[split]
            t[split]=dict(count=m['count'],auc=m['auc'],iou=m['fake_foreground_iou_macro'],
                mse=m['losses']['ar'],precision=m['pixel_precision'],recall=m['pixel_recall'],
                real_fp=m['real_false_positive_area'])
    elif row['event']=='trial_complete': t.update(status='COMPLETED',step=row['step'])
    elif row['event']=='trial_failed': t.update(status='FAILED',error=row['error'])
snapshot=dict(status=overall,trials=trials,last_event=last)
(folder/'live_status.json').write_text(json.dumps(snapshot,indent=2),encoding='utf-8')
print(json.dumps(snapshot,indent=2))
