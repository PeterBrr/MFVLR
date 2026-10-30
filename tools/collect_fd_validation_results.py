"""Retrieve the completed six-trial pilot and aggregate held-out validation by seed."""
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
from kaggle.api.kaggle_api_extended import KaggleApi

api=KaggleApi();api.authenticate()
folder=ROOT/'artifacts/genface_fd_validation_kaggle_20261026/downloaded'
folder.mkdir(parents=True,exist_ok=True)
files,_=api.kernels_output('nha4von/mfvlr-genface-fd-pilot-20261026-001',str(folder),
    file_pattern=r'fd_validation_results/.*(report\.json|preview\.png|preview\.npz|live_log\.jsonl)$',quiet=True)
report=json.loads((folder/'fd_validation_results/report.json').read_text())
assert report['status']=='COMPLETED' and report['source_unchanged'] and not report['test_used']
assert not report['smoke'] and len(report['trials'])==6
assert len(report['train_ids'])==1000 and len(report['val_ids'])==400
assert report['paired_initialization_verified'] and report['different_seed_initializations_verified']
rows=[]
for name,t in report['trials'].items():
    assert t['status']=='COMPLETED' and len(t['history'])==1250 and len(t['evaluations'])==5
    final=t['evaluations'][-1]
    assert final['epoch']==5 and final['step']==1250 and final['val']['count']==400
    metric=final['val']
    rows.append(dict(case=name,seed=t['seed'],lambda_fd=t['loss_weights']['fd'],
        val_auc=metric['auc'],val_iou=metric['fake_foreground_iou_macro'],val_mse=metric['losses']['ar'],
        val_precision=metric['pixel_precision'],val_recall=metric['pixel_recall'],
        val_real_fp=metric['real_false_positive_area'],
        train_auc=final['train']['auc'],train_iou=final['train']['fake_foreground_iou_macro']))
groups={}
keys=('val_auc','val_iou','val_mse','val_precision','val_recall','val_real_fp')
for weight in (.1,1.):
    selected=[r for r in rows if r['lambda_fd']==weight]
    assert len(selected)==3
    groups[str(weight)]={k:dict(mean=statistics.mean(r[k] for r in selected),
                               sample_std=statistics.stdev(r[k] for r in selected)) for k in keys}
deltas=[]
for seed in (0,1,2):
    a=next(r for r in rows if r['seed']==seed and r['lambda_fd']==.1)
    b=next(r for r in rows if r['seed']==seed and r['lambda_fd']==1.)
    deltas.append(dict(seed=seed,**{k:a[k]-b[k] for k in keys}))
summary=dict(per_run=rows,aggregate=groups,paired_deltas_fd0p1_minus_fd1=deltas,
             scope='Epoch5 primary; mean/sample SD over3 seeds; test unused')
(folder.parent/'comparison.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
