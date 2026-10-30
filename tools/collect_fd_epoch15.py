"""Download final report and compare fixed epochs 5/10/15 across three seeds."""
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/kaggle_current_deps'))
from kaggle.api.kaggle_api_extended import KaggleApi

root=ROOT/'artifacts/genface_fd_epoch15_kaggle_20261027'
folder=root/'downloaded'
folder.mkdir(parents=True,exist_ok=True)
api=KaggleApi();api.authenticate()
api.kernels_output('nha4von/mfvlr-genface-fd01-epoch15-20261027',str(folder),
    file_pattern=r'fd_validation_results/report\.json$',quiet=True)
r=json.loads((folder/'fd_validation_results/report.json').read_text())
assert r['status']=='COMPLETED' and r['source_unchanged'] and not r['test_used'] and not r['smoke']
assert len(r['trials'])==3 and len(r['train_ids'])==1000 and len(r['val_ids'])==400
assert r['epochs']==15 and r['different_seed_initializations_verified']
old=json.loads((ROOT/'artifacts/genface_fd_validation_kaggle_20261026/downloaded/fd_validation_results/report.json').read_text())
assert r['data_sha256']==old['data_sha256']
rows=[]
for name,t in r['trials'].items():
    assert t['status']=='COMPLETED' and len(t['history'])==3750 and len(t['evaluations'])==15
    assert t['loss_weights']==dict(ar=1.,fl=1.,fd=.1)
    prior=old['trials'][name]
    assert t['initial_state_sha256']==prior['initial_state_sha256']
    assert t['evaluations'][4]==prior['evaluations'][-1], 'Epoch5 did not reproduce exactly'
    for e in t['evaluations']:
        if e['epoch'] not in (5,10,15): continue
        v=e['val'];assert v['count']==400
        rows.append(dict(seed=t['seed'],epoch=e['epoch'],auc=v['auc'],iou=v['fake_foreground_iou_macro'],
            mse=v['losses']['ar'],precision=v['pixel_precision'],recall=v['pixel_recall'],
            real_fp=v['real_false_positive_area'],accuracy=v['accuracy'],
            train_iou=e['train']['fake_foreground_iou_macro']))
aggregate={}
for epoch in (5,10,15):
    selected=[x for x in rows if x['epoch']==epoch]
    aggregate[str(epoch)]={k:dict(mean=statistics.mean(x[k] for x in selected),
        sample_std=statistics.stdev(x[k] for x in selected))
        for k in ('auc','iou','mse','precision','recall','real_fp','accuracy','train_iou')}
result=dict(status='COMPLETED',epoch5_exact_reproduction=True,per_seed=rows,aggregate=aggregate,
    test_used=False,scope='Fixed epoch15 primary; epochs5/10 learning trajectory, three seeds')
(root/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2),flush=True)
