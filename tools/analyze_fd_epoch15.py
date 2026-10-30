"""Analyze stored epoch metrics and limited gradient probes; no training or test use."""
import json
from pathlib import Path
import statistics as st
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/genface_fd_epoch15_kaggle_20261027'
r=json.loads((OUT/'downloaded/fd_validation_results/report.json').read_text())
assert r['status']=='COMPLETED' and not r['test_used']
summary={'scope':'Validation400; initial/final gradient probes on four fixed training pairs per seed',
         'seeds':{},'gradient_aggregate':{}}
for name,t in r['trials'].items():
    v=t['evaluations'][-1]['val']; c=v['pixel_confusion']
    tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
    summary['seeds'][name]=dict(pixel_confusion=c,
        gt_foreground_fraction=(tp+fn)/(tp+fp+fn+tn),
        predicted_foreground_fraction=(tp+fp)/(tp+fp+fn+tn),
        predicted_to_gt_area=(tp+fp)/(tp+fn),fp_to_fn=fp/fn,
        foreground_iou_global=tp/(tp+fp+fn),
        all_background_accuracy=(fp+tn)/(tp+fp+fn+tn),
        final_real_false_positive_area=v['real_false_positive_area'])
    for phase in ('initial','final'):
        for pair in ('ar_fl','fd_fl','fd_ar'):
            for group in ('encoder_cnn','decoder_trunk','decoder_stage_1','decoder_stage_4'):
                entries=[p['pairs'][pair][group] for p in t['gradients_'+phase]]
                cos=[e['cosine'] for e in entries if e['cosine'] is not None]
                ratios=[e['weighted_right_norm']/e['weighted_left_norm'] for e in entries if e['weighted_left_norm']>0]
                summary['gradient_aggregate'].setdefault(f'{phase}/{pair}/{group}',[]).append(dict(
                    seed=t['seed'],cosine_mean=st.mean(cos) if cos else None,
                    negative=sum(c<0 for c in cos),count=len(cos),
                    right_over_left_norm_median=st.median(ratios) if ratios else None))
        for p in t['gradients_'+phase]:
            for pair in ('fd_fl','fd_ar'):
                for group in ('decoder_trunk','appearance_head','mask_head'):
                    assert p['pairs'][pair][group]['left_connected_tensors']==0
(OUT/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
fig,axs=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
for name,t in r['trials'].items():
    es=t['evaluations'];x=[e['epoch'] for e in es]
    for ax,key,title in zip(axs.flat,('iou','precision','recall','fp','mse','auc'),
        ('Fake IoU (%)','Foreground precision (%)','Foreground recall (%)','FP area on real (%)','AD MSE','Detection AUC (%)')):
        def metric(e,split):
            m=e[split]
            return m['losses']['ar'] if key=='mse' else 100*m[dict(iou='fake_foreground_iou_macro',precision='pixel_precision',recall='pixel_recall',fp='real_false_positive_area',auc='auc')[key]]
        line=ax.plot(x,[metric(e,'val') for e in es],label=f"seed {t['seed']} val")[0]
        ax.plot(x,[metric(e,'train') for e in es],linestyle='--',alpha=.5,color=line.get_color())
        ax.set(title=title,xlabel='Epoch');ax.grid(alpha=.2)
axs[0,0].legend(fontsize=8)
fig.suptitle('GenFace FD=0.1; solid validation, dashed train; no test')
fig.savefig(OUT/'analysis_learning_curves.png',dpi=150);plt.close(fig)
print(json.dumps(summary,indent=2))
