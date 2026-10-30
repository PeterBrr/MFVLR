"""Evaluate the frozen epoch30 visual checkpoint on original GenFace test; no tuning."""
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027'

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    out=ART/'test_001'
    assert not out.exists(),'Use a fresh evaluation directory'
    out.mkdir(parents=True)
    package=ART/'package_001'
    manifest=json.loads((package/'package_manifest.json').read_text())
    bundle=package/'dataset_upload/mfvlr_fd_validation_bundle.bin'
    assert digest(bundle)==manifest['bundle_sha256']
    frozen=out/'frozen_source';frozen.mkdir()
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if name.startswith(('src/','tools/')):
                z.extract(name,frozen)
    source=json.loads((package/'source_sha256.json').read_text())
    assert all(digest(frozen/p)==h for p,h in source.items())
    sys.path[:0]=[str(ROOT/'.pilot_deps'),str(frozen/'src'),str(frozen/'tools')]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    import torch
    from mfvlr_research.pilot_data import PilotFaces
    import run_fd_validation_pilot as runner
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    device='cuda' if torch.cuda.is_available() else 'cpu'
    print(json.dumps(dict(event='environment',device=device,torch=torch.__version__,
        gpu=torch.cuda.get_device_name() if device=='cuda' else None)),flush=True)
    path=ART/'downloaded/fd_validation_results/seed_0_fd_0p1/final.pt'
    checkpoint=torch.load(path,map_location='cpu',weights_only=False)
    report=json.loads((ART/'downloaded/fd_validation_results/report.json').read_text())
    assert checkpoint['schema']=='mfvlr-fd-validation-pilot-v1' and checkpoint['step']==37500 and checkpoint['seed']==0
    assert checkpoint['provenance']['source']==report['source_sha256']
    for p,h in checkpoint['provenance']['source'].items():assert digest(frozen/p)==h
    assert checkpoint['provenance']['data']==report['data_sha256']
    assert checkpoint['loss_weights']==dict(ar=1.,fl=1.,fd=.1)
    audit=json.loads((package/'data_audit.json').read_text())
    known_ids=set(report['train_ids'])|set(report['val_ids'])
    known_hashes=set(audit['splits']['train']['postresize_rgb_hashes'])|set(audit['splits']['val']['postresize_rgb_hashes'])
    def load(split):
        ds=PilotFaces(ROOT/f'artifacts/diffae_pilot/manifests/{split}.jsonl')
        if split=='val':
            by_id={row['sample_id']:row for row in ds.rows}
            ds.rows=[by_id[sid] for sid in report['val_ids']]
        else:
            groups=[[row for row in ds.rows if row['label']==c] for c in (0,1)]
            assert len(groups[0])==len(groups[1])==200
            ds.rows=[r for pair in zip(*groups) for r in pair]
        samples=[ds[i] for i in range(len(ds))]
        if split=='test':
            hashes=[hashlib.sha256((s['image']*255).round().byte().numpy().tobytes()).hexdigest() for s in samples]
            ids=[s['sample_id'] for s in samples]
            assert len(set(ids))==len(set(hashes))==400
            assert not (set(ids)&known_ids),'Test ID overlap'
            assert not (set(hashes)&known_hashes),'Test resized content overlap'
        print(json.dumps(dict(event='data_checked',split=split,count=len(samples),manifest_sha256=ds.digest)),flush=True)
        return samples,ds.digest
    model=runner.seeded_model(checkpoint['config'],0)
    model.load_state_dict(checkpoint['model'],strict=True)
    model.to(device).eval()
    weights=torch.tensor([1.,4.],device=device)
    validation,_=load('val')
    local_val,_=runner.evaluate(model,validation,device,weights)
    expected=next(iter(report['trials'].values()))['evaluations'][-1]['val']
    parity=dict(auc_delta=local_val['auc']-expected['auc'],
        iou_delta=local_val['fake_foreground_iou_macro']-expected['fake_foreground_iou_macro'],
        mse_delta=local_val['losses']['ar']-expected['losses']['ar'])
    print(json.dumps(dict(event='validation_parity',**parity)),flush=True)
    assert abs(parity['auc_delta'])<.001 and abs(parity['iou_delta'])<.001 and abs(parity['mse_delta'])<.00001,'Cross-environment validation mismatch'
    del validation
    samples,test_digest=load('test')
    result,preview=runner.evaluate(model,samples,device,weights)
    runner.base.export_preview(out,preview)
    result.update(split='test',seed=0,epoch=30,step=37500,checkpoint_sha256=digest(path),
        manifest_sha256=test_digest,subject_disjoint_verified=False,exact_test_train_val_content_overlap=False,
        validation_parity=parity,source_verified=True,threshold_policy='Fixed detection0.5 and mask argmax; no test tuning',
        environment=dict(torch=torch.__version__,device=device),language_enabled=False)
    runner.base.write_json(out/'test_metrics.json',result)
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
