"""Freeze 1000 GenFace training and400 validation tensors for private Kaggle trials."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.pilot_deps'),str(ROOT/'src'),str(ROOT/'tools')]
import numpy as np
import torch
from mfvlr_research.pilot_datasets import dataset_for_config
from test_split_mask_encoder_pilot import balanced_subset


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--train-count',type=int,default=1000)
    parser.add_argument('--train-manifest',type=Path)
    parser.add_argument('--dataset-id',default='nha4von/mfvlr-genface-fd-pilot-20261026')
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    parser.add_argument('--fd-weights',type=float,nargs='+',default=[.1,1.])
    args=parser.parse_args()
    if args.output.exists(): parser.error('Choose a fresh output directory')
    args.output.mkdir(parents=True)
    config=json.loads((ROOT/'configs/diffae_pilot_skip_detach.json').read_text())
    config['model']['reconstruction_activation']='identity'
    config['loss_weights'].update(ar=1.,fl=1.,fd=1.)
    if args.train_manifest:
        config['manifests']['train']=str(args.train_manifest.resolve())
    if args.train_count<1000 or args.train_count%4: parser.error('train-count must be >=1000 and divisible by4')
    hashes, ids, audit={}, {}, {}
    for split,count in (('train',args.train_count),('val',400)):
        dataset=dataset_for_config(ROOT,config,split,limit=args.train_count if split=='train' else None)
        images,masks,labels,sample_ids,content=[],[],[],[],[]
        groups=[[i for i,row in enumerate(dataset.rows) if row['label']==cls] for cls in (0,1)]
        assert all(len(g)>=count//2 for g in groups)
        selected=[[g[i] for i in np.linspace(0,len(g)-1,count//2,dtype=int)] for g in groups]
        for number,index in enumerate([i for pair in zip(*selected) for i in pair],1):
            sample=dataset[index]
            rgb=(sample['image']*255).round().byte().numpy()
            mask=sample['mask'].byte().numpy()
            assert torch.equal(torch.from_numpy(rgb).float().div_(255),sample['image'])
            images.append(rgb); masks.append(mask); labels.append(sample['label']); sample_ids.append(sample['sample_id'])
            content.append(hashlib.sha256(rgb.tobytes()).hexdigest())
            if number%100==0: print(f'{split}: checked {number}/{count}',flush=True)
        assert len(set(sample_ids))==len(set(content))==count
        path=args.output/(split+'.npz')
        np.savez_compressed(path,images=np.stack(images),masks=np.stack(masks),
                            labels=np.array(labels),sample_ids=np.array(sample_ids))
        hashes[split]=hashlib.sha256(path.read_bytes()).hexdigest()
        ids[split]=sample_ids
        audit[split]=dict(count=count,real=labels.count(0),Diffae=labels.count(1),
            sample_ids=sample_ids,postresize_rgb_hashes=content,npz_sha256=hashes[split],
            original_manifest_sha256=hashlib.sha256((ROOT/config['manifests'][split]).read_bytes()).hexdigest())
    assert not(set(ids['train']) & set(ids['val']))
    assert not(set(audit['train']['postresize_rgb_hashes']) & set(audit['val']['postresize_rgb_hashes']))
    original16=json.loads((ROOT/'artifacts/kaggle_loss_overfit_20261026/package_001/package_manifest.json').read_text())['sample_ids']
    assert set(original16)<=set(ids['train'])
    original1000=json.loads((ROOT/'artifacts/genface_fd_validation_kaggle_20261026/package_001/data_audit.json').read_text())
    assert set(original1000['splits']['train']['sample_ids'])<=set(ids['train'])
    assert ids['val']==original1000['splits']['val']['sample_ids']
    assert audit['val']['postresize_rgb_hashes']==original1000['splits']['val']['postresize_rgb_hashes']
    portable={key:config[key] for key in ('model','loss_weights')}
    (args.output/'config.json').write_text(json.dumps(portable,indent=2))
    (args.output/'data_audit.json').write_text(json.dumps(dict(splits=audit,test_used=False,
        exact_train_val_overlap=False,subject_disjoint_verified=False,original16_nested_in_train=True),indent=2))
    tools=['run_fd_validation_pilot.py','run_loss_overfit.py','run_fd_weight_ablation.py',
           'measure_ar_fl_gradients.py','pilot_checkpointing.py','test_ad_overfit.py',
           'export_ad_reconstructions.py','test_split_mask_encoder_pilot.py']
    files=[*ROOT.joinpath('src/mfvlr').glob('*.py'),*ROOT.joinpath('src/mfvlr').glob('*.json'),
           *ROOT.joinpath('src/mfvlr_research').glob('pilot_*.py'),ROOT/'src/mfvlr_research/__init__.py',
           *(ROOT/'tools'/name for name in tools)]
    source={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (args.output/'source_sha256.json').write_text(json.dumps(source,indent=2))
    upload=args.output/'dataset_upload';upload.mkdir()
    bundle=upload/'mfvlr_fd_validation_bundle.bin'
    with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path,path.relative_to(ROOT).as_posix())
        for name in ('train.npz','val.npz','config.json','data_audit.json','source_sha256.json'):
            archive.write(args.output/name,'inputs/'+name)
    metadata=dict(id=args.dataset_id,
        title=f'MFVLR GenFace FD train{args.train_count} validation400',licenses=[dict(name='other')],
        description=f'Private fixed GenFace real/Diffae pilot:{args.train_count} train+400 validation, author supplied GT, required source. No test images or credentials.')
    (upload/'dataset-metadata.json').write_text(json.dumps(metadata,indent=2))
    result=dict(dataset=metadata['id'],privacy='private',train_count=args.train_count,val_count=400,test_used=False,
        exact_train_val_overlap=False,subject_disjoint_verified=False,original16_nested_in_train=True,
        npz_sha256=hashes,bundle_bytes=bundle.stat().st_size,
        bundle_sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(),source_sha256=source,
        planned_trials=dict(fd_weights=args.fd_weights,seeds=args.seeds,epochs=args.epochs,
            updates_each=args.train_count//4*args.epochs,
            total_updates=args.train_count//4*args.epochs*len(args.seeds)*len(args.fd_weights)))
    (args.output/'package_manifest.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2),flush=True)


if __name__=='__main__': main()
