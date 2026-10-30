"""Keep the frozen4999 rows and replace the removed real duplicate from original train only."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
from prepare_diffae_pilot import index

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    out=ROOT/'artifacts/genface5000_epoch30_kaggle_20261027'
    out.mkdir(parents=True,exist_ok=True)
    destination=out/'train5000.jsonl'
    assert not destination.exists(), 'Manifest already frozen'
    rows=[json.loads(s) for s in (ROOT/'artifacts/diffae_pilot/manifests/train.jsonl').read_text().splitlines()]
    assert len(rows)==4999 and sum(r['label']==0 for r in rows)==2499
    forbidden={r['sample_id'] for r in rows};hashes={r['decoded_rgb_sha256'] for r in rows}
    small=set()
    for split in ('val','test'):
        for s in (ROOT/f'artifacts/diffae_pilot/manifests/{split}.jsonl').read_text().splitlines():
            r=json.loads(s);forbidden.add(r['sample_id']);hashes.add(r['decoded_rgb_sha256'])
            with Image.open(r['image']) as im:small.add(hashlib.sha256(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).tobytes()).hexdigest())
    data=Path('D:/BaiduNetdiskDownload')
    sources={name:(index(data/'real/real'/name/'images'),index(data/'real/real'/name/'masks',True)) for name in ('CelebA_HQ','FFHQ')}
    for line in (data/'train.txt').read_text(encoding='utf-8-sig').splitlines()[1:]:
        f=line.split()
        if len(f)!=5 or f[1:]!=['0']*4:continue
        matches=[(name,p,masks.get(p.stem.casefold(),[])) for name,(images,masks) in sources.items() for p in images.get(f[0].casefold(),[])]
        if len(matches)!=1:continue
        name,p,ms=matches[0];sid=f'{name}/{f[0]}'
        if sid in forbidden or len(ms)!=1:continue
        with Image.open(p) as im:
            size=im.size;rgb=np.asarray(im.convert('RGB'));h=hashlib.sha256(rgb.tobytes()).hexdigest()
            h224=hashlib.sha256(np.asarray(im.convert('RGB').resize((224,224),Image.Resampling.BILINEAR)).tobytes()).hexdigest()
        if h in hashes or h224 in small:continue
        with Image.open(ms[0]) as mask:
            if mask.size!=size or np.asarray(mask).any():continue
        replacement=dict(rows[0],sample_id=sid,image=str(p),mask=str(ms[0]),image_sha256=digest(p),
            mask_sha256=digest(ms[0]),decoded_rgb_sha256=h,image_size=list(size),source_collection=name)
        rows.append(replacement)
        break
    assert len(rows)==5000 and sum(r['label']==0 for r in rows)==2500
    destination.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    (out/'replacement_audit.json').write_text(json.dumps(dict(added_sample_id=replacement['sample_id'],
        source_split='original train.txt',train_annotation_sha256=digest(data/'train.txt'),
        reason='One exact real duplicate had been removed from the original5000; add one unique original-train real',
        test_images_packaged=False),indent=2),encoding='utf-8')
    print(destination,flush=True)

if __name__=='__main__':main()
