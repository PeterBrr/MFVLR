"""Read-only validation of recorded F2 input seals and promotion coverage."""
import json,sys,numpy as np
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b2.contract import load_fixture,tensor_hash,bundle
from mfvlr_research.d02b5.policy import CELLS,write
OUT=ROOT/'artifacts/phase2d2_d02b5'; records=[]
for cell in CELLS:
    folder=OUT/'cells'/cell/'attempt_01';result=json.loads((folder/'result.json').read_text())
    if result['execution']!='COMPLETE':raise RuntimeError('Incomplete cell')
    ids,v=load_fixture('F2');v=v.astype(np.float64 if cell.endswith('float64') else np.float32)
    views=[(ids,v),(ids[:1],v[:1]),(ids[1:],v[1:]),(ids[[1,0]],v[[1,0]])]
    for sample in (0,1):
        a=ids.copy();b=v.copy();a[sample,250]=1234;b[sample]+=.5;views.append((a,b))
    views.append((ids,v))
    if cell.endswith('float32'):views.extend([(ids[:1],v[:1]),(ids,v)])
    for i,(a,b) in enumerate(views):
        seal=json.loads((folder/f'forward_{i:02d}_seal.json').read_text())
        assert seal['input_ids_sha256']==tensor_hash(a) and seal['visual_sha256']==tensor_hash(b)
        assert seal['ids_shape']==list(a.shape) and seal['visual_shape']==list(b.shape)
    fp=json.loads((folder/'fp32_inventory.json').read_text())['parameters'];live=json.loads((folder/'live_inventory.json').read_text())['parameters']
    expected={r['name']:r['shape'] for r in bundle()['rosters'][result['variant']]}
    assert {r['name']:r['shape'] for r in live}==expected
    if cell.endswith('float64'):
        promotion=json.loads((folder/'promotion.json').read_text());assert promotion['fp32_source_unchanged']
        assert len(promotion['parameters'])==len(fp)==len(live)
        for p,a,b in zip(promotion['parameters'],fp,live):
            assert p['name']==a['name']==b['name'] and p['exact_cast']
            assert p['fp32_sha256']==a['sha256'] and p['fp64_sha256']==b['sha256']
            assert a['object_id']!=b['object_id'] and a['storage_id']!=b['storage_id']
    records.append(dict(cell=cell,status='PASS',sealed_inputs_checked=len(views),parameter_roster_checked=len(live),basis='Read-only hash reconciliation; exact promotion tensor equality was observed by live worker'))
write(OUT/'input_promotion_audit.json',records)
