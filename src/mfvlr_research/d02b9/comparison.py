"""Stream same-forward versus separate-forward raw gradient observations."""
import json
import hashlib
import math
from pathlib import Path
import numpy as np

def _records(root):
    result = {}
    for path in sorted(root.glob('[0-9][0-9][0-9][0-9][0-9].json')):
        row = json.loads(path.read_text(encoding='utf-8'))
        identity = (row['loss'],row['parameter'])
        if identity in result: raise ValueError('Duplicate raw gradient identity')
        if row['allocated']:
            if Path(row['path']).name != row['path']: raise ValueError('Raw shard path')
            file = root / row['path']
            with file.open('rb') as stream: actual = hashlib.file_digest(stream,'sha256').hexdigest()
            if actual != row['sha256'] or file.stat().st_size != row['bytes'] or row['bytes'] != 4*math.prod(row['shape']):
                raise ValueError('Raw gradient integrity')
        result[identity] = row
    if not result: raise ValueError('Missing comparison records')
    return result

def compare_streams(current, historical):
    current,historical=Path(current),Path(historical)
    a,b=_records(current),_records(historical)
    if set(a) != set(b): raise ValueError('Incomplete matched gradient roster')
    result=[]
    for key,x in a.items():
        y=b[key]
        if x['shape'] != y['shape']: raise ValueError('Shape mismatch')
        row=dict(loss=key[0],parameter=key[1],same_forward_allocated=x['allocated'],historical_allocated=y['allocated'])
        if x['allocated'] != y['allocated']:
            row['structural']='FAIL'
        elif not x['allocated']:
            row.update(structural='PASS',comparison='BOTH_NONE')
        else:
            count=0;delta2=0.;ref2=0.;maxabs=0.
            with (current/x['path']).open('rb') as fa,(historical/y['path']).open('rb') as fb:
                while True:
                    aa=np.frombuffer(fa.read(262144*4),dtype='<f4')
                    bb=np.frombuffer(fb.read(262144*4),dtype='<f4')
                    if not aa.size:break
                    if not np.isfinite(aa).all() or not np.isfinite(bb).all():raise ValueError('Nonfinite raw gradient')
                    count+=int((np.abs(aa-bb)>np.float32(1e-6)+np.float32(1e-5)*np.abs(bb)).sum())
                    d=aa.astype('float64')-bb.astype('float64');ref=bb.astype('float64')
                    delta2+=float(d@d);ref2+=float(ref@ref);maxabs=max(maxabs,float(np.abs(d).max()))
            row.update(structural='PASS',comparison='FAIL' if count else 'PASS',outside_tolerance_count=count,
                absolute_l2=math.sqrt(delta2),relative_l2=math.sqrt(delta2/ref2) if ref2 else None,max_abs=maxabs)
        result.append(row)
    return result
