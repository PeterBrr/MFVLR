import hashlib,json,math
from pathlib import Path
CELLS=tuple(f'D02B3-F0-{v}-{d}' for d in ('float32','float64') for v in ('A0','P3','Q3'))
RAM_BYTES=5*1024**3
DISK_BYTES=8*1024**3
def identity(cell):
    if type(cell) is not str or cell not in CELLS: raise PermissionError('Only six explicit D02B3 F0 cells authorized')
    _,fixture,variant,dtype=cell.split('-')
    return fixture,variant,dtype
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(value,indent=2,allow_nan=False)
    with path.open('x',encoding='utf-8') as f: f.write(data+'\n')
def hashfile(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()
def resources(available,disk):
    if any(type(x) is not int or x<0 for x in (available,disk)): raise ValueError('Invalid resource observation')
    return available>=RAM_BYTES and disk>=DISK_BYTES
def check_backend(actual,expected):
    if actual!=expected: raise RuntimeError('Pinned backend mismatch')
def classify(comparisons,dtype):
    outside=sum(r['outside_tolerance_count'] for r in comparisons)
    if any(not r['finite'] for r in comparisons): raise FloatingPointError('Nonfinite comparison')
    return dict(numerical='FAIL' if outside else 'PASS',outside_tolerance_count=outside,
                mandatory_failure=bool(outside and dtype=='float64'),
                policy='original N2 FP32 cross-batch diagnostic' if dtype=='float32' else 'mandatory FP64 agreement')
