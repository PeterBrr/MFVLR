"""Full streamed-output integrity; array bytes are hashed without loading models."""
import hashlib
import json
import math
from pathlib import Path

def validate_stream(path,roster,analysis):
    from mfvlr_research.d02b1.schema import PASSES
    path=Path(path)
    names=[r['name'] for r in roster]
    if len(set(names))!=len(names) or set(analysis['parameters'])!=set(names):raise RuntimeError('Evidence roster')
    if len(list(path.glob('[0-9][0-9][0-9][0-9][0-9].json')))!=7*len(names):raise RuntimeError('Raw evidence count')
    def verify_file(name,digest,size):
        if Path(name).name!=name:raise RuntimeError('Evidence traversal path')
        file=path/name
        if file.is_symlink() or file.stat().st_size!=size:raise RuntimeError('Evidence file/size')
        with file.open('rb') as stream:actual=hashlib.file_digest(stream,'sha256').hexdigest()
        if actual!=digest:raise RuntimeError('Evidence hash')
    for i,(loss,row) in enumerate((l,r) for l in PASSES for r in roster):
        record=json.loads((path/f'{i:05d}.json').read_text())
        if record['loss']!=loss or record['parameter']!=row['name'] or record['shape']!=row['shape']:raise RuntimeError('Raw identity/shape')
        saved=analysis['parameters'][row['name']]['passes'][loss]
        if any(record.get(k)!=v for k,v in saved.items()):raise RuntimeError('Analysis/raw mismatch')
        if record['allocated']:verify_file(record['path'],record['sha256'],4*math.prod(row['shape']))
    for row in roster:
        record=analysis['parameters'][row['name']]
        verify_file(record['sum_path'],record['sum_sha256'],8*math.prod(row['shape']))
    return dict(gradient_records=7*len(names),sum_shards=len(names),status='PASS')
