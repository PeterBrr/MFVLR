import json
from pathlib import Path
import numpy as np
import pytest
from mfvlr_research.d02b1.gradients import GradientStore
from mfvlr_research.d02b1.schema import PASSES,validate_inventory
from mfvlr_research.d02b10.evidence import validate_stream

@pytest.mark.parametrize('corruption',[None,'gradient','sum','metadata','missing'])
def test_full_evidence_integrity(tmp_path,corruption):
    store=GradientStore(tmp_path/'stream',{'x':dict(shape=[2],presence={n:'allocated' for n in PASSES})})
    for n in PASSES:store.put(n,'x',np.array([1,2],dtype='float32'))
    analysis=store.analyze();path=tmp_path/'stream';roster=[dict(name='x',shape=[2])]
    if corruption=='gradient':(path/'00000.f32').write_bytes(b'corrupt!')
    elif corruption=='sum':(path/analysis['parameters']['x']['sum_path']).write_bytes(b'bad')
    elif corruption=='metadata':analysis['parameters']['x']['passes']['fd']['allocated']=False
    elif corruption=='missing':roster.append(dict(name='extra',shape=[1]))
    if corruption:
        with pytest.raises(RuntimeError):validate_stream(path,roster,analysis)
    else:assert validate_stream(path,roster,analysis)['gradient_records']==7

@pytest.mark.parametrize('corruption',[None,'owner','shape','count','dtype'])
def test_frozen_508_roster_metadata_only(corruption):
    roster=json.loads(Path('tests/research_d02b1/fixtures/historical_rosters.json').read_text())['rosters']['A0']
    rows=[dict(name=r['name'],shape=list(r['shape']),dtype='float32',sha256='1'*64,object_id=str(i),storage_id=str(i)) for i,r in enumerate(roster)]
    if corruption=='owner':rows[1]['storage_id']=rows[0]['storage_id']
    elif corruption=='shape':rows[0]['shape']=[1]
    elif corruption=='count':rows.pop()
    elif corruption=='dtype':rows[0]['dtype']='float64'
    if corruption:
        with pytest.raises(ValueError):validate_inventory(rows,'A0',roster)
    else:validate_inventory(rows,'A0',roster)
