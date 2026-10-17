import json
import numpy as np
import pytest
from mfvlr_research.d02b1.gradients import GradientStore
from mfvlr_research.d02b9.comparison import compare_streams
from mfvlr_research.d02b9.same_forward import ORDER

def stream(path,value):
    specs={'x':dict(shape=[2],presence={n:'allocated' for n in (*ORDER,'total')})}
    store=GradientStore(path,specs)
    for name in (*ORDER,'total'):store.put(name,'x',np.array(value,dtype=np.float32))

def test_raw_pairing(tmp_path):
    stream(tmp_path/'old',[1.,-1.]);stream(tmp_path/'new',[1.,0.])
    rows=compare_streams(tmp_path/'new',tmp_path/'old')
    assert len(rows)==7 and all(r['outside_tolerance_count']==1 for r in rows)
    assert all(r['absolute_l2']==1. and r['relative_l2']==pytest.approx(2**-.5) for r in rows)

def test_raw_integrity(tmp_path):
    stream(tmp_path/'old',[1.,-1.]);stream(tmp_path/'new',[1.,-1.])
    assert all(r['comparison']=='PASS' for r in compare_streams(tmp_path/'new',tmp_path/'old'))
    (tmp_path/'new'/'00000.f32').write_bytes(b'bad')
    with pytest.raises(ValueError):compare_streams(tmp_path/'new',tmp_path/'old')
