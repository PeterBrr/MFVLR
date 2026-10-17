"""Read-only reconciliation of all 18 recorded N2 cells. No torch or workers."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b1.schema import write_new,file_hash
from mfvlr_research.d02b2.contract import stages,COUNTS
from mfvlr_research.d02b6.adapter import primary_matrix,cell_plan
OUT=ROOT/'artifacts/phase2d2_d02b6'
sources={};rows=[]
def read(path):
    sources[path.relative_to(ROOT).as_posix()]=file_hash(path)
    return json.loads(path.read_text(encoding='utf-8'))
for fixture,phase in (('F0','d02b3'),('F1','d02b4'),('F2','d02b5')):
    folder=ROOT/'artifacts'/f'phase2d2_{phase}'
    matrix=read(folder/'execution_matrix.json')
    expected=[f'{phase.upper()}-{fixture}-{v}-{p}' for p in ('float32','float64') for v in ('A0','P3','Q3')]
    assert [r['cell'] for r in matrix]==expected
    for row in matrix:
        cell=row['cell'];variant,precision=cell.split('-')[2:];f=folder/'cells'/cell/'attempt_01'
        result=read(f/'result.json');assert row['exit_code']==0 and result['execution']=='COMPLETE'
        before=read(f/'live_inventory.json');after=read(f/'final_inventory.json');assert before==after
        assert len(before['parameters'])==COUNTS[variant][0] and before['vocabulary_tied']
        for name in ('permutation.json','independence_changed0.json','independence_changed1.json'):
            exact=read(f/name);assert set(exact)==set(stages(fixture,2)) and all(x is True for x in exact.values())
        failed=outside=compared=0
        for sample in (0,1):
            records=read(f/f'cross_batch_sample{sample}.json');assert set(records)==set(stages(fixture,1))
            for name,m in records.items():
                assert m['shape']==stages(fixture,1)[name] and m['finite']
                assert (m['atol'],m['rtol'])==((1e-6,1e-5) if precision=='float32' else (1e-10,1e-8))
                compared+=1;failed+=int(m['outside_tolerance_count']>0);outside+=m['outside_tolerance_count']
            d=read(f/f'cross_derivative_{sample}.json')
            assert d['other_visual_nonzero']==d['other_lookup_nonzero']==0
            assert d['own_visual']['l2_norm']>0 and d['own_lookup']['l2_norm']>0
        if precision=='float64':
            assert outside==0
            p=read(f/'promotion.json');assert p['fp32_source_unchanged']
            assert len(p['parameters'])==COUNTS[variant][0] and all(x['exact_cast'] for x in p['parameters'])
        else:
            for batch in (1,2):
                g=read(f/f'gradients_B{batch}.json');assert set(g['parameters'])=={p['name'] for p in before['parameters']}
                assert len(g['vim_qk'])==38 and all(g['parameters'][n]['nonzero']==0 for n in g['vim_qk'])
                assert all(v['allocated'] and v['finite'] for v in g['parameters'].values())
        rows.append(dict(cell=cell,fixture=fixture,variant=variant,precision=precision,execution='COMPLETE',
            structural_exact='PASS (record audit)',state_ownership='PASS (record audit)',
            comparisons=compared,failed_comparisons=failed,outside_coordinates=outside,
            promoted_fp64='PASS' if precision=='float64' else 'NOT APPLICABLE',
            fp32_diagnostic='FAIL' if precision=='float32' and failed else 'NOT APPLICABLE',
            parameter_gradients='B1/B2 PASS (record audit)' if precision=='float32' else 'NOT RUN by original scope'))
write_new(OUT/'n2_closure_matrix.json',{'rows':rows,'source_hashes':sources,'basis':'Read-only verification of recorded live observations; no numerical rerun','phase2d2':'BLOCKED'})
with (OUT/'n2_closure_matrix.csv').open('x',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
primary=primary_matrix();write_new(OUT/'g2_readiness_matrix.json',primary)
write_new(OUT/'g2_pass_plans.json',[cell_plan(c['experiment_id']) for c in primary])
g2={}
for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
    path=ROOT/'artifacts/phase2d2_n31/g2'/f'final_{case}'/'summary.json'
    value=read(path);g2[case]=dict(path=path.relative_to(ROOT).as_posix(),sha256=file_hash(path),historical_status=value['status'],fixture=value['fixture'],state=value['initialized_state_sha256'])
write_new(OUT/'historical_g2_anchors.json',g2)
lines=['# D02B6 N2 evidence closure','', '2026-10-16, Asia/Saigon. **Read-only closure; Phase 2D.2 remains BLOCKED.**',
'', 'All 18 registered cells completed. The following statuses audit existing observations; none was rerun in D02B6. Execution and structural checks do not convert FP32 diagnostic FAILs into PASS. The original tolerances remain FP32 1e-6/1e-5 and promoted FP64 1e-10/1e-8. N1 is not transferred.','',
'| Cell | Execution | Exact/structural | Comparisons | Numerical FAILs | Outside coordinates | Promoted FP64 |',
'| --- | --- | --- | ---: | ---: | ---: | --- |']
for r in rows:lines.append(f"| {r['cell']} | COMPLETE | PASS, record audit | {r['comparisons']} | {r['failed_comparisons']} | {r['outside_coordinates']} | {r['promoted_fp64']} |")
lines+=['','F0 records 39 captures (27 core + 12 detailed), while F1 and F2 each record 27. All nine promoted-FP64 cells have zero outside-tolerance comparisons; all nine FP32 cells retain failed comparisons. Complete per-cell hashes, actual before/after inventories, derivative records and allocation/zero evidence are indexed in the JSON closure matrix. The compact package includes those source records.','',
'## Prerequisites demonstrated and outstanding','',
'The former D02B1 outstanding P3/Q3 full-language N2 prerequisite is now demonstrated for F0/F1/F2: seed-0 common FP32 initialization, privately owned variants, exact deep-copy promotion, both B1 slices, exact fixed-batch independence/permutation, both genuine cross-sample derivative directions and connected own-sample controls. All 18 cells preserved state and tied vocabulary. FP32 B1/B2 full gradients were finite and allocated, including exact singleton VIM Q/K zeros. FP64 full parameter-backward sweeps were not part of N2 and remain NOT RUN. Weak ordinary Q/K and key-bias residuals remain observations, not universal accuracy proofs.','',
'A1.1/D02A LN forward/VJP and ordinary-attention primitive controls remain previously verified for their named local scope. The 414-job local schedule remains complete with every original failure retained. D02B6 does not rerun or strengthen those controls. Original A0 N3 same-forward and CMC-chain evidence still includes genuine failures. P3/Q3 same-forward six-loss controls are NOT RUN, separately scoped; optional new unchanged-path CMC/visual controls are NOT RUN. Backend controls, NG2, LR-category parameter-gradient decomposition and isolated tied-vocabulary path probes remain separately authorized/outstanding. Scalar LR category statistics cannot satisfy those decomposition requirements.','',
'## Boundary of closure','',
'The pre-G2 full-language N2 observation gap is closed under the already scoped diagnostic policy, not by a new waiver. The historical eight failing assertions (including the six A0 G2 fixtures) remain unresolved and unchanged. There is no full-six-loss numerical acceptance, optimizer/update stability, convergence, capacity or accuracy claim. All G01–G06 × A0/P3/Q3 cells remain NOT RUN. B0-D03 and B0-D08 remain disabled pending separate approval.']
with (ROOT/'docs/phase2d2_d02b6_n2_closure.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
print('18 N2 cells reconciled read-only; 18 primary G2 cells NOT RUN')
