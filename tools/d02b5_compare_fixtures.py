"""Read-only F0/F1/F2 aggregation. Never import or execute torch."""
import json,sys,collections
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b5.policy import write,hashfile
from mfvlr_research.d02b2.contract import stages
OUT=ROOT/'artifacts/phase2d2_d02b5'
core=set(stages('F2',1)); summaries=[]; sources={}; details=[]; directions=[]
for fixture,phase in (('F0','d02b3'),('F1','d02b4'),('F2','d02b5')):
    folder=ROOT/'artifacts'/f'phase2d2_{phase}'
    records={}
    for name in ('cross_batch_all.json','activation_scales.json','execution_matrix.json'):
        path=folder/name;sources[path.relative_to(ROOT).as_posix()]=hashfile(path)
        records[name]=json.loads(path.read_text())
    raw=records['cross_batch_all.json']
    lookup={(r['cell'].split('-')[2],r['cell'].split('-')[3],r['sample'],r['stage']):r for r in raw}
    for row in records['execution_matrix.json']:
        cell=row['cell']; variant,precision=cell.split('-')[2:]
        allrows=[r for r in raw if r['cell']==cell]; common=[r for r in allrows if r['stage'] in core]
        if row['result']['execution']!='COMPLETE':raise RuntimeError('Incomplete input cannot be summarized as complete')
        if len(common)!=54 or len(allrows)!=(78 if fixture=='F0' else 54):raise RuntimeError('Coverage mismatch')
        summaries.append(dict(fixture=fixture,variant=variant,precision=precision,capture_count=len(allrows)//2,
            full_comparisons=len(allrows),full_failed=sum(not r['allclose'] for r in allrows),full_outside=sum(r['outside_tolerance_count'] for r in allrows),
            core_comparisons=len(common),core_failed=sum(not r['allclose'] for r in common),core_outside=sum(r['outside_tolerance_count'] for r in common),
            wall_seconds=row['wall_seconds'],sampled_peak_rss_bytes=row['sampled_peak_rss_bytes']))
        for r in common:details.append(dict(fixture=fixture,variant=variant,precision=precision,**r))
    for precision in ('float32','float64'):
        for variant in ('P3','Q3'):
            counts={m:collections.Counter() for m in ('max_absolute_difference','relative_l2_difference','outside_proportion')}
            for sample in ('0','1'):
                for stage in sorted(core):
                    a=lookup['A0',precision,sample,stage];b=lookup[variant,precision,sample,stage]
                    for metric,counter in counts.items():
                        x,y=a[metric],b[metric]
                        counter['undefined' if x is None or y is None else 'lower' if y<x else 'higher' if y>x else 'equal']+=1
            directions.append(dict(fixture=fixture,variant=variant,precision=precision,scope='54 common-core stage/sample records',metrics={k:dict(v) for k,v in counts.items()}))
write(OUT/'f0_f1_f2_comparison.json',dict(source_hashes=sources,summary=summaries,paired_directions=directions,
    interpretation='Read-only. Core subset excludes twelve F0 detailed captures. Within-fixture lower/higher are sensitivity differences, not accuracy or superiority. Across fixtures inputs differ.'))
write(OUT/'f0_f1_f2_core_records.json',details)
lines=['# Read-only F0/F1/F2 scientific comparison','',
'All values come from existing records; no F0/F1 computation was rerun. F0 has 39 capture stages, including 12 detailed stages; F1/F2 have 27. Common-core columns use exactly the same 27 stage names and both samples. Full columns preserve every original failure. Different fixtures have different prompt and visual bytes, so cross-fixture totals are not controlled architecture comparisons.','',
'| Fixture | Variant | Precision | Full failed / comparisons | Core failed / 54 | Full outside | Core outside |','| --- | --- | --- | ---: | ---: | ---: | ---: |']
for r in summaries:
    lines.append(f"| {r['fixture']} | {r['variant']} | {r['precision']} | {r['full_failed']}/{r['full_comparisons']} | {r['core_failed']} | {r['full_outside']} | {r['core_outside']} |")
lines+=['','Paired core sensitivity counts against A0 within each identical fixture and precision (lower / higher / equal; undefined shown separately in JSON):','',
'| Fixture | Variant | Precision | Max absolute | Relative L2 | Outside proportion |','| --- | --- | --- | --- | --- | --- |']
for r in directions:
    vals=[' / '.join(str(r['metrics'][m].get(k,0)) for k in ('lower','higher','equal')) for m in ('max_absolute_difference','relative_l2_difference','outside_proportion')]
    lines.append('| '+ ' | '.join([r['fixture'],r['variant'],r['precision']]+vals)+' |')
lines+=['','Lower and higher error counts preserve improvements and regressions without choosing favorable stages. Full per-stage data, including weak values, remains available in the included core records and each original raw table. Neither lower activation magnitudes nor lower total failure counts establish superiority, harmless drift, training stability or overall N2 acceptance. Phase 2D.2 remains BLOCKED.']
with (ROOT/'docs/phase2d2_d02b5_f0_f1_f2_comparison.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
