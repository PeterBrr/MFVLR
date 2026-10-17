"""Aggregate actual F2 evidence; never rerun or reinterpret numerical failures."""
import json,hashlib,sys,xml.etree.ElementTree as ET,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/phase2d2_d02b5'
sys.path.insert(0,str(ROOT/'src'))
from mfvlr_research.d02b5.policy import CELLS,write,hashfile
def main():
    matrix=json.loads((OUT/'execution_matrix.json').read_text())
    suite=ET.Element('testsuite',name='D02B5 live F2 checks'); tests=failures=skips=0
    table=[];raw=[];scales=[];preservation=[];gradient_categories=[]
    for item in matrix:
        cell=item['cell'];result=item['result'];folder=OUT/'cells'/cell/'attempt_01'
        node=ET.SubElement(suite,'testcase',name=cell+'::mandatory_execution',classname='D02B5.live')
        tests+=1
        if result['execution']=='NOT RUN':
            ET.SubElement(node,'skipped',message=result.get('reason','not run'));skips+=1
        elif result['execution']!='COMPLETE':
            ET.SubElement(node,'failure',message=result.get('exception',result['execution'])).text=result.get('traceback','');failures+=1
        for path in sorted(folder.glob('cross_batch_sample*.json')) if folder.exists() else []:
            for stage,m in json.loads(path.read_text()).items():
                row=dict(cell=cell,sample=path.stem[-1],stage=stage,**m);raw.append(row)
                test=ET.SubElement(suite,'testcase',name=f'{cell}::{path.stem}::{stage}',classname='D02B5.numerical');tests+=1
                if m['outside_tolerance_count']:
                    ET.SubElement(test,'failure',message=f"{m['outside_tolerance_count']} outside unchanged tolerance; {'FP32 diagnostic only' if cell.endswith('float32') else 'mandatory FP64'}");failures+=1
        table.append(f"| {cell} | {result['execution']} | {result.get('numerical','NOT RUN')} | {result.get('outside_tolerance_count','â€”')} | {item.get('wall_seconds',0):.2f} | {item.get('sampled_peak_rss_bytes',0)/1024**2:.1f} |")
        if (folder/'activations.json').exists():
            activations=json.loads((folder/'activations.json').read_text())
            for name,values in activations.items():scales.append(dict(cell=cell,stage=name,**values))
        if (folder/'final_inventory.json').exists():
            a=json.loads((folder/'live_inventory.json').read_text());b=json.loads((folder/'final_inventory.json').read_text())
            if a!=b:raise RuntimeError('Actual inventory changed')
            preservation.append(dict(cell=cell,parameter_tensors=len(a['parameters']),identities_hashes_unchanged=True))
        for path in sorted(folder.glob('gradients_B*.json')) if folder.exists() else []:
            record=json.loads(path.read_text())
            for name,g in record['parameters'].items():
                role='singleton_vim_qk' if '.vim.q.' in name or '.vim.k.' in name else 'ordinary_key_bias' if name.endswith('.k.bias') and 'attention' in name else 'ordinary_qk_weight' if name.endswith(('.q.weight','.k.weight')) and 'attention' in name else 'other'
                if role!='other':gradient_categories.append(dict(cell=cell,batch=int(path.stem[-1]),parameter=name,category=role,**g))
    suite.attrib.update(tests=str(tests),failures=str(failures),errors='0',skipped=str(skips))
    ET.ElementTree(suite).write(OUT/'live_results.junit.xml',encoding='utf-8',xml_declaration=True)
    write(OUT/'cross_batch_all.json',raw)
    write(OUT/'activation_scales.json',scales)
    write(OUT/'parameter_preservation.json',preservation)
    write(OUT/'gradient_categories.json',gradient_categories)
    paired=[]
    lookup={(r['cell'],r['sample'],r['stage']):r for r in raw}
    for precision in ('float32','float64'):
        for variant in ('P3','Q3'):
            for (cell,sample,stage),base in lookup.items():
                if cell!=f'D02B5-F2-A0-{precision}':continue
                other=lookup.get((f'D02B5-F2-{variant}-{precision}',sample,stage))
                if other is None:continue
                metrics={}
                for metric in ('max_absolute_difference','relative_l2_difference','outside_proportion'):
                    a,b=base[metric],other[metric]
                    metrics[metric]={'A0':a,variant:b,'direction':'UNDEFINED' if a is None or b is None else 'lower' if b<a else 'higher' if b>a else 'equal'}
                paired.append(dict(variant=variant,precision=precision,sample=sample,stage=stage,metrics=metrics,
                                   interpretation='Different functions; difference in batch-shape sensitivity, not architecture ranking'))
    write(OUT/'paired_sensitivity.json',paired)
    write(OUT/'aggregation.json',dict(cells=len(matrix),complete=sum(x['result']['execution']=='COMPLETE' for x in matrix),
          numerical_comparisons=len(raw),failed_comparisons=sum(bool(x['outside_tolerance_count']) for x in raw),
          outside_coordinates=sum(x['outside_tolerance_count'] for x in raw),junit=dict(tests=tests,failures=failures,skipped=skips),
          scope='JUnit generated directly from live raw outcomes; diagnostic failures deliberately retained',phase2d2='BLOCKED'))
    report='''# D02B5 controlled F2 full-language numerical pilot

2026-10-16, Asia/Saigon. **Phase 2D.2 remains BLOCKED.** Only the six explicitly authorized F2 FLT-only cells were eligible. Frozen D02B1/D02B2 dispatchers, production models, objectives, fixtures and original tests remain unchanged. No F0/F1, six-loss G2, NG2, optimizer, training, G3/G4 or benchmark was executed.

## Actual execution

| Cell | Execution | Numerical comparison | Outside coordinates | Wall seconds | Sampled peak RSS MiB |
| --- | --- | --- | ---: | ---: | ---: |
'''+ '\n'.join(table)+'''

Execution completion is distinct from numerical PASS. FP32 cross-batch failed comparisons remain genuine failures under the original N2 diagnostic scope. Any promoted FP64 disagreement or structural failure stops dependent cells. NOT RUN rows are not successes. See `execution_matrix.json` and each `result.json` for precise failure traces and status.

## Provenance and live checks

A new versioned D02B5 layer allows only the exact six cell IDs. A single exclusive coordinator owns the sequence using PID plus process creation time. A fresh worker per cell independently initializes the canonical seed-0 CPU FP32 reference, verifies its original N2 state hash, then calls the frozen A1.1 builder. All common values and disjoint ownership are verified by the frozen parity checker. The source reference is released before captures. Private candidates retain 387/429 parameter tensors, 12 LE blocks, seven LD blocks, 19 VIM roles and tied vocabulary; counts and per-tensor hashes are in each live inventory.

FP64 cells deep-copy the actual candidate and compare every promoted tensor, chunk by chunk, with its actual FP32 source converted to FP64. They do not trust a supplied expected digest or merely round-trip to FP32. Original F2 token/visual bytes are loaded from the hash-pinned historical input bundle. Before every model forward, a live pre-hook checks parameter bytes/identities and runtime backend and writes an exclusive state/input/source/backend seal. Forward values are observed under no-grad for numerical captures; hooks are removed and captures checked for graph retention.

Each completed cell records all 27 original F2 core captures (no additional F0 detailed captures), both B1 row slices versus B2, exact fixed-batch independence in both directions, exact permutation and genuine visual/embedding-output cross-sample autograd derivatives with connected own-sample positive controls. The user-authorized derivative isolation checks are performed for both precisions; no FP64 full parameter-gradient sweep is added. FP32 cells additionally execute full probe backward at B1 and B2. Every parameter gradient is recorded with finite norms, maxima, nonzero counts and hashes; all 38 singleton VIM Q/K gradients must be allocated exact zeros. Ordinary Q/K weights and ordinary key-bias residuals remain unfiltered in those records. No ordinary-attention derivative accuracy oracle or training loss is substituted for the original N2 probe.

The probe is `mean(logits^2)+mean(decoder_hidden^2)+mean(global_language^2)`. Cross-sample probes use the corresponding sample only and differentiate the actual `output.low` embedding boundary and connected visual leaf. The seven original positive-control parameter roles and visual gradient remain mandatory. A state check after all probes verifies unchanged parameters/ownership.

## Policy and resources

Original comparison constants remain FP32 atol=1e-6/rtol=1e-5 and promoted FP64 atol=1e-10/rtol=1e-8. Fixed-batch equality, permutation and cross-sample zeros are exact. N1 is not used. No policy extends to additivity, six losses or another category. `live_results.junit.xml` represents raw observed outcomes, including failed FP32 diagnostic comparisons; it is not a pytest suite claiming those discrepancies passed. `preflight_junit.xml` is the separately executed isolated 25-case static/mock pytest suite.

The operational launch reservation is 5 GiB available RAM and 8 GiB free disk for either precision. This conservative engineering safeguard is not a measured guarantee of capacity. Insufficient resources are recorded WAIT/NOT RUN. Workers run sequentially; a second in-worker resource/ownership check occurs before importing torch. Available RAM, disk, PID/creation, runtime build/thread metadata, process time and sampled RSS are preserved. Peak RSS is sampled, not a guaranteed instantaneous maximum. No GPU/B8 capacity claim is made.

All outputs use exclusive paths. No interrupted attempt is overwritten or auto-replayed. Unexpected interruption has UNKNOWN cause unless evidence identifies it. Source preservation uses before/after SHA-256 inventories. The compact package includes complete source, isolated tests, original fixture/source references, raw cell records/logs, actual failure traces, JUnit, manifests and preservation audits; old large archives are referenced by hashes rather than duplicated.

## Interpretation and limits

These are bounded no-update F2 observations. `activation_scales.json` retains every capture's RMS/max/norm; `paired_sensitivity.json` preserves lower, higher, equal and undefined metrics for every matched stage/sample/precision, rather than selecting favorable layers. Compare each variant with its own B1 reference: the variants represent different functions. Do not infer architecture superiority from activation magnitudes, total failed comparisons or global relative L2. FP32 diagnostic success cannot establish harmless numerical drift. Full numerical agreement, repeated-step stability, convergence, accuracy and author architecture fidelity are not established. A mandatory stop leaves later authorized cells NOT RUN pending diagnosis and review; authorization does not permit weakening thresholds or silently retrying modified conditions.

Stop for user review. Phase 2D.2 remains BLOCKED.
'''
    with (ROOT/'docs/phase2d2_d02b5_f2_numerical_report.md').open('x',encoding='utf-8') as f:f.write(report)
    print(json.dumps(dict(tests=tests,failures=failures,skipped=skips)))
def audit():
    matrix=json.loads((OUT/'execution_matrix.json').read_text())
    if [r['cell'] for r in matrix]!=list(CELLS):raise RuntimeError('Six exact cells required in matrix')
    checks=[]
    from mfvlr_research.d02b2.contract import stages,COUNTS,bundle
    for row in matrix:
        cell=row['cell'];result=row['result'];folder=OUT/'cells'/cell/'attempt_01'
        if result['execution']!='COMPLETE':
            checks.append(dict(cell=cell,status=result['execution'],complete_audit='NOT APPLICABLE'));continue
        variant=cell.split('-')[2];precision=cell.split('-')[3]
        a=json.loads((folder/'live_inventory.json').read_text());b=json.loads((folder/'final_inventory.json').read_text())
        if a!=b or len(a['parameters'])!=COUNTS[variant][0]:raise RuntimeError('Ownership changed')
        if result['fp32_base_sha256']!=bundle()['fixtures']['F2']['a0_fp32_state_sha256']:raise RuntimeError('Seed/state')
        for name in ('permutation.json','independence_changed0.json','independence_changed1.json'):
            values=json.loads((folder/name).read_text())
            if set(values)!=set(stages('F2',2)) or not all(v is True for v in values.values()):raise RuntimeError('Exact independence')
        for sample in (0,1):
            cross=json.loads((folder/f'cross_batch_sample{sample}.json').read_text())
            if set(cross)!=set(stages('F2',1)):raise RuntimeError('Capture coverage')
            for name,m in cross.items():
                if m['shape']!=stages('F2',1)[name] or not m['finite']:raise RuntimeError('Shape/finite')
                if (m['atol'],m['rtol'])!=((1e-6,1e-5) if precision=='float32' else (1e-10,1e-8)):raise RuntimeError('Tolerance')
                if precision=='float64' and m['outside_tolerance_count']:raise RuntimeError('Mandatory FP64 failure mislabeled')
            derivative=json.loads((folder/f'cross_derivative_{sample}.json').read_text())
            if derivative['sample']!=sample or derivative['other_visual_nonzero'] or derivative['other_lookup_nonzero'] or derivative['own_visual']['l2_norm']<=0 or derivative['own_lookup']['l2_norm']<=0:raise RuntimeError('Derivative isolation')
        seals=sorted(folder.glob('forward_*_seal.json'))
        if len(seals)!=result['forward_calls'] or len(seals)!=(9 if precision=='float32' else 7):raise RuntimeError('Missing forward seals')
        for i,path in enumerate(seals):
            seal=json.loads(path.read_text())
            if seal['index']!=i or seal['state_sha256']!=result['live_state_sha256'] or seal['backend']!=bundle()['fixtures']['F2']['backend']:raise RuntimeError('Forward seal')
        if precision=='float32':
            for batch in (1,2):
                g=json.loads((folder/f'gradients_B{batch}.json').read_text())
                if set(g['parameters'])!={p['name'] for p in a['parameters']} or len(g['vim_qk'])!=38:raise RuntimeError('Gradient coverage')
                if any(not r['allocated'] or not r['finite'] for r in g['parameters'].values()):raise RuntimeError('Gradient finite/presence')
                if any(g['parameters'][name]['nonzero'] for name in g['vim_qk']):raise RuntimeError('Structural zero')
        checks.append(dict(cell=cell,status='PASS',basis='Read-only audit of complete live records; no experiment rerun'))
    write(OUT/'live_evidence_audit.json',dict(checks=checks,phase2d2='BLOCKED'))
    print('live evidence audit complete')
def package():
    paths=set()
    for folder in ('src','tests/research_d02b5','artifacts/phase2d2_d02b5'):
        paths.update(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.zip' and p.name not in ('manifest.json','package_verification.json'))
    paths.update((ROOT/'tools').glob('d02b5_*.py'))
    paths.add(ROOT/'docs/phase2d2_d02b5_f2_numerical_report.md')
    paths.add(ROOT/'docs/phase2d2_d02b5_f0_f1_f2_comparison.md')
    for phase in ('d02b3','d02b4'):
        for name in ('cross_batch_all.json','activation_scales.json','execution_matrix.json'):
            paths.add(ROOT/'artifacts'/f'phase2d2_{phase}'/name)
    from mfvlr_research.d02b2.contract import bundle
    paths.update(ROOT/p for p in bundle()['sources'])
    files={p.relative_to(ROOT).as_posix():dict(sha256=hashfile(p),bytes=p.stat().st_size) for p in sorted(paths)}
    write(OUT/'manifest.json',dict(files=files,phase2d2='BLOCKED'))
    archive=OUT/'phase2d2_d02b5_review.zip'
    with zipfile.ZipFile(archive,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for rel in files:z.write(ROOT/rel,rel)
        z.write(OUT/'manifest.json','manifest.json')
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:raise RuntimeError('CRC')
        if set(z.namelist())!=set(files)|{'manifest.json'} or len(z.namelist())!=len(files)+1:raise RuntimeError('Membership')
        if any(hashlib.sha256(z.read(p)).hexdigest()!=v['sha256'] for p,v in files.items()):raise RuntimeError('Payload hash')
        if z.read('manifest.json')!=(OUT/'manifest.json').read_bytes():raise RuntimeError('Manifest')
    write(OUT/'package_verification.json',dict(status='PASS',crc='PASS',payload_hashes=len(files),archive_bytes=archive.stat().st_size,
          archive_sha256=hashfile(archive),manifest_sha256=hashfile(OUT/'manifest.json')))
    print(archive)
if __name__=='__main__': {'aggregate':main,'audit':audit,'package':package}[sys.argv[1]]()
