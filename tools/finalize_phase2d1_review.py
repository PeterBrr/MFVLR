"""Validate actual evidence, preserve history, assemble a source review package."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d1'
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def suite(path):
    root=ET.parse(path).getroot();cases=root.findall('.//testcase')
    result={'path':path.relative_to(ROOT).as_posix(),'cases':len(cases),'passed':0,'failed':0,'errors':0,'skipped':0,
            'seconds':sum(float(s.attrib.get('time',0)) for s in root.findall('.//testsuite')),'case_ids':[]}
    for case in cases:
        result['case_ids'].append([case.attrib['classname'],case.attrib['name']])
        if case.find('failure') is not None:result['failed']+=1
        elif case.find('error') is not None:result['errors']+=1
        elif case.find('skipped') is not None:result['skipped']+=1
        else:result['passed']+=1
    assert result['cases']==len({tuple(x) for x in result['case_ids']}),'duplicate cases'
    return result


def latest(stage):
    execution=sorted((OUT/'logs').glob(f'{stage}_*.execution.json'))[-1]
    data=read(execution);assert data['exit_code']==0,data
    return suite(OUT/'junit'/execution.name.replace('.execution.json','.xml')),data


def main():
    import torch
    import yaml
    from mfvlr.heads import DetectionHeads,head_parameter_report
    from mfvlr.objectives import SixObjectives
    from mfvlr.heads_losses_config import load_heads_losses_config
    baseline=read(OUT/'baseline_manifest.json')
    bad=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not bad,bad
    complete,execution=latest('complete');regression,_=latest('regression');unit,_=latest('unit')
    previous=suite(ROOT/'artifacts/phase2c2_n2_acceptance/junit/complete.xml')
    earlier=suite(ROOT/'artifacts/phase2c1/junit/complete_01.xml')
    ids=lambda r:{tuple(x) for x in r['case_ids']}
    assert regression['cases']==164 and previous['cases']==164 and earlier['cases']==119
    assert ids(regression)==ids(previous) and ids(previous)<=ids(complete)
    assert ids(earlier)<=ids(complete)
    assert complete['cases']==164+unit['cases'] and ids(complete)==ids(regression)|ids(unit)
    for r in (complete,regression,unit):assert r['passed']==r['cases'] and r['failed']==r['errors']==r['skipped']==0
    for p in (OUT/'logs').glob('*.execution.json'):assert read(p)['exit_code']==0,p
    stdout=(OUT/'logs'/execution['command'][-1].split('=')[-1].replace('\\','/').split('/')[-1].replace('.xml','.stdout.log')).read_text()
    assert not re.search(r'\b(?:XFAIL|XPASS|SKIPPED|deselected)\b',stdout),'hidden outcome'
    coverage={}
    for classname,name in unit['case_ids']:
        match=re.match(r'test_head_t(\d+)',name)
        key=f'HEAD-T{int(match[1]):02}' if match else None
        match=re.match(r'test_loss_(fd|lr|cmc|fl|ar|kl)(\d+)',name)
        if match:key=f'LOSS-{match[1].upper()}{int(match[2]):02}'
        if key:coverage.setdefault(key,[]).append([classname,name])
    required={f'HEAD-T{i:02}' for i in range(1,11)}
    for name,number in (('FD',4),('LR',6),('CMC',7),('FL',5),('AR',4),('KL',6)):
        required|={f'LOSS-{name}{i:02}' for i in range(1,number+1)}
    assert set(coverage)==required
    write(OUT/'verification.json',{'status':'PASS','complete':complete,'regression':regression,'independent_unit':unit,
          'previous_164_case_identity_preserved':True,'earlier_119_present_and_passed':True,
          'all_42_unit_ids_covered':True,'unit_id_coverage':coverage,'no_skips_xfails_xpasses_or_deselections':True,
          'remaining_test_portions_deferred_to_2D2':{'HEAD-T08':'future composed-root ownership/serialization',
             'LOSS-LR05':'new full-FLT LR three-path gradient integration (toy oracle and accepted tie checks pass)',
             'LOSS-KL06':'new composed graph no-LD-dependency probe'},
          'JOINT_I01_I12_and_G2':'NOT RUN; separate handoff required'})
    write(OUT/'preservation.json',{'status':'PASS','baseline_files':len(baseline),'all_baseline_sha256_match':True,
          'mismatches':bad,'accepted_source_tests_configs_and_historical_evidence_unchanged':True,
          'prior_archives_preserved_and_hash_checked':[p for p in baseline if p.endswith('.zip')],
          'integration_gradient_history':'original bytes archived; fresh regression measurements captured separately; originals restored',
          'N1':'unchanged scoped MVE-T12 only','N2':'unchanged scoped FLT-T17 only'})
    heads=DetectionHeads();objective=SixObjectives();parameters=[*heads.named_parameters(),
                          *((f'objectives.{n}',p) for n,p in objective.named_parameters())]
    assert len({id(p) for _,p in parameters})==len({p.data_ptr() for _,p in parameters})==5
    assert sum(p.numel() for _,p in parameters)==263683
    report={**head_parameter_report(heads),'cmc_log_temperature':1,'new_subtotal':263683,
            'names_shapes_ownership':[{'name':n,'shape':list(p.shape),'numel':p.numel(),'requires_grad':p.requires_grad,
                                     'owner':'SixObjectives.cmc' if n.startswith('objectives.') else 'DetectionHeads'} for n,p in parameters],
            'objective_state_names':list(objective.state_dict()),'all_five_parameter_objects_and_storages_independent':True,
            'new_localization_or_reconstruction_parameters':0,
            'accepted_counts_regression_verified':{'MVE':32078912,'VD':9055365,'FLT':112758272,'visual':41134277,
                                                  'visual_plus_FLT':153892549},
            'conditional_composed_count_DERIVED_not_instantiated':154156232,
            'conditional_image_only_count_DERIVED_not_instantiated':41135303,
            'paper_approx_141M_discrepancy_DERIVED':13156232,
            'parameter_counts_do_not_establish_author_fidelity':True}
    write(OUT/'parameter_ownership.json',report)
    for profile in ('reference','debug'):
        config=load_heads_losses_config(ROOT/f'configs/heads_losses_{profile}.yaml')
        write(OUT/f'resolved_configs/{profile}.json',config.data)
        target=OUT/f'resolved_configs/{profile}.yaml';target.write_text(yaml.safe_dump(config.data,sort_keys=False),encoding='utf-8')
    write(OUT/'approval.json',{**load_heads_losses_config(ROOT/'configs/heads_losses_reference.yaml').data['approval'],
          'design_documents_preserved_as_historical_proposals':True,'superseding_record':'this approval and scoped configurations',
          'P2D15_status':'UNAPPROVED','P2D19_status':'UNAPPROVED','phase2d2_integration_and_G2':'DEFERRED; separate explicit handoff'})
    versions={name:importlib.metadata.version(name) for name in ('torch','pytest','PyYAML','psutil','ftfy','regex','wcwidth')}
    write(OUT/'runtime_versions.json',{'python':sys.version,'versions':versions,'torch_runtime':torch.__version__,
          'cuda_available':torch.cuda.is_available(),'tokenizer_pins_unchanged':'configs/flt_requirements.lock',
          'new_dependency_installations':False,'pretrained_model_weights_loaded':False,
          'bundled_regex_wheel':'cp314 Windows AMD64 only; other-platform wheel availability NOT VERIFIED; no substitutions'})
    diagnostics=[read(OUT/f'diagnostics/b{b}_{d}.json') for b in (1,2) for d in ('float32','float64')]
    assert all(d['status']=='PASS' and not d['graph_composition_with_accepted_modules'] and d['optimizer_updates']==0 for d in diagnostics)
    write(OUT/'loss_numerical_summary.json',{'scope':'independent controlled loss fixtures; not measured full-model losses',
          'uniform_analytical_baselines':{'FD':0.6931471805599453,'FL':0.6931471805599453,'LR':3328.82323108241},
          'diagnostics':diagnostics,'B1_CMC_exact_zero_loss_and_allocated_zero_gradients':True,
          'GPU_B1_B2_B8':'NOT RUN: CPU-only Torch build, no CUDA device',
          'CPU_full_six_loss_model_B1_B2_B8':'NOT RUN: Phase 2D.2 not authorized',
          'CPU_standalone_six_losses_B8':'NOT RUN: outside approved initial unit resource fixtures; no capacity inference',
          'CPU_head_only_B8':'PASS: HEAD-T01 shape test; not a full-graph memory measurement',
          'training_stability_convergence_accuracy':'NOT VERIFIED: no optimizer/updates/training/evaluation'})
    with (OUT/'loss_gradient_norms.csv').open('w',newline='',encoding='utf-8') as output:
        writer=csv.writer(output);writer.writerow(['batch','dtype','loss','input_or_parameter','l2','max_abs','nonzero','finite'])
        for d in diagnostics:
            for loss,r in d['records'].items():
                for name,g in r['gradients'].items():writer.writerow([d['batch_size'],d['dtype'],loss,name,g['l2'],g['max_abs'],g['nonzero'],g['finite']])
    # T17 remains a diagnostic: compare scientific data, do not turn it into a tolerance override.
    original_paths=[ROOT/p for p in baseline if '/t17_' in p and p.endswith('/cross_batch.json')]
    reference=read(original_paths[0]);new_paths=[p for p in (ROOT/'artifacts/phase2c2_n2_acceptance').glob('t17_*/cross_batch.json')
                                             if p.relative_to(ROOT).as_posix() not in baseline]
    n2=[]
    for path in new_paths:
        data=read(path);same=data['cross_batch']==reference['cross_batch']
        n2.append({'path':path.relative_to(ROOT).as_posix(),'cross_batch_statistics_exactly_match_preserved_T17':same,
                   'backend_matches':data['backend']==reference['backend'],'original_atol':data['atol'],'original_rtol':data['rtol'],
                   'N1_applied':data['N1_applied']})
        assert same and data['backend']==reference['backend'] and not data['N1_applied'],'changed N2 evidence needs review'
    write(OUT/'n2_regression_diagnostic_comparison.json',{'status':'PASS','reference':original_paths[0].relative_to(ROOT).as_posix(),
         'new_diagnostics':n2,'meaning':'Recorded numerical drift preserved; not a claim of harmless drift or training stability'})
    write_report(complete,regression,unit,diagnostics,len(baseline))
    # Full source/tests/assets/evidence payload; prior ZIPs stay unchanged on disk.
    paths=[]
    excluded={OUT/'manifest.json',OUT/'package_verification.json',OUT/'phase2d1_review_package.zip',OUT/'phase2d1_review_package.zip.sha256'}
    for directory in ('src','tests','configs','docs','tools','artifacts','papers'):
        for p in (ROOT/directory).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.zip') and p not in excluded:
                paths.append(p)
    paths.extend(p for p in (ROOT/'pyproject.toml',ROOT/'README.md') if p.exists())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    write(OUT/'manifest.json',{'phase':'2D.1','files_sha256':hashes,'suite_cases':complete['cases'],
          'source_config_test_and_fixture_hashes_included':True,
          'excluded':'nested previous ZIPs, current ZIP/hash/manifest/package-verification self-reference, pycache, installed dependency directory',
          'historical_manifests_remain_unmodified':True})
    archive=OUT/'phase2d1_review_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for name in hashes:package.write(ROOT/name,name)
        package.write(OUT/'manifest.json','artifacts/phase2d1/manifest.json')
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        mismatches=[name for name,h in hashes.items() if hashlib.sha256(package.read(name)).hexdigest()!=h]
        assert not mismatches
        assert len(package.namelist())==len(hashes)+1
    archive_hash=sha(archive)
    (OUT/'phase2d1_review_package.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n',encoding='utf-8')
    write(OUT/'package_verification.json',{'status':'PASS','archive':archive.name,'sha256':archive_hash,
          'bytes':archive.stat().st_size,'payload_files':len(hashes)+1,'crc_errors':0,'payload_sha256_mismatches':[],
          'prior_zip_files_preserved':True,'package_verification_record_external_to_zip':True})
    assert all(sha(ROOT/p)==h for p,h in baseline.items())
    print(json.dumps({'status':'PASS','new_cases':unit['cases'],'accepted_regressions':regression['cases'],
                      'complete_cases':complete['cases'],'unchanged_baseline_files':len(baseline),'zip_sha256':archive_hash}))


def write_report(complete,regression,unit,diagnostics,preserved):
    resources='\n'.join(f"| {d['batch_size']} | {d['dtype']} | {d['resources']['rss_baseline_bytes']/2**20:.1f} | {d['resources']['sampled_peak_rss_bytes']/2**20:.1f} | {d['resources']['elapsed_seconds']:.3f} | PASS |" for d in diagnostics)
    exemplar=next(d for d in diagnostics if d['batch_size']==2 and d['dtype']=='float64')
    gradients='\n'.join(f"| {name} | {key} | {g['l2']:.9g} | {g['max_abs']:.9g} |" for name,r in exemplar['records'].items() for key,g in r['gradients'].items())
    lr=exemplar['records']['LR'];cmc=exemplar['records']['CMC'];kl=exemplar['records']['KL']
    report=f'''# Phase 2D.1 implementation and verification report

Phase 2D.1 is complete and **awaits user review**. Independent heads/adapter and six executable objectives passed **{unit['passed']}/{unit['cases']} new cases**, the unchanged accepted regression passed **{regression['passed']}/164**, and the combined unfiltered suite passed **{complete['passed']}/{complete['cases']}**. Exit codes are 0; no failures, errors, skips, xfails, xpasses, deselections or tolerance changes. This is an independent-component result, not full-model six-loss integration, optimizer readiness, training stability, convergence or accuracy.

## Authorization, fidelity and preservation

The latest explicit user instruction approves P2D01-P2D14 and unit-verification portions of P2D16-P2D18/P2D20 **as engineering assumptions for Phase 2D.1 only**. [Scoped approval](../artifacts/phase2d1/approval.json) and [reference configuration](../configs/heads_losses_reference.yaml) record this approval; [debug configuration](../configs/heads_losses_debug.yaml) repeats the same approval and changes diagnostic settings only. P2D15 mask preprocessing and P2D19 optimizer execution remain unapproved. Full synthetic integration/G2 requires the separate Phase 2D.2 handoff. Earlier design documents remain unchanged historical proposals; the scoped record supersedes their approval status only within this authorization.

**EXPLICIT** paper reference: [MFVLR.pdf](../papers/MFVLR.pdf), Section III-E p. 7 Eq. (17), p. 8 Eqs. (18)-(27), single-FC detector/adapter and equal six-objective sum. **ASSUMED, approved engineering baseline**: biases, initialization, hard-index class semantics, probability/logit interpretations, all reductions absent from the paper, raw-dot resolution of conflicting similarity prose, log-temperature parameterization, both-sided KL and special/pad policy. **DERIVED** counts and controlled mathematical expectations are identified below. The original equations are preserved in [loss contract](phase2d_losses_contract.md); implementation never labels an executable interpretation as recovered author code.

SHA-256 preservation: **{preserved}/{preserved} pre-existing files unchanged**, including accepted modules, tests, configurations, tokenizer assets/golden fixtures and historical reports/logs/manifests/ZIPs. [Preservation audit](../artifacts/phase2d1/preservation.json) references the pre-edit [baseline](../artifacts/phase2d1/baseline_manifest.json). Unmodified integration regression tests write fixed historical gradient paths: the new runner archives the old bytes, captures fresh evidence separately, and restores those old bytes in `finally`. Accepted N1 remains MVE-T12-only; accepted N2 remains FLT-T17-only. All old 164 case identities and the earlier 119 are present and pass; no old test or fixture was edited.

## Source, interfaces and strict configuration

- [heads.py](../src/mfvlr/heads.py): `DetectionHeads` owns `detector: Linear(512,2,bias=True)` and independent `adapter: Linear(512,512,bias=True)`. Input `[B,1,512]`; outputs raw `[B,2]` Real/Fake logits (0/1) and unrestricted `[B,1,512]` adapted feature. `detect()` is language-independent; `align_language()` is explicit. Presentation-only softmax helpers never feed the approved losses automatically. No extra MD/AD, normalization, activation, dropout or pretrained weights.
- Dedicated CPU generator seed 0 draws detector then adapter weights using Xavier uniform gain 1, zero biases. Caller RNG is restored; accepted owners are never initialized or registered here. Analytical tests promote FP32-initialized weights to FP64.
- [objectives.py](../src/mfvlr/objectives.py): independent functions plus `CrossModalContrastiveLoss` and `SixObjectives`. Only one objective parameter, `cmc.log_temperature`, is registered. `SixObjectives` accepts already-produced tensors and supplied integer targets; it owns no visual/language backbone, detector or adapter. No composed production MFVLR model is introduced.
- [heads_losses_config.py](../src/mfvlr/heads_losses_config.py) and separate immutable snapshot: closed keys/types/values, unique YAML keys, exact shapes/reductions/weights/tolerances/approval; one diagnostics-only overlay level, whole diagnostics replacement, no nested base or changed architecture, no implicit self-approval. Resolved copies are [here](../artifacts/phase2d1/resolved_configs). Accepted schemas/snapshots remain unchanged. The verified execution route is a source checkout; production wheel packaging has not been audited or changed.

| Objective | Approved operation | Reference |
|---|---|---|
| FD | Class log-softmax/gather, B mean; raw logits, int64 Real/Fake IDs | p. 8 Eq. (26), P2D06 |
| LR | Vocabulary log-softmax/gather on `[B,308,49408]`; all token sum then B mean; unshifted IDs including BOS/EOS/pad | p. 8 Eqs. (24)-(25), P2D07 |
| CMC | Raw `V L^T / exp(theta)`, diagonal CE each direction, arithmetic mean; local batch only | p. 8 Eqs. (20)-(23), P2D08-10 |
| FL | Two-class pixel CE with supplied int64 background=0/foreground=1; BHW mean | p. 8 Eq. (18), P2D11/14 |
| AR | `(I-Ipre)^2` mean all RGB elements/images; I in [0,1], unrestricted finite Ipre; both input derivatives retained | p. 7 Eq. (17), P2D12 |
| KL | `KL(text || adapted vision)`, softmax features /0.5, sum 512 then B mean, both distributions differentiated, no T² | p. 8 Eq. (19), P2D13 |
| Total | Literal left-to-right `fd + lr + cmc + fl + ar + kl`, six coefficients 1 | p. 8 Eq. (27), P2D16 unit portion |

Every return is finite rank zero; all required shapes/devices/FP32-or-explicit-FP64 dtypes and integer ranges are validated. Nonfinite intermediates, exp(theta) under/overflow, or finite-input dot/square overflow fail visibly. There is no clipping, balancing, feature normalization, hidden epsilon, teacher detach, target smoothing or ignore mask. Supplied real masks must be zero; explicit EFS masks must be one; local masks may be mixed, empty or full and retain Fake labels. Consumer checks never generate, threshold, resize or infer a source-difference mask (P2D15 remains unimplemented).

## Verification scope and actual executions

All **42 applicable unit IDs** are covered: HEAD-T01-T10; LOSS-FD01-FD04, LR01-LR06, CMC01-CMC07, FL01-FL05, AR01-AR04, KL01-KL06. Parameterization and independent scalar/ownership units produce {unit['cases']} new pytest cases. [Coverage and exact case IDs](../artifacts/phase2d1/verification.json) distinguish unit scope from future integration portions.

Independent [fixtures](../tests/objectives/reference_fixtures.json) and [Python mathematical oracles](../tests/objectives/oracles.py) use explicit max/logsumexp, sums and hand-derived derivatives, with no production loss imported in the oracle. Full API dimensions are tested. FP64 analytical `atol=1e-10, rtol=1e-8`; FP32 `atol=1e-6, rtol=1e-5`; central finite difference epsilon 1e-6 and derivative `atol=1e-5, rtol=1e-3`. These preapproved thresholds were unchanged.

Tests verify affine sum-of-products and gradients, exact fixed-batch head independence/permutation/zero cross-sample derivatives, initialization/RNG/state ownership, invalid inputs, rank-zero reductions, full all-token LR derivative `(p-onehot)/B`, both-sided KL derivatives and finite extremes. The accepted `shift_low` method is probed directly using a vocabulary-only owner at all segment boundaries; loss targets remain unshifted. A clearly identified independent toy graph checks prompt lookup, direct BOS lookup and tied output gradient additivity. The unchanged full-FLT regressions verify production tying and serialization. New full-FLT LR three-path gradient composition, composed-root head ownership and KL no-LD-dependency probes are deferred to Phase 2D.2, together with all JOINT-I01-I12 and G2; no toy result is presented as a full-model integration pass.

CMC B=1 executes the full formula: exact zero scalar and allocated zero visual/text/theta gradients, with no special detached return. At B=2 controlled nonsymmetric fixtures yield nonzero gradients on all three operands. Deliberate contrastive cross-sample coupling is verified; it is distinct from exact head/model forward isolation. All six independently produced fixture inputs can be scalarized together and differentiated; this unit arithmetic graph contains none of MVE/VD/FLT.

Incremental runs: heads 25, FD/FL/AR 34, alignment 31, LR 17; aggregate new units {unit['cases']}; regression 164; combined {complete['cases']}. Every invocation returned 0 with no test failures. Raw stdout/stderr, command/environment/exit/runtime records and JUnit are under [logs](../artifacts/phase2d1/logs) / [JUnit](../artifacts/phase2d1/junit). Final [stdout](../artifacts/phase2d1/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d1/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d1/junit/complete_01.xml): **{complete['passed']} passed in {complete['seconds']:.3f} s**. Failed historical Phase 2A/2C runs and amendment proposals remain preserved; no new failing test attempt was suppressed.

Accepted FLT-T17 FP32 per-stage cross-batch diagnostic statistics and backend match the preserved T17 data exactly in these regression runs: [comparison](../artifacts/phase2d1/n2_regression_diagnostic_comparison.json). Required fixed-batch/permutation/cross-sample gradient and promoted-weight FP64 assertions remain active. This matching drift record is not proof that drift is harmless or that an optimizer trajectory will be stable.

## Parameter ownership

| New owner | Actual unique parameters |
|---|---:|
| Detector | 1,026 |
| Adapter | 262,656 |
| Head subtotal | 263,682 |
| CMC theta | 1 |
| New subtotal | **263,683** |

[Actual ownership report](../artifacts/phase2d1/parameter_ownership.json): five independent Parameter objects/storages; only theta is objective state. Accepted counts pass unchanged regressions: MVE 32,078,912; VD 9,055,365; FLT 112,758,272; visual+FLT 153,892,549. **DERIVED conditional** future composed count 154,156,232; image-only visual+detector 41,135,303. No full new root was instantiated. The conditional count exceeds the paper's approximately 141M by 13,156,232; the discrepancy remains unresolved. Dimensions were not changed to force agreement.

## Independent numerical evidence

[Machine-readable summary](../artifacts/phase2d1/loss_numerical_summary.json), four fresh-process [diagnostics](../artifacts/phase2d1/diagnostics), [gradient CSV](../artifacts/phase2d1/loss_gradient_norms.csv). These are controlled leaf-tensor experiments, **not paper-loss values measured on the accepted full model**.

Uniform baselines: FD/FL ln(2) = 0.69314718056; LR 308 ln(49,408) = 3,328.82323108. At B=2 FP64 the golden real prompt (repeated across the batch) has 552 pad, 16 special and 48 content tokens. Controlled target-score offsets give batch-mean category contributions padding {lr['categories']['padding']['batch_mean_nll_contribution']:.9f}, special {lr['categories']['special']['batch_mean_nll_contribution']:.9f}, content {lr['categories']['content']['batch_mean_nll_contribution']:.9f}, sum **{lr['loss']:.9f}**. The nonuniform value differs from the uniform baseline by design; padding and specials are not ignored.

CMC FP64 promoted theta {cmc['theta']:.12g} gives tau {cmc['tau']:.12g}; slight decimal deviation from 0.07 reflects FP32 initialization and is preserved in the oracle. Raw max score {cmc['raw_score_max']:.9g}, scaled max {cmc['scaled_score_max']:.9g}, loss {cmc['loss']:.9f}. Identical controlled feature rows at B=2 yield ln(2) despite diagonal-only positives; no duplicate deduplication/multi-positive labels. This is a duplicate-embedding fixture, not an assumption that identical prompt IDs necessarily yield identical visually conditioned LE features. B=1 is zero, so it does not provide an effective CMC learning signal.

KL probability fixture loss {kl['loss']:.9f}, teacher entropy {kl['teacher_entropy'][0]:.9f}, student entropy {kl['student_entropy'][0]:.9f}; each probability distribution sums to 1. Both gradients are nonzero on this asymmetric fixture. AR target 0/reconstruction 2 yields loss 4 with finite gradients and unchanged out-of-range reconstruction. Invalid/overflow probes reject intentionally bad inputs instead of modifying the mathematics.

Per-loss B=2 FP64 input/parameter gradient norms (separate graphs; scales depend on the registered fixtures):

| Loss | Variable | L2 | Max abs |
|---|---|---:|---:|
{gradients}

The large difference between LR's token sum and FD/FL means remains a training-readiness risk. These leaf gradients do not measure how the six objectives interact downstream through the unnormalized accepted LD, shared visual graph, reconstruction residual or common vocabulary. N2 and finite independent losses do not establish full-model gradient stability or finite optimizer updates. No additional LayerNorm, AMP, clipping, scaling, altered attention or architectural fix was introduced.

## Resources and limitations

Actual standalone full-shape six-loss diagnostics, sequential independent graphs in fresh processes, Windows CPU:

| B | Dtype | RSS baseline MiB | Sampled peak RSS MiB | Elapsed s | Result |
|---:|---|---:|---:|---:|---|
{resources}

RSS is process RSS sampled at nominal 5 ms intervals; actual sample counts are in JSON. Peaks are sampled observations, not exact allocator peaks or GPU capacity. Times include fixture construction/diagnostics/garbage collection, exclude interpreter/Torch startup. Four CPU threads, deterministic algorithms, highest FP32 matmul precision; backend and exact dependency provenance are logged. B=8 standalone head shape test passes, but B=8 six-loss diagnostics and full graphs are **NOT RUN**. CUDA B=1/2/8 are **NOT RUN**: CPU-only build/no device. New full-model six-loss B=1/2/8, G2, optimizer/G3/G4, repeated updates, datasets and benchmarks are **NOT RUN / unauthorized**. Accepted full FLT B=1/2 finiteness regressions pass, which is a distinct verification boundary.

## Reproducible review package and next boundary

[Review ZIP](../artifacts/phase2d1/phase2d1_review_package.zip), [SHA-256 manifest](../artifacts/phase2d1/manifest.json), [external package verification](../artifacts/phase2d1/package_verification.json), [reproduction instructions](../artifacts/phase2d1/REPRODUCE.md). The archive includes full actual source, accepted/new tests, scoped and accepted configurations, tokenizer source/BPE assets/golden fixtures, pinned tokenizer wheels and raw evidence including failed historical artifacts. Prior nested ZIPs are omitted from this ZIP to avoid redundant copies and remain unchanged in the workspace, with baseline hashes verified. CRC and every payload hash are verified. Current manifest/ZIP/package hash records avoid self-referential hashing.

Phase 2D.1 stops here for review. Phase 2D.2 full synthetic integration and G2 no-update tests require a separate handoff after acceptance. Dataset mask preprocessing and any optimizer/training/evaluation remain unapproved. Unresolved reproduction uncertainty includes the author architecture/141M gap, mask preprocessing and joint-loss/LD stability; standalone verification does not resolve them.
'''
    (ROOT/'docs/phase2d1_implementation_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
