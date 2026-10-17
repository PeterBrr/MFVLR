"""Publish honest Phase 2D.2 evidence, including mandatory numerical failures."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/phase2d2'
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src'),str(ROOT)]
from tools.finalize_phase2d1_review import suite,sha
from tools.phase2d2_diagnostics import write
from tests.joint.test_readiness import scientific


def read(path):return json.loads(path.read_text())


def main():
    import psutil
    import yaml
    baseline=read(OUT/'baseline_manifest.json')
    mismatches=[p for p,h in baseline.items() if not (ROOT/p).is_file() or sha(ROOT/p)!=h]
    assert not mismatches,mismatches
    complete=suite(OUT/'junit/complete_01.xml');regression=suite(OUT/'junit/regression_01.xml')
    previous=suite(ROOT/'artifacts/phase2d1/junit/complete_01.xml')
    old_ids={tuple(x) for x in previous['case_ids']}
    assert regression['cases']==regression['passed']==276
    assert {tuple(x) for x in regression['case_ids']}==old_ids
    assert old_ids<={tuple(x) for x in complete['case_ids']}
    assert complete['skipped']==0
    import xml.etree.ElementTree as ET
    failures=[];old_passed=0
    for case in ET.parse(OUT/'junit/complete_01.xml').getroot().findall('.//testcase'):
        node=case.find('failure')
        if node is None:node=case.find('error')
        key=(case.attrib['classname'],case.attrib['name'])
        if node is not None:failures.append({'id':list(key),'message':node.attrib.get('message',''),'traceback':node.text})
        elif key in old_ids:old_passed+=1
    assert old_passed==276
    new_count=complete['cases']-276
    status='BLOCKED' if failures else 'PASS_AWAITING_REVIEW'
    # Numerical failures are not removed, skipped, xfailed or reclassified as passes.
    write(OUT/'verification.json',{'status':status,'complete_suite':complete,'accepted_regression':regression,
          'accepted_276_passed_unchanged':True,'new_cases':new_count,'failed_cases':failures,
          'no_skips_xfails_or_deselections':True,'tolerances_unchanged':True,'N2_applied_to_new_gradient_assertions':False})
    write(OUT/'preservation.json',{'status':'PASS','files':len(baseline),'all_baseline_sha256_match':True,'mismatches':mismatches,
           'historical_failed_logs_manifests_configs_assets_and_ZIPs_preserved':True,
           'new_failed_runs_preserved':['g2_real','joint_01','all_final_G2_cases','readiness_01','complete_01'],
           'accepted_gradient_files_restored_after_regression':True})
    approval={'phase':'2D.2','status':'APPROVED','classification':'ASSUMED','author_architecture_confirmed':False,
        'scope':'P2D04/P2D16-P2D18/P2D20 full integration and G2 no-update verification only',
        'P2D01_P2D14':'previously approved engineering interpretations reused unchanged',
        'P2D15':'UNAPPROVED','P2D19':'UNAPPROVED','optimizer_updates':0,'datasets':False,'benchmarks':False,
        'implementation_approval_is_not_acceptance':True,'verification_acceptance':status,
        'new_numerical_amendment':'NONE; neither proposed as approved nor applied',
        'authority':'history/user_phase2d2_request.txt'}
    write(OUT/'approval.json',approval)
    ownership=read(OUT/'joint/I12.json');write(OUT/'parameter_ownership_serialization.json',ownership)
    assert ownership['full']['total_unique']==154156232 and ownership['deployment']['total_unique']==41135303
    versions={n:importlib.metadata.version(n) for n in ('torch','pytest','PyYAML','psutil','numpy','ftfy','regex','wcwidth')}
    write(OUT/'runtime_versions.json',{'python':sys.version,'versions':versions,'new_dependency_installations':False,
          'tokenizer_pins_and_assets':'unchanged; Windows cp314 regex wheel only bundled; other platform wheels NOT VERIFIED'})
    for profile in ('reference','debug'):
        from mfvlr.integration_config import load_integration_config
        c=load_integration_config(ROOT/f'configs/integration_{profile}.yaml')
        write(OUT/f'resolved_configs/{profile}.json',c)
    comparisons=[]
    for i in range(3):
        for dtype in ('float32','float64'):
            name=f'fixture{i}_{dtype}_threads4.json'
            fresh=read(OUT/'n2'/name);old=read(ROOT/'artifacts/phase2c2_n2_acceptance/diagnostics'/name)
            same=scientific(fresh)==scientific(old)
            comparisons.append({'fixture':i,'dtype':dtype,'all_scientific_fields_and_backend_exactly_preserved':same,
                                'excluded_fields':['elapsed_seconds','seconds'],'status':fresh['status']})
    assert all(r['all_scientific_fields_and_backend_exactly_preserved'] for r in comparisons)
    write(OUT/'n2_comparison.json',{'status':'PASS','six_original_diagnostics':comparisons,
          'old_N1_and_N2_scope_unchanged':True,'new_gradient_gate_failure_not_exempted':True})
    data=[]
    for case in ('real','efs','local','real_efs','real_local','local_duplicate'):
        directory=OUT/'g2'/('final_'+case)
        summary=read(directory/'summary.json');resources=read(directory/'resources.json')
        joint=read(directory/'total.json');comparison=read(directory/'gradient_additivity.json')
        row={'case':case,'summary':summary,'resources':resources,'joint':joint,'comparison':comparison}
        data.append(row)
        write(OUT/f'fixtures/{case}.json',summary['fixture'])
    write(OUT/'g2_summary.json',{'status':status,'untrained_synthetic_full_graph':True,'fixtures':[
        {'name':d['case'],'batch':d['summary']['fixture']['batch'],'status':d['summary']['status'],
         'losses':d['summary']['losses'],'joint_gradient_l2':d['comparison']['joint_l2'],
         'sum_gradient_relative_L2_difference':d['comparison']['relative_l2_difference'],
         'outside_tolerance_elements':sum(p['outside_tolerance_count'] for p in d['comparison']['parameters'].values()),
         'failing_parameter_tensors':sum(not p['allclose'] for p in d['comparison']['parameters'].values()),
         'LE_final_max':d['joint']['activations']['LE11']['max_abs'],'LD_final_max':d['joint']['activations']['LD6']['max_abs'],
         'logit_max':d['joint']['outputs']['tensors']['logits_lr']['max_abs'],'resources':d['resources']}
        for d in data]})
    with (OUT/'module_gradients.csv').open('w',newline='',encoding='utf-8') as output:
        writer=csv.writer(output);writer.writerow(['fixture','loss','module','allocated_parameters','L2','max_abs','nonzero_elements'])
        for d in data:
            for loss in ('fd','lr','cmc','fl','ar','kl','total'):
                r=read(OUT/'g2'/('final_'+d['case'])/f'{loss}.json')
                for name,g in r['gradients']['modules'].items():writer.writerow([d['case'],loss,name,g['allocated_tensors'],g['l2'],g['max_abs'],g['nonzero_elements']])
    resources=read(OUT/'resource_availability.json');resources['available_RAM_after_diagnostics_bytes']=psutil.virtual_memory().available
    write(OUT/'resource_availability.json',resources)
    write_report(status,complete,regression,new_count,failures,data,len(baseline),versions)
    excluded={OUT/'manifest.json',OUT/'package_verification.json',OUT/'phase2d2_review_package.zip',OUT/'phase2d2_review_package.zip.sha256'}
    paths=[]
    for directory in ('src','tests','configs','docs','tools','artifacts','papers'):
        paths.extend(p for p in (ROOT/directory).rglob('*') if p.is_file() and p not in excluded and
                     '__pycache__' not in p.parts and p.suffix not in ('.pyc','.zip'))
    paths.extend(ROOT/p for p in ('README.md','pyproject.toml') if (ROOT/p).is_file())
    hashes={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}
    write(OUT/'manifest.json',{'phase':'2D.2','verification_status':status,'files_sha256':hashes,
           'excluded':'prior ZIPs (unchanged on disk), current ZIP/hash/manifest/verification self-reference, pycache, installed dependency folder'})
    archive=OUT/'phase2d2_review_package.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as package:
        for p in hashes:package.write(ROOT/p,p)
        package.write(OUT/'manifest.json','artifacts/phase2d2/manifest.json')
    with zipfile.ZipFile(archive) as package:
        assert package.testzip() is None
        assert all(hashlib.sha256(package.read(p)).hexdigest()==h for p,h in hashes.items())
        count=len(package.namelist())
    archive_hash=sha(archive);(OUT/'phase2d2_review_package.zip.sha256').write_text(f'{archive_hash}  {archive.name}\n')
    write(OUT/'package_verification.json',{'status':'PASS','phase_verification_status':status,'bytes':archive.stat().st_size,
          'files':count,'sha256':archive_hash,'CRC_errors':0,'payload_SHA256_mismatches':[],'current_record_external_to_zip':True})
    assert all(sha(ROOT/p)==h for p,h in baseline.items())
    print(json.dumps({'status':status,'complete_cases':complete['cases'],'passed':complete['passed'],'failed':complete['failed'],
                      'accepted_regressions':276,'new_cases':new_count,'preserved_files':len(baseline),'zip':str(archive)}))


def write_report(status,complete,regression,new_count,failures,data,preserved,versions):
    loss_rows='\n'.join('| '+d['case']+' | '+str(d['summary']['fixture']['batch'])+' | '+
        ' | '.join(f"{d['summary']['losses'][n]:.9g}" for n in ('fd','lr','cmc','fl','ar','kl','total'))+' |' for d in data)
    numerical_rows='\n'.join(f"| {d['case']} | {sum(p['outside_tolerance_count'] for p in d['comparison']['parameters'].values()):,} | "+
        f"{d['comparison']['relative_l2_difference']:.9g} | {d['joint']['activations']['LE11']['max_abs']:.6f} | "+
        f"{d['joint']['activations']['LD6']['max_abs']:.6f} | {d['comparison']['joint_l2']:.9g} |" for d in data)
    resource_rows='\n'.join(f"| {d['case']} | {d['summary']['fixture']['batch']} | {d['resources']['sampled_peak_rss_bytes']/2**20:.3f} | {d['resources']['elapsed_seconds']:.3f} | finite execution; G2 FAIL |" for d in data)
    failed_names='\n'.join('- `'+f['id'][0]+'::'+f['id'][1]+'`' for f in failures)
    report=f'''# Phase 2D.2 implementation and G2 verification report

**Status: {status}; Phase 2D.2 is not accepted.** The full orchestration is implemented. The final unfiltered suite discovered **{complete['cases']}** cases: **{complete['passed']} passed, {complete['failed']} failed, {complete['errors']} errors, {complete['skipped']} skips**; exit code 1. The separate unchanged regression run passed **276/276**, exit 0. No xfails, hidden skips, deselections, relaxed thresholds, architecture changes or stabilization methods were used.

All six requested synthetic full-size FP32 fixtures completed their six isolated losses and literal joint backward with finite scalars, features, logits and parameter/intermediate gradients. Every LE/LD block is captured, all connected singleton VIM Q/K gradients remain allocated exact zeros, and weights/identities remain unchanged. **Every fixture fails the full-parameter isolated-sum versus joint-gradient comparison at the unchanged FP32 tolerance.** A separate CMC chain-rule comparison fails at 3/1,024 coordinates. Finite execution is not a passed G2 numerical gate, training stability, optimizer readiness or benchmark performance.

## Scope, approval and preserved baseline

The latest [user request](../artifacts/phase2d2/history/user_phase2d2_request.txt) extends engineering approval only to P2D04/P2D16-P2D18/P2D20 full integration and G2. [Scoped approval](../artifacts/phase2d2/approval.json) records ASSUMED engineering status, not recovered author architecture. P2D15 mask preprocessing and P2D19 optimizer/repeated updates remain UNAPPROVED. No optimizer, scheduler, parameter update, training step, dataset loader or evaluation was implemented. G3/G4 remain future work.

All **{preserved} pre-existing files are byte-identical by SHA-256**, including all accepted MVE/VD/tokenizer/VIM/FLT/head/objective source, 276 tests, configurations, assets, golden fixtures, logs/manifests/review ZIPs and original failures. [Baseline](../artifacts/phase2d2/baseline_manifest.json), [audit](../artifacts/phase2d2/preservation.json). Existing accepted tests emit fixed historical gradient files: the new runner archives old bytes, captures fresh data separately and restores old bytes in `finally`. All 276 old case identities pass in both regression and complete invocations. N1 stays MVE-T12-only; N2 stays FLT-T17-only.

## Full orchestration and physical deployment

[integration.py](../src/mfvlr/integration.py) owns one `MVE`, one bottleneck-only `VisionDecoder`, one accepted `DetectionHeads`, one full `FineGrainedLanguageTransformer`, and one `SixObjectives`. It does not register the plain accepted external orchestrator or duplicate its children. Appearance encoding runs once, VD trunk once, abs reconstruction residual runs through the same MVE, and appearance/residual global features add to raw Iv. Detector, adapter and FLT consume that same raw fusion. The existing two-channel MD and RGB AD are reused, not duplicated. No production detach, feature normalization, rescaling, gating, activation change or temperature change is introduced.

`forward(image)` / `visual_only(image)` return visual/VD output plus raw Real/Fake logits without tokens, targets, FLT, adapter or losses. `with_language(image, token_ids, sample_ids=...)` explicitly opts into full FLT and adapter, including diagnostic eval-mode use. `compute_objectives(predictions, SuppliedTargets)` validates unique ordered sample IDs, original unshifted token IDs and the accepted strict objective input/target semantics. Per-call records are caller-owned tensors, not deep immutable snapshots or root caches; callers must not mutate or reuse records across unrelated inputs/updates.

`ImageOnlyMFVLR` physically instantiates only MVE, VD and a detector. Its dedicated detector initialization reproduces exactly the first accepted head-generator draw; it does not instantiate an adapter/FLT/objective then discard them. [Actual ownership/serialization report](../artifacts/phase2d2/parameter_ownership_serialization.json) verifies full **154,156,232** and image-only **41,135,303** unique parameters, no duplicate registered parameter/storage owners, exact tied vocabulary identity, save/load identity preservation and exact same-runtime logits. Temporary full checkpoints are tested and cleaned; no trained weights are claimed or bundled.

New [reference](../configs/integration_reference.yaml) / [debug](../configs/integration_debug.yaml) configurations and closed [validator](../src/mfvlr/integration_config.py) preserve the immutable approved seed-0 topology and objective policies. Accepted configurations remain unchanged. New roots use the exact accepted constructors/initialization; configuration cannot grant P2D15/P2D19 or alter counts/tolerances. Resolved copies and source hashes are packaged. Source-checkout execution is verified; wheel packaging is not modified or audited.

Paper traceability remains [heads contract](phase2d_heads_contract.md), [loss/equation ledger](phase2d_losses_contract.md), [numerical readiness](phase2d_numerical_readiness.md), [test plan](phase2d_test_plan.md). **EXPLICIT** paper anchors: Fig.4 p.5, III-B/C pp.5-6 shared appearance/residual encoding and decoder roles; III-D pp.6-7 Eqs.(3)-(16) language/VIM; III-E Eq.(17) p.7 and Eqs.(18)-(27) p.8. Executable reductions/probability interpretations, biases/initialization, baseline architecture and orchestration are **ASSUMED, approved engineering choices**. Counts are **DERIVED**, not evidence of author fidelity. The approximately 141M versus 154,156,232 discrepancy remains unresolved; no dimension was reduced to match it. Published equations are unchanged in the preserved reference ledger.

## Synthetic fixtures and full-graph losses

[Fixture definitions](../tools/phase2d2_fixtures.py), [literal provenance/hash records](../artifacts/phase2d2/fixtures). Full RGB224x224 base values use `0.1+0.8*rand`, seeds real17/EFS412/local997. Local changes complement RGB inside `[64:160,80:176]`; the exact rectangle is an oracle mask, not dataset threshold preprocessing. EFS masks are all one, real masks zero. Duplicate local/local uses seed997 and seed412 with identical DiffFace prompt IDs (golden case5) and distinct image perturbations. Using seed412 for the second local is a recorded fixture resolution within the specified seed set, not a recovered paper data protocol. Token IDs come from unchanged golden real/DDPM/DiffFace cases0/1/5 and are independently verified against the accepted builder/tokenizer. Sample IDs remain unique despite duplicated prompts.

The following are **actual losses on the untrained full synthetic model graph**, not benchmark/detection performance or measured training:

| Fixture | B | FD | LR | CMC | FL | AR | KL | Literal total |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{loss_rows}

All six raw components retain the exact approved interpretations and unit weights, in order FD/LR/CMC/FL/AR/KL. B=1 CMC is the exact connected-zero formula, not a detached constant. B=2 local negatives/diagonal positives are unchanged; duplicated prompts are not deduplicated or relabeled. Per-case outputs record pad/BOS/EOS/content LR sums/counts, CMC raw/scaled scores/tau/row-column entropy/duplicate counts, KL teacher/student entropy and feature masses, Ipre unrestricted range/outside-[0,1] counts and residual positive/negative/equality coverage. These are detached statistics, not additional production normalizations or loss balancing.

## Actual tests and failure history

New discovery: **{new_count} cases** covering JOINT-I01-I12 and READY-N01-N07, including parameterizations. Complete logs/JUnit include every mandatory old/new case, with no filtering. [Raw final stdout](../artifacts/phase2d2/logs/complete_01.stdout.log), [stderr](../artifacts/phase2d2/logs/complete_01.stderr.log), [JUnit](../artifacts/phase2d2/junit/complete_01.xml), [verification/case IDs](../artifacts/phase2d2/verification.json). Original real-case failure, the initial 19-pass/2-fail JOINT run, initial diagnostic source snapshots, all later failed G2 worker exits and final failed pytest logs are preserved.

Passed JOINT controls verify one-owner DAG, exact loss input pairing/literal sum, gradient presence matrix, FD/LR/CMC/KL reconstruction-mediated AD gradients and detached controls, AR/residual derivative additivity and abs zero subgradient, localization/appearance isolation/shared-trunk accumulation, two-sided KL/visual conditioning/no-LD path, all19 VIM instances and 38 exact-zero Q/K task gradients, text-free inference and actual full-root serialization. Independent FP64 CMC partial-derivative and small FP64 sum oracles pass. CMC partial/cross-sample-coupling checks execute before its strict full FP32 chain assertion; metadata is retained even when that assertion fails.

Final failing cases:

{failed_names}

JOINT-I08: full FP32 `dCMC/dIv` versus direct plus LE-mediated partial derivatives fails 3/1,024 coordinates; initial pytest reports maximum mismatched absolute difference 2.288818359375e-5 and maximum mismatched relative difference 4.72433494e-5. [Detailed comparator](../artifacts/phase2d2/joint/I08.json) separately reports global maxima/relative-L2/counts, which need not equal pytest's maxima restricted to mismatched coordinates. Independent FP64 derivative mathematics and deliberate objective batch coupling pass.

JOINT-I10 and all six READY-N02 cases: a streamed FP64 sum of already rounded isolated FP32 gradients disagrees with the joint FP32 backward under atol=1e-6/rtol=1e-5. Reference is cast once to FP32 for elementwise `isclose`; high-precision error/L2 diagnostics use the uncast accumulator. No N2 exemption or threshold modification applies.

| Fixture | Outside-tolerance parameter elements | Relative L2 joint vs isolated sum | Final LE max | Final LD max | Joint gradient L2 |
|---|---:|---:|---:|---:|---:|
{numerical_rows}

The initial real case had 8,772,234 outside-tolerance elements across 293 parameter tensors, global relative L2 about 9.169735e-7; some weak ordinary LE attention Q/K gradients have much larger relative discrepancies. Global norm alone cannot establish harmlessness. Original failure history remains distinct from the later characterization run adding gradient-cosine/contribution statistics.

## Diagnosis and numerical preservation

[Gradient review](../artifacts/phase2d2/gradient_review/review.json) verifies the streamed file's multi-chunk offsets and arithmetic using an independent exact integer-gradient control. On one full real forward graph, direct FD/LR/mask/adapter prediction derivatives match their isolated sums at the unchanged tolerance. Upstream Iv/Ipre comparisons fail. All seven deterministic real forward-statistic records and isolated/joint scalar values match exactly; parameters do not update. This supports an FP32 backward accumulation/rounding-amplification explanation in shared upstream Jacobians. It does not prove every discrepancy harmless, exclude every numerical risk, or establish full promoted-weight FP64 *gradient-additivity* for this new visual graph. No hidden architectural correction or automatic numerical-policy proposal was applied.

The first G2 gate stopped for diagnosis. Subsequent independent fixtures ran as explicitly labeled numerical characterization after the diagnosed finite failure; **the acceptance gate remained BLOCKED throughout**, with each unchanged allclose assertion failing honestly. Any nonfinite/new failure category would halt remaining characterization. There are no parameter updates between cases or calls.

Original N2 F0/F1/F2 were re-executed in both FP32 and promoted-weight FP64, using the unchanged helper, seeds, initial weights, inputs and backend. [Comparison](../artifacts/phase2d2/n2_comparison.json): every scientific field/backend matches preserved accepted evidence exactly, excluding elapsed timing fields only. Original exact fixed-batch/permutation/cross-sample checks and FP64 cross-batch criteria remain active. Old helper PROPOSED labels remain historical; the accepted scoped N2 record is not rewritten. Original N2 agreement does not waive the new failures or prove training stability.

Per-loss/joint [module-gradient CSV](../artifacts/phase2d2/module_gradients.csv), complete parameter gradients and intermediary path norms are preserved under [G2](../artifacts/phase2d2/g2). Selected detached gradient subsets supply pairwise cosine/contribution/cancellation statistics; unused/zero gradients have undefined cosine, not zero. Large LR token-summed values and gradients can dominate other tasks; raw magnitude alone is not a performance claim. The accepted unnormalized LD sensitivity remains unresolved. No LayerNorm, AMP, rescaling, balancing, clipping, changed attention/temperature or teacher detach was introduced.

## Resources and graph lifecycle

All requested CPU B=1/2 cases ran separately in fresh processes. Sampled process RSS (nominal5ms; actual sample count in JSON) and wall time include model construction, seven fresh full forward/backward graphs, reductions, hash checks and streamed diagnostic storage; they are not pure throughput or exact allocator-peak measurements.

| Fixture | B | Sampled peak RSS MiB | Wall s | Status |
|---|---:|---:|---:|---|
{resource_rows}

Batch8 CPU: **NOT RUN**, explicit [resource-gating record](../artifacts/phase2d2/resource_availability.json); available RAM during diagnostics was about 2GiB while even B=1/2 consume multi-GiB working sets, safe B=8 headroom is unestablished and strict G2 is already blocked. No accumulation surrogate or invented OOM/capacity result. GPU B=1/2/8: **NOT RUN**, CPU-only Torch/no CUDA device. READY-N06/07 passing validates reporting of these facts, not execution of unrun capacities.

No six full graphs coexist: each loss gets a fresh deterministic forward, records detached statistics, backpropagates, streams its parameter gradients, releases outputs/hooks and clears gradients. Weakrefs verify requested outputs are gone; all forward hooks are removed. A temporary1,233,249,856-byte FP64 file stores the sum in262,144-element chunks; selected detached gradient vectors are small diagnostic subsets, not graphs. File/selected tensors close/release in `finally`. Full production tensors/gradients remain FP32. Weights and Parameter identities are hash-verified unchanged. This is no-update execution, not repeated optimizer training.

## Reproducibility and review boundary

Recorded versions: {versions}. [Reproduction guide](../artifacts/phase2d2/REPRODUCE.md), [review ZIP](../artifacts/phase2d2/phase2d2_review_package.zip), [payload manifest](../artifacts/phase2d2/manifest.json), [external ZIP verification](../artifacts/phase2d2/package_verification.json). Archive includes full source, all accepted/new tests/configs, tokenizer/BPE assets/golden fixtures/pinned Windows wheels, original paper, complete raw evidence and historical failures/source snapshots. Previous nested ZIPs are omitted only from the new archive to avoid redundant copies and remain unchanged on disk. ZIP CRC and every payload hash are checked; current manifest/ZIP/verification avoid self-reference. Different-platform pinned regex wheels are not verified or silently substituted.

Phase 2D.2 stops here for review. It cannot be accepted under the current numerical criteria. A further explicitly scoped numerical review or approved remedy is needed before acceptance; this report does not approve a policy amendment or architectural stabilization. P2D15/P2D19/G3/G4, datasets, training and benchmarks remain unapproved.
'''
    (ROOT/'docs/phase2d2_implementation_report.md').write_text(report,encoding='utf-8')


if __name__=='__main__':main()
