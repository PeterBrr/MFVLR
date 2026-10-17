"""Collect recorded measurements, immutable preservation and resolved configs."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'.phase2c1_deps'),str(ROOT/'src')]
from mfvlr.flt_config import load_flt_config
from mfvlr.config import load_config
from mfvlr.decoder_config import load_decoder_config
from mfvlr.foundation_config import load_foundation_config

OUT=ROOT/'artifacts/phase2c2'


def write(name,value):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2)+'\n',encoding='utf-8')


def main():
    old=json.loads((ROOT/'artifacts/phase2c1/manifest.json').read_text())
    mismatches=[p for p,h in old['files_sha256'].items()
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h]
    write('preservation.json',{'baseline_manifest':'artifacts/phase2c1/manifest.json',
        'files_verified':len(old['files_sha256']),'mismatches':mismatches,
        'files_sha256':old['files_sha256']})
    if mismatches:raise RuntimeError(f'Accepted files changed: {mismatches}')
    loaders={'flt':load_flt_config,'mve':load_config,'vd':load_decoder_config,
             'foundation':load_foundation_config}
    names={'flt':('flt_reference.yaml','flt_debug.yaml'),
           'mve':('paper_reference.yaml','debug.yaml'),
           'vd':('decoder_reference.yaml','decoder_debug.yaml'),
           'foundation':('foundation_reference.yaml','foundation_debug.yaml')}
    for group,paths in names.items():
        for profile,name in zip(('reference','debug'),paths):
            write(f'resolved_configs/{group}_{profile}.json',loaders[group](ROOT/'configs'/name).data)
    resources=json.loads((OUT/'resources.json').read_text())
    integrated=next(c for c in resources['cases'] if c['route']=='integrated' and c['batch']==1)
    parameters=integrated['parameter_report']
    assert parameters['total_unique']==112758272
    assert integrated['parameter_bytes']//4==153892549
    write('parameter_report.json',{'classification':'approved engineering baseline; not author confirmation',
        'flt':parameters,'mve_unique':32078912,'vd_unique':9055365,'visual_unique':41134277,
        'combined_visual_flt_unique':integrated['parameter_bytes']//4,
        'paper_reported_approximate':141000000,'excess_before_future_heads':12892549,
        'output_weight_same_parameter':True,'output_bias_parameters':0,
        'source':'Actual module inventory in fresh integrated CPU resource worker; tests FLT-T14/I07'})
    traces=[];table=[]
    for c in resources['cases']:
        traces.append({k:c[k] for k in ('route','batch','status','trace','gradients',
            'logits','encoder','decoder','logits_shape','live_refs_after_each_backward')})
        table.append({'route':c['route'],'batch':c['batch'],'status':c['status'],
            'sampled_peak_rss_MiB':c['rss_sampled_peak_bytes']/1024**2,
            'lifetime_peak_rss_MiB':c['platform_lifetime_peak_bytes']/1024**2,
            'forward_seconds_mean':statistics.mean(c['forward']['seconds']),
            'forward_backward_seconds_mean':statistics.mean(c['forward_backward']['seconds']),
            'logits_MiB':c['logits_bytes']/1024**2,'parameter_MiB':c['parameter_bytes']/1024**2,
            'gradient_MiB':c['gradient_bytes']/1024**2})
    write('forward_backward_traces.json',{'source':'resources.json, actual final measured iteration',
        'cases':traces,'gradient_probe':'mean(logits^2)+mean(LD_hidden^2)+mean(LE_global^2); test instrumentation only'})
    write('resource_summary.json',{'environment':resources['environment'],'cases':table,
        'gpu':resources['gpu'],'batch8':resources['batch8'],
        'methodology':'Fresh processes; FP32; four threads; one warmup plus three measured iterations per mode. RSS sampled every 2 ms including diagnostic overhead, no optimizer. Short peaks may be missed; lifetime peak also recorded.'})
    expected={'ftfy':'6.3.1','regex':'2026.6.28','wcwidth':'0.2.13'}
    installed={n:importlib.metadata.version(n) for n in expected}
    assert installed==expected
    lock=ROOT/'configs/flt_requirements.lock'
    assert lock.read_bytes()==(ROOT/'configs/foundation_requirements.lock').read_bytes()
    write('dependency_report.json',{'required':expected,'installed':installed,'silent_substitutions':False,
        'python':sys.version,'wheel_manifest':'artifacts/phase2c1/dependency_wheels.json',
        'lock_sha256':hashlib.sha256(lock.read_bytes()).hexdigest(),
        'bundled_regex_wheel_platform':'CPython 3.14 Windows amd64',
        'other_python_or_platform_wheels':'UNAVAILABLE IN BUNDLED ARTIFACTS; no availability claim or substitute version',
        'model_weights_loaded':False,'tokenizer_assets_verified_by_resolved_loaders':True})
    print(json.dumps({'preserved':len(old['files_sha256']),'resource_summary':table},indent=2))


if __name__=='__main__':main()
