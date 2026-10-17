"""Read-only numerical aggregation. Never constructs a model or runs backward.

Writes a new output directory; original evidence is never replaced. Ratios with
zero denominators remain null, and numerical FAILs are never execution PASSes.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from tools.d02a.common import OUT, compare, sha, write

KEYS = ('kind', 'precision', 'stage', 'cut', 'scale', 'parameter', 'role', 'fixture')
METRICS = ('outside_count', 'outside_proportion', 'max_abs_error', 'relative_l2',
           'error_l2', 'max_relative_nonzero_reference', 'sign_disagreements',
           'zero_reference_nonzero_actual')
CSV_FIELDS = ('job', 'family', 'j', 'batch', 'index', 'variant', 'threads', 'native',
              *KEYS, 'status', 'elements', 'atol', 'rtol', *METRICS,
              'actual_l2', 'reference_l2', 'cosine', 'rms', 'max_abs', 'l2',
              'singleton_QK', 'exact_structural_zero', 'extra_json')


def load_rows(path):
    with path.open(encoding='utf-8') as handle:
        for line in handle:
            yield json.loads(line)


def key(row):
    return tuple(row.get(k) for k in KEYS)


def jobkey(job, omit):
    return tuple(job[k] for k in ('family', 'j', 'batch', 'index', 'variant', 'threads', 'native') if k != omit)


def csvrow(job, row):
    value = {'job': job['id'], **{k: v for k, v in job.items() if k != 'id'}, **row}
    extras = {k: v for k, v in value.items() if k not in CSV_FIELDS}
    return {**{k: value.get(k) for k in CSV_FIELDS if k != 'extra_json'},
            'extra_json': json.dumps(extras, allow_nan=False, sort_keys=True)}


def change(a, b):
    if a is None or b is None:
        return 'UNDEFINED'
    return 'LOWER' if b < a else 'HIGHER' if b > a else 'EQUAL'


def main(destination):
    destination.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((OUT/'pilot_protocol.json').read_text())
    ln_inputs = json.loads((OUT/'LN_preexecution_input_manifest.json').read_text())['inputs']
    complete = []; job_status = []; failures = Counter(); comparison_counts = Counter()
    group_values = defaultdict(list); captures = []; backend_records = {}; resources = []
    raw_path = destination/'all_local_records.csv'
    with raw_path.open('x', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS); writer.writeheader()
        for job in protocol['jobs']:
            directory = OUT/'jobs'/job['id']; summary_path = directory/'summary.json'
            if not summary_path.exists():
                job_status.append({**job, 'execution': 'NOT RUN', 'numerical': 'NOT RUN'})
                continue
            summary = json.loads(summary_path.read_text())
            provenance = json.loads((directory/'provenance.json').read_text())
            assert provenance['job'] == job
            assert provenance['protocol_sha256'] == sha(OUT/'pilot_protocol.json')
            backend_records[job['id']] = provenance['backend']
            resources.append({**job, **summary['resources'],
                              'temporary_gradient_peak_bytes': summary.get('temporary_gradient_peak_bytes')})
            count = fail = 0; stages = defaultdict(set); structural = 0; cut_scales = set()
            for row in load_rows(directory/'comparisons.jsonl'):
                writer.writerow(csvrow(job, row)); count += 1
                group = (job['family'], job['variant'], row['kind'], row.get('precision', 'comparison'))
                if row.get('status') in ('PASS', 'FAIL'):
                    comparison_counts[group] += 1
                if row.get('status') == 'FAIL': fail += 1; failures[group] += 1
                for metric in METRICS:
                    if row.get(metric) is not None:
                        group_values[(*group, metric)].append(row[metric])
                if row['kind'] == 'activation': stages[row['precision']].add(row['stage'])
                if row['kind'] == 'LN_fixture_forward':
                    expected = ln_inputs[f"j{job['j']}_b{job['batch']}_{row['fixture']}"]['FP32_sha256']
                    assert row['input_sha256'] == expected
                if row.get('singleton_QK'):
                    assert row['allocated'] and row['exact_structural_zero']; structural += 1
                if row['kind'] == 'VJP_additivity':
                    cut_scales.add((row['precision'], row['cut'], row['scale']))
            assert count == summary['records'] and fail == summary['failed_comparisons']
            execution = 'FAIL' if summary['status'] == 'HARD_STOP' else 'PASS'
            job_status.append({**job, 'execution': execution, 'numerical': 'FAIL' if fail or execution == 'FAIL' else 'PASS',
                               'failed_comparisons': fail, 'records': count})
            if execution == 'PASS':
                assert summary['optimizer_steps'] == 0
                if job['family'] == 'NLD':
                    expected = {'state', 'memory', 'visual'} | {f'LD.{i}.{s}' for i in range(1,8) for s in ('self','memory','ff','vim','output')}
                    assert stages == {'float32': expected, 'float64': expected}
                    assert cut_scales == {(p,c,s) for p in ('float32','float64') for c in (1,4,7) for s in (0,1e-4,1,1024)}
                    assert structural == 192  # (1+4+7) blocks * Q/K * four scales * two precisions
                complete.append(job)
            captures.append({'job':job['id'], 'stages':{p:sorted(s) for p,s in stages.items()},
                             'singleton_zero_records':structural, 'records':count, 'failed_comparisons':fail})
    write(destination/'coverage.json', {'jobs':job_status, 'complete':len(complete),
          'registered':len(protocol['jobs']), 'not_run':sum(j['execution']=='NOT RUN' for j in job_status),
          'capture_integrity':captures, 'LN_fixture_prior_hashes_verified':True})
    write(destination/'backend_and_resources.json', {'backends':backend_records, 'resources':resources})
    summary_rows=[]
    for group,count in sorted(comparison_counts.items()):
        summary_rows.append(dict(zip(('family','variant','kind','precision'),group)) |
            {'comparisons':count,'failed':failures[group], 'metrics':{
                metric:{'count':len(group_values[(*group,metric)]),
                        'median':statistics.median(group_values[(*group,metric)]),
                        'maximum':max(group_values[(*group,metric)])}
                for metric in METRICS if group_values[(*group,metric)]}})
    write(destination/'comparison_summary.json', summary_rows)
    # Pair architectures only within the same registered fixture/backend/precision.
    # These describe metric changes, never prediction equality across architectures.
    bases={jobkey(j,'variant'):j for j in complete if j['variant']=='A0'}
    changes=defaultdict(Counter); paired=0; unpaired=0
    with (destination/'paired_variant_metrics.csv').open('x',newline='',encoding='utf-8') as handle:
        fields=('A0_job','variant_job',*KEYS,'metric','A0','variant','change')
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader()
        for job in complete:
            if job['variant']=='A0':continue
            base=bases.get(jobkey(job,'variant'))
            if base is None:continue
            original={key(r):r for r in load_rows(OUT/'jobs'/base['id']/'comparisons.jsonl')}
            for row in load_rows(OUT/'jobs'/job['id']/'comparisons.jsonl'):
                ref=original.get(key(row))
                if ref is None:unpaired+=1
                for metric in (*METRICS,'rms','max_abs'):
                    if metric not in row:continue
                    a=ref.get(metric) if ref else None; b=row.get(metric); label=change(a,b)
                    writer.writerow({'A0_job':base['id'],'variant_job':job['id'],
                        **{k:row.get(k) for k in KEYS},'metric':metric,'A0':a,'variant':b,'change':label})
                    changes[(job['family'],job['variant'],row['kind'],row.get('precision','comparison'),metric)][label]+=1
                    paired+=1
    write(destination/'paired_change_summary.json',{'metric_rows':paired,'variant_only_records':unpaired,
        'meaning':'LOWER/HIGHER are metric directions, not blanket improvement/acceptance. Added LN parameters have no A0 counterpart.',
        'groups':[dict(zip(('family','variant','kind','precision','metric'),g))|dict(c) for g,c in sorted(changes.items())]})
    # Same-variant B1/B2 forward differences; no new tolerance policy.
    batches={jobkey(j,'batch'):j for j in complete if j['batch']==2}; cross_count=0; cross_fail=0
    with (destination/'cross_batch.jsonl').open('x',encoding='utf-8') as handle, (destination/'cross_batch.csv').open('x',newline='',encoding='utf-8') as csv_handle:
        writer=csv.DictWriter(csv_handle,fieldnames=CSV_FIELDS);writer.writeheader()
        for job in complete:
            if job['batch']!=1:continue
            second=batches.get(jobkey(job,'batch'))
            if second is None:continue
            for precision in ('float32','float64'):
                if job['family']=='NLD':
                    with np.load(OUT/'jobs'/job['id']/f'activations_{precision}.npz') as a, np.load(OUT/'jobs'/second['id']/f'activations_{precision}.npz') as b:
                        assert set(a.files)==set(b.files)
                        pairs={name:(a[name],b[name][:1]) for name in a.files}
                else:
                    pairs={'composition':(np.load(OUT/'jobs'/job['id']/f'activation_{precision}.npy'),np.load(OUT/'jobs'/second['id']/f'activation_{precision}.npy')[:1])}
                for stage,(a,b) in pairs.items():
                    result={'kind':'same_variant_cross_batch','precision':precision,'stage':stage,
                            'B2_job':second['id'],**compare(a,b,precision)}
                    if stage in ('state','memory','visual'):assert np.array_equal(a,b)
                    handle.write(json.dumps({'B1_job':job['id'],**result},allow_nan=False)+'\n')
                    writer.writerow(csvrow(job,result));cross_count+=1;cross_fail+=result['status']=='FAIL'
    write(destination/'aggregation_record.json', {'status':'PASS','aggregation_source_sha256':sha(__file__),
        'protocol_sha256':sha(OUT/'pilot_protocol.json'),'complete_jobs':len(complete),
        'comparison_failures':sum(failures.values()),'cross_batch_comparisons':cross_count,
        'cross_batch_failed_comparisons':cross_fail,'models_constructed':0,'optimizer_steps':0,
        'original_evidence_modified':False,'thresholds_changed':False})
    print(json.dumps({'complete_jobs':len(complete),'comparison_failures':sum(failures.values()),'cross_batch_failures':cross_fail}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    main(parser.parse_args().out)
