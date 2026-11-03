"""Run experiment configs over paired seeds; completed runs are skipped, partial runs resume."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / '.pilot_deps'), str(ROOT / 'src')]
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from mfvlr_research.experiment import compare_reports, run_experiment, validate_config, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs', type=Path, nargs='+', required=True, help='The first config is the paired baseline')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--budget-seconds', type=int, help='Pause with a checkpoint after this long, per run')
    parser.add_argument('--max-updates', type=int, help='Smoke limit')
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error('Seeds must be unique and nonnegative')
    configs = [validate_config(json.loads(path.read_text(encoding='utf-8'))) for path in args.configs]
    if len({config['name'] for config in configs}) != len(configs):
        parser.error('Config names must be unique')
    reports = {config['name']: {} for config in configs}
    # Finish one seed across all arms first, so an interruption still leaves paired runs.
    for seed in args.seeds:
        for config in configs:
            folder = args.output / config['name'] / f'seed_{seed}'
            report_path, status_path = folder / 'report.json', folder / 'status.json'
            if status_path.exists() and json.loads(status_path.read_text(encoding='utf-8'))['status'] == 'COMPLETED':
                reports[config['name']][seed] = json.loads(report_path.read_text(encoding='utf-8'))
                print(json.dumps(dict(event='skip_completed', name=config['name'], seed=seed)), flush=True)
                continue
            print(json.dumps(dict(event='run_start', name=config['name'], seed=seed)), flush=True)
            report = run_experiment(config, seed, folder, root=ROOT, device=args.device,
                                    budget_seconds=args.budget_seconds, max_updates=args.max_updates)
            if report['status'] == 'COMPLETED':
                reports[config['name']][seed] = report
            print(json.dumps(dict(event='run_end', name=config['name'], seed=seed, status=report['status'])), flush=True)
    baseline = configs[0]['name']
    summary = {config['name']: compare_reports(reports[baseline], reports[config['name']])
               for config in configs[1:] if reports[baseline] and reports[config['name']]}
    write_json(args.output / 'summary.json', dict(baseline=baseline, primary='mean validation IoU of the last 3 epochs',
                                                  test_used=False, comparisons=summary))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
