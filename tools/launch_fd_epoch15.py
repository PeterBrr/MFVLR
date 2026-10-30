"""Launch three 15-epoch FD0.1 trials using the existing private data bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    args = parser.parse_args()
    old = ROOT/'artifacts/genface_fd_validation_kaggle_20261026/notebook_001'
    root = ROOT/'artifacts/genface_fd_epoch15_kaggle_20261027'
    folder = root/'notebook_001'
    folder.mkdir(parents=True, exist_ok=True)
    runner = (ROOT/'tools/run_fd_validation_pilot.py').read_text(encoding='utf-8')
    compile(runner, 'runner.py', 'exec')
    script = (old/'run.py').read_text(encoding='utf-8')
    marker = "import torch\n"
    assert script.count(marker) == 1
    patch = ("# Explicit runner update: configurable FD weights; dataset and model unchanged.\n"
             f"runner_source={runner!r}\n"
             f"assert hashlib.sha256(runner_source.encode()).hexdigest()=={hashlib.sha256(runner.encode()).hexdigest()!r}\n"
             "(project/'tools/run_fd_validation_pilot.py').write_text(runner_source,encoding='utf-8')\n"
             "print('RUNNER UPDATE: epochs15, FD0.1, seeds0/1/2; fresh initialization',flush=True)\n")
    script = script.replace(marker, patch+marker)
    assert "'--epochs','5'" in script
    script = script.replace("'--epochs','5'", "'--epochs','15','--fd-weights','0.1'")
    compile(script, 'epoch15_bootstrap.py', 'exec')
    (folder/'run.py').write_text(script, encoding='utf-8')
    metadata = json.loads((old/'kernel-metadata.json').read_text())
    metadata.update(id='nha4von/mfvlr-genface-fd01-epoch15-20261027',
                    title='MFVLR GenFace FD01 Epoch15 20261027')
    assert metadata['is_private'] and not metadata['enable_internet']
    (folder/'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    plan = dict(ref=metadata['id'], epochs=15, seeds=[0,1,2], fd_weights=[.1],
                train=1000, validation=400, test_used=False, resume=False,
                dataset_sources=metadata['dataset_sources'], runner_sha256=hashlib.sha256(runner.encode()).hexdigest())
    (root/'plan.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    print(json.dumps(plan, indent=2), flush=True)
    if args.launch:
        assert not (root/'launch_record.json').exists(), 'Already launched; inspect existing notebook first'
        sys.path.insert(0, str(ROOT/'tmp/kaggle_current_deps'))
        from kaggle.api.kaggle_api_extended import KaggleApi
        api=KaggleApi(); api.authenticate()
        response=api.kernels_push(str(folder),timeout='14400',acc='NvidiaTeslaT4')
        result=dict(plan,url=response.url,version=response.version_number)
        result['ref']='/'.join(response.url.rstrip('/').split('/')[-2:])
        (root/'launch_record.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    main()
