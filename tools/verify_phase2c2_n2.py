"""Run unchanged targeted tests into NEW N2 logs; retain T17's actual failure."""
from pathlib import Path
import json
import os
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'artifacts/phase2c2_n2'
    attempt=1
    while (out/f'verification_{attempt:02}').exists():attempt+=1
    target=out/f'verification_{attempt:02}';target.mkdir(parents=True)
    command=[sys.executable,'-m','pytest','tests/foundation/test_vim.py',
        'tests/language/test_encoder_stage.py::test_flt_t08_fp64_attention_reference',
        'tests/language/test_flt.py::test_flt_t17_fixed_batch_independence',
        'tests/language/test_flt.py::test_flt_t17_cross_batch_original_tolerance',
        '-vv',f'--junitxml={target/"pytest.xml"}']
    env=os.environ.copy();env['PYTHONPATH']=os.pathsep.join([str(ROOT/'.phase2c1_deps'),str(ROOT/'src')])
    completed=subprocess.run(command,cwd=ROOT,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                             text=True,encoding='utf-8',errors='replace')
    (target/'pytest.log').write_text('COMMAND: '+json.dumps(command)+'\n'+completed.stdout+
        f'\nEXIT_CODE: {completed.returncode}\n',encoding='utf-8')
    (target/'execution.json').write_text(json.dumps({'command':command,'exit_code':completed.returncode,
        'tests_modified':False,'N2_applied':False,'complete_new_suite_run':False},indent=2)+'\n')
    print(completed.stdout)
    raise SystemExit(completed.returncode)


if __name__=='__main__':main()
