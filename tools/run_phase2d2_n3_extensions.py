"""Run local precision extensions and an isolated UNAPPLIED test dry run."""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT)]


def main():
    from tools import run_phase2d2 as runner
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'artifacts/phase2d2_n3');args=parser.parse_args()
    out=args.output.resolve();runner.OUT=out
    names=('tools/review_phase2d2_n3_attention.py','tools/review_phase2d2_n3_decoder_vjp.py','tools/prepare_phase2d2_n3_patch.py')
    (out/'extension_source_hashes.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names},indent=2)+'\n')
    for name,tool,file in (('actual_LE08','review_phase2d2_n3_attention.py','actual_LE08.json'),
                           ('decoder_vjp','review_phase2d2_n3_decoder_vjp.py','decoder_vjp.json')):
        code=runner.command(name,[sys.executable,str(ROOT/'tools'/tool),'--output',str(out/file)])
        if code:raise RuntimeError('Extension control failed; inspect actual log before proceeding')
    # New test source is reconstructed from the proposed patch in artifacts/, never tests/.
    import os
    os.environ['MFVLR_N3_REVIEW_ROOT']=str(ROOT);os.environ['MFVLR_N3_EVIDENCE_ROOT']=str(out)
    (out/'junit').mkdir(exist_ok=True)
    code=runner.command('proposed_controls_dryrun',[sys.executable,'-m','pytest',
        str(ROOT/'artifacts/phase2d2_n3/proposal/dryrun/n3_proposed_tests.py'),'-q','-p','no:cacheprovider',
        '--junitxml',str(out/'junit/proposed_controls_dryrun.xml')])
    if code:raise RuntimeError('Isolated proposed controls failed; no policy acceptance implied')
    print('Local precision controls and isolated proposed test dry run complete; policy remains UNAPPLIED')


if __name__=='__main__':main()
