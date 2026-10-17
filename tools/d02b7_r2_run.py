"""R2 uses the unchanged D02B7 coordinator; only its output directory differs."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools import d02b7_run as frozen
frozen.OUT=ROOT/'artifacts/phase2d2_d02b7_r2'
if __name__=='__main__':frozen.main()
