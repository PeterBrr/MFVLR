"""Use the official SDK when MCP drops the notebook's dataset attachment."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/kaggle_upload_deps'))
package = ROOT / 'artifacts/kaggle_loss_overfit_20261026/package_001'
folder = package / 'kernel_push'
folder.mkdir(exist_ok=True)
script = (package / 'notebook_dataset.py').read_text()
script = script.replace("assert len(candidates) == 1, 'Missing or ambiguous private bundle'",
                        "print('Bundle candidates:', [str(p) for p in candidates], flush=True)\n"
                        "assert len(candidates) == 1, 'Missing or ambiguous private bundle'")
(folder / 'run.py').write_text(script)
metadata = dict(id='nha4von/mfvlr-genface-loss-overfit-20261026-001', id_no=140976160,
                title='MFVLR GenFace Loss Overfit 20261026 001',
                code_file='run.py', language='python', kernel_type='script', is_private=True,
                enable_gpu=True, enable_tpu=False, enable_internet=False,
                dataset_sources=['nha4von/mfvlr-genface-loss16-20261026'])
(folder / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2))
assert os.environ.get('KAGGLE_API_TOKEN')
os.environ['KAGGLE_USERNAME'] = 'nha4von'
os.environ['KAGGLE_KEY'] = 'unused-bearer-auth'
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
response = api.kernels_push(str(folder), timeout=7200)
print(json.dumps(dict(url=getattr(response, 'url', None),
                      version_number=getattr(response, 'version_number', None)), indent=2))
