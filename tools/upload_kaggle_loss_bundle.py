"""Upload the prepared 16-image experiment as a new private Kaggle dataset."""
import argparse
import ast
import base64
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tmp/kaggle_upload_deps'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--owner', required=True)
    args = parser.parse_args()
    assert os.environ.get('KAGGLE_API_TOKEN'), 'Kaggle token unavailable in this process'
    manifest = json.loads((args.package / 'package_manifest.json').read_text())
    script = (args.package / 'notebook.py').read_text()
    tree = ast.parse(script)
    assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                      and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'payload')
    payload = base64.b64decode(ast.literal_eval(assignment.value.args[0]))
    assert hashlib.sha256(payload).hexdigest() == manifest['payload_sha256']
    folder = args.package / 'dataset_upload'
    folder.mkdir(exist_ok=False)
    (folder / 'mfvlr_loss_bundle.bin').write_bytes(payload)
    dataset_slug = 'mfvlr-genface-loss16-20261026'
    metadata = dict(id=f'{args.owner}/{dataset_slug}', title='MFVLR GenFace loss16 private 20261026',
                    licenses=[{'name': 'other'}],
                    description='Private reproducibility bundle: 16 fixed GenFace train images/author GT, required source, no credentials. Training-set overfit diagnostics only.')
    (folder / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
    # CLI initialization still expects these legacy fields, while its bundled
    # SDK prefers KAGGLE_API_TOKEN and sends actual Bearer authentication.
    os.environ['KAGGLE_USERNAME'] = args.owner
    os.environ['KAGGLE_KEY'] = 'unused-bearer-auth'
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    response = api.dataset_create_new(str(folder), public=False, quiet=True, convert_to_csv=False)
    print(json.dumps(dict(dataset=metadata['id'], url=getattr(response, 'url', None),
                          status=getattr(response, 'status', None), private_requested=True)), flush=True)
    short = script.replace(ast.get_source_segment(script, assignment),
        "candidates = list(Path('/kaggle/input').rglob('mfvlr_loss_bundle.bin'))\n"
        "assert len(candidates) == 1, 'Missing or ambiguous private bundle'\n"
        "payload = candidates[0].read_bytes()")
    assert len(short.encode()) < 1000000
    (args.package / 'notebook_dataset.py').write_text(short, encoding='utf-8')
    (args.package / 'dataset_reference.json').write_text(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
