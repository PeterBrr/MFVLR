"""Frozen GenFace language-trained checkpoint test and per-image visual exports."""
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'artifacts/language5000_scratch_kaggle_20261028'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    out = ART / 'test_001'
    assert not out.exists(), 'Choose a fresh output directory'
    out.mkdir(parents=True)
    package = ART / 'package_001'
    manifest = json.loads((package / 'package_manifest.json').read_text())
    bundle = package / 'dataset_upload/mfvlr_language_assets.bin'
    assert sha(bundle) == manifest['bundle_sha256']
    frozen = out / 'frozen_source'
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            assert not Path(name).is_absolute() and '..' not in Path(name).parts
            if name.startswith(('src/', 'tools/')):
                z.extract(name, frozen)
    assert all(sha(frozen / p) == h for p, h in manifest['source_sha256'].items())
    sys.path[:0] = [str(ROOT / '.pilot_deps'), str(frozen / 'src'), str(frozen / 'tools')]
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from mfvlr_research.pilot_data import PilotFaces
    from mfvlr_research.pilot_variant import training_model
    from run_genface_ablation import MetricView
    from test_split_mask_encoder_pilot import evaluate
    from run_loss_overfit import export_preview, write_json

    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    assert torch.cuda.is_available()
    checkpoint_path = ART / 'downloaded_stage2/language5000_results/final.pt'
    report = json.loads((checkpoint_path.parent / 'report.json').read_text())
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    assert ckpt['schema'] == 'mfvlr-language5000-stage1-final-v1'
    assert ckpt['step'] == 18750 and report['status'] == 'COMPLETED'
    assert ckpt['config'] == report['config']
    assert ckpt['provenance']['source'] == report['source_sha256']
    assert ckpt['provenance']['data'] == report['data_sha256']
    assert all(sha(frozen / p) == h for p, h in ckpt['provenance']['source'].items())
    model = training_model(ckpt['config'])
    model.load_state_dict(ckpt['model'], strict=True)
    del ckpt
    gc.collect()
    model.cuda().eval()
    view = MetricView(model)
    weights = torch.tensor([1., 4.], device='cuda')

    def load(split):
        ds = PilotFaces(ROOT / f'artifacts/diffae_pilot/manifests/{split}.jsonl')
        if split == 'val':
            by_id = {row['sample_id']: row for row in ds.rows}
            ds.rows = [by_id[sid] for sid in report['val_ids']]
        else:
            groups = [[r for r in ds.rows if r['label'] == c] for c in (0, 1)]
            assert [len(g) for g in groups] == [200, 200]
            ds.rows = [r for pair in zip(*groups) for r in pair]
        samples = [ds[i] for i in range(len(ds))]
        return samples, ds.digest

    validation, _ = load('val')
    local_val, _ = evaluate(view, validation, 'cuda', weights, 2)
    expected = report['evaluations'][-1]['val']
    parity = dict(auc_delta=local_val['auc'] - expected['auc'],
                  iou_delta=local_val['fake_foreground_iou_macro'] - expected['fake_foreground_iou_macro'],
                  mse_delta=local_val['losses']['ar'] - expected['losses']['ar'])
    print(json.dumps(dict(event='validation_parity', **parity)), flush=True)
    assert abs(parity['auc_delta']) < .001 and abs(parity['iou_delta']) < .001
    assert abs(parity['mse_delta']) < .00001
    del validation
    samples, test_digest = load('test')
    audit = json.loads((ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/package_001/data_audit.json').read_text())
    known_hashes = set(audit['splits']['train']['postresize_rgb_hashes']) | set(audit['splits']['val']['postresize_rgb_hashes'])
    hashes = [hashlib.sha256((s['image'] * 255).round().byte().numpy().tobytes()).hexdigest() for s in samples]
    ids = [s['sample_id'] for s in samples]
    assert len(set(hashes)) == len(set(ids)) == 400
    assert not set(hashes) & known_hashes
    assert not set(ids) & (set(report['train_ids']) | set(report['val_ids']))

    folders = ('input', 'AD', 'GT', 'mask', 'overlay', 'probability')
    for folder in folders:
        (out / folder).mkdir()
    index, selected = [], []

    class ExportView:
        def eval(self):
            view.eval()
            return self

        def train(self):
            view.train()
            return self

        def __call__(self, images):
            output = view(images)
            ad = output['reconstruction'].detach().cpu().permute(0, 2, 3, 1).numpy()
            prob = output['mask_logits'].softmax(1)[:, 1].detach().cpu().numpy()
            pred = output['mask_logits'].argmax(1).detach().cpu().numpy().astype(bool)
            scores = output['detection_logits'].softmax(1)[:, 1].detach().cpu().tolist()
            for j in range(len(images)):
                i = len(index)
                s = samples[i]
                rgb = s['image'].permute(1, 2, 0).numpy()
                gt = s['mask'].numpy().astype(bool)
                stem = f'{i:03d}_' + ('real' if s['label'] == 0 else 'fake')
                overlay = rgb.copy()
                overlay[pred[j]] = .55 * overlay[pred[j]] + .45 * np.array([1., 0., 0.])
                arrays = dict(input=rgb, AD=ad[j], GT=gt, mask=pred[j], overlay=overlay)
                for name, array in arrays.items():
                    Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8)).save(out / name / (stem + '.png'))
                np.save(out / 'probability' / (stem + '.npy'), prob[j], allow_pickle=False)
                union = int((pred[j] | gt).sum())
                index.append(dict(file=stem, sample_id=s['sample_id'], label=int(s['label']),
                    fake_score=scores[j], predicted_area=float(pred[j].mean()),
                    iou=float((pred[j] & gt).sum() / union) if union else 1.))
                if i < 16:
                    selected.append(dict(sample_id=s['sample_id'], label=int(s['label']), image=rgb,
                        reconstruction=ad[j], target=gt, prediction=pred[j], probability=prob[j]))
            print(f'Test/export {len(index)}/400', flush=True)
            return output

    result, _ = evaluate(ExportView(), samples, 'cuda', weights, 2)
    export_preview(out, selected)
    # A shorter contact sheet for opening directly in the IDE.
    sheet = Image.new('RGB', (224 * 5, 250 * 8), 'white')
    draw = ImageDraw.Draw(sheet)
    for row, entry in enumerate(index[:8]):
        for col, name in enumerate(('input', 'AD', 'GT', 'mask', 'overlay')):
            draw.text((col * 224 + 4, row * 250 + 4), f'{entry["file"]} {name}', fill='black')
            with Image.open(out / name / (entry['file'] + '.png')) as im:
                sheet.paste(im.convert('RGB'), (col * 224, row * 250 + 24))
    sheet.save(out / 'comparison.png')
    counts = result['pixel_confusion']
    result.update(pixel_precision=counts['tp'] / max(1, counts['tp'] + counts['fp']),
        pixel_recall=counts['tp'] / max(1, counts['tp'] + counts['fn']),
        split='test', epoch=15, step=18750, seed=0, checkpoint_sha256=sha(checkpoint_path),
        manifest_sha256=test_digest, source_verified=True, validation_parity=parity,
        exact_test_train_val_content_overlap=False, subject_disjoint_verified=False,
        language_trained=True, inference='visual_only; no class prompts or test labels passed to model',
        threshold_policy='Fixed detection0.5 and mask argmax; no test tuning',
        display_policy='AD PNG clipped to [0,1] for display only; metrics use raw AD',
        test_previously_used_in_project=True,
        environment=dict(torch=torch.__version__, gpu=torch.cuda.get_device_name()))
    write_json(out / 'index.json', index)
    write_json(out / 'test_metrics.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
