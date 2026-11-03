"""Grow the frozen 5k/400/400 GenFace split to 10k/2k/2k without crossing official splits.

Every Diffae image is an edit of FFHQ image k (`Diffae_<attr>_k`), and the official
splits cut FFHQ by index: train 0-55999, val 56000-62999, test 63000-69999. Until now
every real was CelebA-HQ, so real/fake also meant CelebA/FFHQ. Train and val therefore
take half their reals from FFHQ inside their own index range; test stays CelebA because
the local FFHQ copy stops at 63001 and every other FFHQ image is the source of a train
or val fake. Old rows stay, in their old order, so the 5k/400 results remain a nested
subset. Test images are never packaged.
"""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'tools')]
import numpy as np
from PIL import Image
from mfvlr_research.pilot_data import PilotFaces, read_manifest
from prepare_diffae_pilot import index

DATA = Path('D:/BaiduNetdiskDownload')
OLD = dict(train=ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/train5000.jsonl',
           val=ROOT / 'artifacts/diffae_pilot/manifests/val.jsonl', test=ROOT / 'artifacts/diffae_pilot/manifests/test.jsonl')
OLD_NPZ = ROOT / 'artifacts/genface5000_epoch30_kaggle_20261027/package_001'
FFHQ_RANGE = dict(train=range(0, 56000), val=range(56000, 63000))
# Rows to add: (CelebA real, FFHQ real, Diffae).
ADD = dict(train=(0, 2500, 2500), val=(300, 500, 800), test=(800, 0, 800))


def digest(data):
    return hashlib.sha256(data).hexdigest()


def resized_hash(image):
    return digest(np.asarray(image.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)).tobytes())


def official(split):
    rows = [line.split() for line in (DATA / f'{split}.txt').read_text(encoding='utf-8-sig').splitlines()[1:] if line.split()]
    assert all(len(r) == 5 for r in rows)
    return rows


def candidates(split, fakes, fake_masks):
    rows = official(split)
    celeba = DATA / 'real/real/CelebA_HQ'
    pools = dict(celeba=[], ffhq=[], diffae=[])
    for name, *annotation in rows:
        if annotation == ['0'] * 4:
            # Official reals are CelebA-HQ: 30,000 unpadded names 0-29999. FFHQ 10000-29999
            # share the basename but are different pictures, which is what made them look ambiguous.
            pools['celeba'].append(dict(sample_id=f'CelebA_HQ/{name}', image=celeba / 'images' / name,
                                        mask=celeba / 'masks' / name, label=0, annotation=annotation, source='CelebA_HQ'))
        elif name.startswith('Diffae_'):
            images = fakes.get(name.casefold(), [])
            masks = fake_masks.get(Path(name).stem.casefold(), []) if len(images) == 1 else []
            pools['diffae'].append(dict(sample_id=f'Diffae/{name}', image=images[0] if len(images) == 1 else None,
                                        mask=masks[0] if len(masks) == 1 else None, label=1, annotation=annotation,
                                        source='Diffae', problem=None if len(images) == 1 and len(masks) == 1 else 'missing_or_ambiguous'))
    for k in FFHQ_RANGE.get(split, ()):
        image = DATA / f'real/real/FFHQ/images/{k:05d}.jpg'
        if image.exists():
            pools['ffhq'].append(dict(sample_id=f'FFHQ/{k:05d}.jpg', image=image, mask=DATA / f'real/real/FFHQ/masks/{k:05d}.png',
                                      label=0, annotation=['0'] * 4, source='FFHQ'))
    return pools


def checked(candidate, split, seen_full, seen_small):
    """A manifest row in the existing format, or the reason it was rejected."""
    if candidate.get('problem'):
        return None, candidate['problem']
    image, mask = Path(candidate['image']), Path(candidate['mask'])
    if not image.is_file() or not mask.is_file():
        return None, 'missing_file'
    with Image.open(image) as source:
        size = source.size
        rgb = np.asarray(source.convert('RGB'))
        small = resized_hash(source)
    full = digest(rgb.tobytes())
    if full in seen_full or small in seen_small:
        return None, 'duplicate_content'
    with Image.open(mask) as source:
        if source.size != size:
            return None, 'size_mismatch'
        values = np.asarray(source)
    if values.ndim == 3:
        if values.shape[2] != 3 or not np.all(values == values[:, :, :1]):
            return None, 'mask_channels'
        values = values[:, :, 0]
    if not set(np.unique(values).tolist()) <= {0, 1, 255}:
        return None, 'mask_not_binary'
    fraction = float((values > 0).mean())
    if (candidate['label'] == 0) != (fraction == 0):
        return None, 'real_mask_nonzero' if candidate['label'] == 0 else 'fake_mask_empty'
    seen_full.add(full)
    seen_small.add(small)
    return dict(annotation=candidate['annotation'], decoded_rgb_sha256=full, foreground_fraction=fraction,
                generator=None if candidate['label'] == 0 else 'Diffae', image=str(image), image_sha256=digest(image.read_bytes()),
                image_size=list(size), kind='real' if candidate['label'] == 0 else 'local', label=candidate['label'],
                mask=str(mask), mask_sha256=digest(mask.read_bytes()), readiness='CANDIDATE_ONLY',
                sample_id=candidate['sample_id'], source_collection=candidate['source'], split=split, subject_group=None), None


def package(manifest, path, expected_prefix=None):
    """Same tensors as PilotFaces, written through memmaps so a 10k split fits in little RAM."""
    dataset = PilotFaces(manifest)
    count = len(dataset)
    scratch = path.parent / f'.{path.stem}_tmp'
    scratch.mkdir(exist_ok=True)
    images = np.lib.format.open_memmap(scratch / 'images.npy', 'w+', np.uint8, (count, 3, 224, 224))
    masks = np.lib.format.open_memmap(scratch / 'masks.npy', 'w+', np.uint8, (count, 224, 224))
    labels, ids, content = [], [], []
    for i in range(count):
        sample = dataset[i]
        images[i] = (sample['image'] * 255).round().byte().numpy()
        masks[i] = sample['mask'].byte().numpy()
        labels.append(sample['label'])
        ids.append(sample['sample_id'])
        content.append(digest(images[i].tobytes()))
        if (i + 1) % 500 == 0:
            print(f'{path.stem}: packed {i + 1}/{count}', flush=True)
    if expected_prefix is not None:
        with np.load(expected_prefix, allow_pickle=False) as old:
            n = len(old['labels'])
            assert ids[:n] == old['sample_ids'].tolist() and labels[:n] == old['labels'].tolist()
            old_images, old_masks = old['images'], old['masks']
            for start in range(0, n, 500):
                stop = min(n, start + 500)
                assert np.array_equal(images[start:stop], old_images[start:stop]), 'Old rows changed'
                assert np.array_equal(masks[start:stop], old_masks[start:stop]), 'Old rows changed'
    assert len(set(ids)) == len(set(content)) == count
    images.flush()
    masks.flush()
    np.savez_compressed(path, images=images, masks=masks, labels=np.array(labels), sample_ids=np.array(ids))
    del images, masks
    for name in ('images.npy', 'masks.npy'):
        (scratch / name).unlink()
    scratch.rmdir()
    return ids, content


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/genface10k_20261103')
    parser.add_argument('--dataset-id', default='nha4von/mfvlr-genface10k-20261103')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a fresh output directory')
    (args.output / 'package_001/dataset_upload').mkdir(parents=True)
    old = {split: read_manifest(path) for split, path in OLD.items()}
    assert [len(old[s]) for s in ('train', 'val', 'test')] == [5000, 400, 400]
    # Old train/val keep their packaged order; old test keeps its manifest order.
    for split in ('train', 'val'):
        with np.load(OLD_NPZ / f'{split}.npz', allow_pickle=False) as data:
            order = data['sample_ids'].tolist()
        by_id = {r['sample_id']: r for r in old[split]}
        assert sorted(order) == sorted(by_id)
        old[split] = [by_id[i] for i in order]
    used = {r['sample_id'] for rows in old.values() for r in rows}
    seen_full = {r['decoded_rgb_sha256'] for rows in old.values() for r in rows}
    seen_small = set()
    for rows in old.values():
        for r in rows:
            with Image.open(r['image']) as source:
                seen_small.add(resized_hash(source))
    fake_dir = DATA / 'fake/AM/Diffusion'
    fakes, fake_masks = index(fake_dir / 'image'), index(fake_dir / 'mask', True)
    rng = random.Random(0)
    final, rejected, available = {}, collections.Counter(), {}
    for split in ('train', 'val', 'test'):
        pools = candidates(split, fakes, fake_masks)
        added = []
        for pool, quota in zip(('celeba', 'ffhq', 'diffae'), ADD[split]):
            items = sorted((c for c in pools[pool] if c['sample_id'] not in used), key=lambda c: c['sample_id'])
            available[f'{split}/{pool}'] = len(items)
            rng.shuffle(items)
            taken = []
            for candidate in items:
                if len(taken) == quota:
                    break
                row, reason = checked(candidate, split, seen_full, seen_small)
                if reason:
                    rejected[f'{split}/{pool}/{reason}'] += 1
                    continue
                used.add(row['sample_id'])
                taken.append(row)
            assert len(taken) == quota, f'{split}/{pool}: only {len(taken)} of {quota}'
            added.append(taken)
            print(f'{split}/{pool}: added {quota}', flush=True)
        reals = added[0] + added[1]
        rng.shuffle(reals)
        fakes_added = added[2]
        final[split] = old[split] + [r for pair in zip(reals, fakes_added) for r in pair]
    manifests = {}
    for split, rows in final.items():
        path = args.output / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows), encoding='utf-8')
        read_manifest(path)
        manifests[split] = path
    # Same face as real and as its Diffae edit is allowed only inside one split.
    faces = {}
    for split, rows in final.items():
        for r in rows:
            name = r['sample_id'].split('/')[-1]
            if r['source_collection'] in ('FFHQ', 'Diffae'):
                faces.setdefault(int(Path(name).stem.split('_')[-1]), set()).add(split)
    assert all(len(splits) == 1 for splits in faces.values()), 'An FFHQ face crosses splits'
    paired = {}
    for split, rows in final.items():
        edited = {int(Path(r['sample_id']).stem.split('_')[-1]) for r in rows if r['label'] == 1}
        paired[split] = sum(int(Path(r['sample_id']).stem) in edited for r in rows if r['source_collection'] == 'FFHQ')
    package_dir = args.output / 'package_001'
    audit = dict(test_used=False, test_packaged=False, subject_disjoint_verified=False, rejected=dict(rejected),
                 available_after_old=available, ffhq_index_ranges={k: [v.start, v.stop - 1] for k, v in FFHQ_RANGE.items()},
                 ffhq_real_with_own_edit_in_same_split=paired, splits={})
    hashes = {}
    for split in ('train', 'val'):
        ids, content = package(manifests[split], package_dir / f'{split}.npz', OLD_NPZ / f'{split}.npz')
        hashes[split] = digest((package_dir / f'{split}.npz').read_bytes())
        audit['splits'][split] = dict(count=len(ids), npz_sha256=hashes[split], sample_ids=ids, postresize_rgb_hashes=content)
    assert not set(audit['splits']['train']['postresize_rgb_hashes']) & set(audit['splits']['val']['postresize_rgb_hashes'])
    for split, rows in final.items():
        audit['splits'].setdefault(split, {})['composition'] = dict(collections.Counter(r['source_collection'] for r in rows))
        audit['splits'][split]['manifest_sha256'] = digest(manifests[split].read_bytes())
    (package_dir / 'data_audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    bundle = package_dir / 'dataset_upload/mfvlr_fd_validation_bundle.bin'
    # The 5k arm trains on the old file itself, so its inputs and schedule match the 01/11 runs bit for bit.
    old_train = OLD_NPZ / 'train.npz'
    hashes['train5k'] = digest(old_train.read_bytes())
    with zipfile.ZipFile(bundle, 'w', compression=zipfile.ZIP_STORED) as archive:
        for name in ('train.npz', 'val.npz', 'data_audit.json'):
            archive.write(package_dir / name, 'inputs/' + name)
        archive.write(old_train, 'inputs/train5k.npz')
    metadata = dict(id=args.dataset_id, title='MFVLR GenFace train10k validation2k', licenses=[dict(name='other')],
                    description='Private GenFace real (CelebA-HQ + FFHQ) / Diffae: 10,000 train + 2,000 validation. No test images or credentials.')
    (package_dir / 'dataset_upload/dataset-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    result = dict(dataset=args.dataset_id, privacy='private', npz_sha256=hashes, bundle_bytes=bundle.stat().st_size,
                  bundle_sha256=digest(bundle.read_bytes()), rejected=dict(rejected), ffhq_real_with_own_edit_in_same_split=paired,
                  composition={s: audit['splits'][s]['composition'] for s in final})
    (package_dir / 'package_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
