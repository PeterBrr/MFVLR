"""Pack 50k Diffae fakes and 50k reals, with masks and split manifests, into one zip for an HPC server.

Splits follow the official GenFace lists. Diffae is sampled from each split in proportion to its
size. Reals are every CelebA-HQ image the official lists put in that split, then FFHQ images from
the split's own index range (train 0-55999, val 56000-62999), so no FFHQ face is in two splits.
Test reals stay CelebA only: the local FFHQ copy stops at 63001. The reals test cannot take go to
train. Every image of the 10k/2k/2k set is kept, so earlier runs stay a subset.
"""
import argparse
import csv
import io
import json
from pathlib import Path
import random
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = Path('D:/BaiduNetdiskDownload')
DIFFAE = DATA / 'fake/AM/Diffusion'
CELEBA = DATA / 'real/real/CelebA_HQ'
FFHQ = DATA / 'real/real/FFHQ'
FFHQ_RANGE = dict(train=range(0, 56000), val=range(56000, 63000), test=range(0))
SPLITS = ('train', 'val', 'test')
KEEP = ROOT / 'artifacts/genface10k_20261103'


def official(split):
    rows = [line.split() for line in (DATA / f'{split}.txt').read_text(encoding='utf-8-sig').splitlines()[1:] if line.split()]
    return rows


def quotas(sizes, total):
    """Largest-remainder split of `total` in proportion to `sizes`."""
    exact = {k: total * v / sum(sizes.values()) for k, v in sizes.items()}
    out = {k: int(v) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[:total - sum(out.values())]:
        out[k] += 1
    return out


def kept_ids():
    ids = {}
    for split in SPLITS:
        for line in (KEEP / f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
            ids[json.loads(line)['sample_id'].casefold()] = split
    return ids


def row(split, source, label, image, mask, annotation):
    return dict(split=split, source=source, label=label, image=image, mask=mask, annotation=annotation)


def select(n_fake, n_real, seed):
    rng = random.Random(seed)
    keep = kept_ids()
    disk = {p.name.casefold(): p for p in (DIFFAE / 'image').glob('Diffae_*.png')}
    masks = {p.name.casefold(): p for p in (DIFFAE / 'mask').glob('Diffae_*.png')}
    fakes, celeba = {}, {}
    for split in SPLITS:
        fakes[split], celeba[split] = [], []
        for name, *annotation in official(split):
            if annotation == ['0'] * 4:
                if (CELEBA / 'images' / name).exists() and (CELEBA / 'masks' / name).exists():
                    celeba[split].append(row(split, 'CelebA_HQ', 0, CELEBA / 'images' / name, CELEBA / 'masks' / name, annotation))
            elif name.startswith('Diffae_') and name.casefold() in disk and name.casefold() in masks:
                fakes[split].append(row(split, 'Diffae', 1, disk[name.casefold()], masks[name.casefold()], annotation))
    fake_quota = quotas({s: len(fakes[s]) for s in SPLITS}, n_fake)
    rejected = []

    def take(rows, n):
        """Forced rows first, then the rest in seeded random order, skipping rows that fail `check`."""
        rows = sorted(rows, key=lambda r: f'{r["source"]}/{r["image"].name}'.casefold() not in keep)
        forced = sum(f'{r["source"]}/{r["image"].name}'.casefold() in keep for r in rows)
        rest = rows[forced:]
        rng.shuffle(rest)
        out = []
        for r in rows[:forced] + rest:
            if len(out) == n:
                break
            reason = check(r)
            if reason:
                rejected.append((str(r['image']), reason))
            else:
                out.append(r)
            if len(out) % 10000 == 0 and not reason:
                print(r['split'], r['source'], len(out), n, flush=True)
        assert len(out) == n, f'only {len(out)} usable {rows[0]["source"]} rows in {rows[0]["split"]}'
        return out

    chosen = {}
    good_celeba = {}
    for split in SPLITS:
        good_celeba[split] = []
        for r in celeba[split]:
            reason = check(r)
            if reason:
                rejected.append((str(r['image']), reason))
            else:
                good_celeba[split].append(r)
        chosen[split] = take(fakes[split], fake_quota[split]) + good_celeba[split]
    # FFHQ fills each split up to its fake count; what test cannot take goes to train.
    need = {s: max(0, fake_quota[s] - len(good_celeba[s])) for s in ('train', 'val')}
    need['train'] += n_real - sum(need.values()) - sum(len(c) for c in good_celeba.values())
    need['test'] = 0
    for split in SPLITS:
        if need[split]:
            pool = [row(split, 'FFHQ', 0, FFHQ / f'images/{k:05d}.jpg', FFHQ / f'masks/{k:05d}.png', ['0'] * 4)
                    for k in FFHQ_RANGE[split] if (FFHQ / f'images/{k:05d}.jpg').exists() and (FFHQ / f'masks/{k:05d}.png').exists()]
            chosen[split] += sorted(take(pool, need[split]), key=lambda r: r['image'].name)
    picked = {f'{r["source"]}/{r["image"].name}'.casefold() for s in SPLITS for r in chosen[s]}
    missing = [i for i in keep if i not in picked]
    assert not missing, f'10k/2k/2k images left out: {missing[:5]}'
    return chosen, fake_quota, rejected


def check(r):
    try:
        with Image.open(r['image']) as image:
            image.verify()
        mask = np.asarray(Image.open(r['mask']).convert('L'))
    except Exception as error:
        return f'unreadable: {type(error).__name__}'
    positive = (mask > 127).mean()
    if r['label'] == 1 and positive == 0:
        return 'fake mask empty'
    if r['label'] == 0 and positive > 0:
        return 'real mask not empty'
    return None


def arcname(r, kind):
    folder = dict(Diffae='diffae', CelebA_HQ='celeba_hq', FFHQ='ffhq')[r['source']]
    return f'genface50k/{folder}/{kind}/{r[kind].name}'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('D:/GenFace_HPC_50k_20261103'))
    parser.add_argument('--fakes', type=int, default=50000)
    parser.add_argument('--reals', type=int, default=50000)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    chosen, fake_quota, rejected = select(args.fakes, args.reals, args.seed)
    print('rejected', len(rejected), rejected[:20], flush=True)
    counts = {s: {src: sum(r['source'] == src for r in chosen[s]) for src in ('Diffae', 'CelebA_HQ', 'FFHQ')} for s in SPLITS}
    total = {src: sum(c[src] for c in counts.values()) for src in ('Diffae', 'CelebA_HQ', 'FFHQ')}
    assert total['Diffae'] == args.fakes and total['CelebA_HQ'] + total['FFHQ'] == args.reals, total
    names = [arcname(r, 'image') for s in SPLITS for r in chosen[s]]
    assert len(names) == len(set(names)), 'duplicate file in zip'
    args.output.mkdir(parents=True, exist_ok=True)
    target = args.output / 'genface50k.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_STORED, allowZip64=True) as archive:
        for split in SPLITS:
            text = io.StringIO()
            writer = csv.writer(text, lineterminator='\n')
            writer.writerow(['image', 'mask', 'label', 'source', 'annotation'])
            for r in chosen[split]:
                writer.writerow([arcname(r, 'image')[len('genface50k/'):], arcname(r, 'mask')[len('genface50k/'):],
                                 r['label'], r['source'], ' '.join(r['annotation'])])
            archive.writestr(f'genface50k/{split}.csv', text.getvalue())
        archive.writestr('genface50k/README.md', readme(counts, args.seed))
        done = 0
        for split in SPLITS:
            for r in chosen[split]:
                archive.write(r['image'], arcname(r, 'image'))
                archive.write(r['mask'], arcname(r, 'mask'))
                done += 1
                if done % 10000 == 0:
                    print('zipped', done, flush=True)
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        assert bad is None, f'CRC error in {bad}'
        members = len(archive.namelist())
    summary = dict(zip=str(target), bytes=target.stat().st_size, members=members, counts=counts, total=total, seed=args.seed,
                   rejected=rejected)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))


def readme(counts, seed):
    lines = ['# GenFace 50k Diffae + 50k real', '',
             'Splits follow the official GenFace train/val/test lists. `label` 1 = Diffae fake, 0 = real.',
             'Masks: white = edited region; real masks are all black. `annotation` is the official 4-number label.', '',
             '| split | Diffae | CelebA_HQ | FFHQ |', '|---|---:|---:|---:|']
    lines += [f'| {s} | {c["Diffae"]} | {c["CelebA_HQ"]} | {c["FFHQ"]} |' for s, c in counts.items()]
    lines += ['', 'Notes:',
              '- `Diffae_<attr>_k` is an edit of FFHQ image k. FFHQ reals come only from their split\'s index range',
              '  (train 0-55999, val 56000-62999), so no FFHQ face appears in two splits.',
              '- Test reals are CelebA only (no leak-free FFHQ test images were available); the shortfall went to train.',
              '- Models trained on CelebA-only reals learn a CelebA-vs-FFHQ shortcut; keep the FFHQ reals in training.',
              '- Contains every image of the earlier 10k/2k/2k set. Diffae and FFHQ were sampled with seed '
              f'{seed}.', '- GenFace licence terms apply; do not redistribute.', '']
    return '\n'.join(lines)


if __name__ == '__main__':
    main()
