"""Read-only candidate inventory. Does not certify masks, identities or train readiness."""
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path


def index(folder, stem=False):
    result = collections.defaultdict(list)
    for p in folder.rglob('*'):
        if any(part.startswith('.') for part in p.relative_to(folder).parts):
            continue
        if p.is_file() and p.suffix.lower() in {'.png', '.jpg', '.jpeg'} and p.stat().st_size:
            result[(p.stem if stem else p.name).casefold()].append(p)
    return result


def prepare(root, output, seed=0):
    root, output = root.resolve(), output.resolve()
    if output == root or root in output.parents:
        raise ValueError('Output must be outside source dataset')
    real = {}
    for source in ('CelebA_HQ', 'FFHQ'):
        base = root / 'real' / 'real' / source
        real[source] = (index(base / 'images'), index(base / 'masks', True))
    base = root / 'fake' / 'AM' / 'Diffusion'
    fake, masks = index(base / 'image'), index(base / 'mask', True)
    output.mkdir(parents=True, exist_ok=True)
    totals, selected, rejected = {}, {}, []
    seen = {}
    for split, quota in (('train', 2500), ('val', 200), ('test', 200)):
        buckets = {0: [], 1: []}
        counts = collections.Counter()
        manifest = root / (split + '.txt')
        with manifest.open(encoding='utf-8-sig') as stream:
            next(stream)
            for line in stream:
                fields = line.split()
                if not fields:
                    continue
                name = fields[0]
                if len(fields) != 5:
                    raise ValueError(f'Malformed annotation: {line}')
                is_real = fields[1:] == ['0'] * 4
                if not is_real and not name.startswith('Diffae_'):
                    continue
                counts['eligible_annotation'] += 1
                if name.casefold() in seen:
                    raise ValueError(f'Duplicate annotation {name} in {split}/{seen[name.casefold()]}')
                seen[name.casefold()] = split
                matches = []
                if is_real:
                    for source, (images, source_masks) in real.items():
                        for image in images.get(name.casefold(), []):
                            matches.append((source, image, source_masks.get(image.stem.casefold(), [])))
                else:
                    for image in fake.get(name.casefold(), []):
                        matches.append(('Diffae', image, masks.get(image.stem.casefold(), [])))
                reason = None
                if len(matches) != 1:
                    reason = 'missing_image' if not matches else 'ambiguous_image'
                elif len(matches[0][2]) != 1:
                    reason = 'missing_mask' if not matches[0][2] else 'ambiguous_mask'
                if reason:
                    counts[reason] += 1
                    rejected.append(dict(split=split, filename=name, reason=reason))
                    continue
                source, image, mask_paths = matches[0]
                label = int(not is_real)
                buckets[label].append(dict(sample_id=f'{source}/{name}', split=split,
                    image=str(image), mask=str(mask_paths[0]), label=label,
                    generator=None if is_real else 'Diffae', kind='real' if is_real else 'local',
                    annotation=fields[1:], source_collection=source,
                    subject_group=None, readiness='CANDIDATE_ONLY'))
        rng = random.Random(seed)
        rows = []
        for label, records in buckets.items():
            records.sort(key=lambda r: r['sample_id'])
            rng.shuffle(records)
            rows.extend(records[:quota])
            counts[f'available_class_{label}'] = len(records)
            counts[f'selected_class_{label}'] = min(quota, len(records))
        rng.shuffle(rows)
        data = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows)
        (output / f'{split}.candidates.jsonl').write_text(data, encoding='utf-8')
        totals[split] = dict(counts)
        selected[split] = hashlib.sha256(data.encode()).hexdigest()
    (output / 'rejected.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rejected), encoding='utf-8')
    report = dict(seed=seed, status='NOT_TRAIN_READY', counts=totals,
        candidate_sha256=selected, checks_pending=['decode and dimensions', 'mask semantics and provenance',
        'annotation mapping verification', 'source/identity grouping across all generators and splits',
        'content hash leakage audit', 'environment and numerical/resource gate'],
        inventory_correction='Only images folders count as real inputs; masks folders and hidden staging excluded.')
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.root, args.output), indent=2))
