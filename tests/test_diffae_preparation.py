"""Standalone stdlib tests; no model or optimizer execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).parents[1] / 'tools/prepare_diffae_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PreparationTests(unittest.TestCase):
    def test_unique_pairs_staging_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, out = Path(tmp) / 'data', Path(tmp) / 'output'
            root.mkdir()
            for split in ('train', 'val', 'test'):
                (root / f'{split}.txt').write_text('image level1 level2 level3 level4\n')
            (root / 'train.txt').write_text('image level1 level2 level3 level4\n7.jpg 0 0 0 0\n8.jpg 0 0 0 0\nDiffae_Bangs_0.png 1 2 3 2\n')
            files = ['real/real/CelebA_HQ/images/7.jpg', 'real/real/CelebA_HQ/masks/7.jpg',
                     'real/real/CelebA_HQ/images/.bdpan_staging/7.jpg',
                     'real/real/CelebA_HQ/images/8.jpg', 'real/real/FFHQ/images/8.jpg',
                     'fake/AM/Diffusion/image/Diffae_Bangs_0.png',
                     'fake/AM/Diffusion/mask/Diffae_Bangs_0.png']
            for file in files:
                p = root / file
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b'fixture')
            report = module.prepare(root, out)
            self.assertEqual(report['counts']['train']['ambiguous_image'], 1)
            self.assertEqual(report['counts']['train']['selected_class_0'], 1)
            self.assertEqual(report['counts']['train']['selected_class_1'], 1)
            rows = [json.loads(x) for x in (out / 'train.candidates.jsonl').read_text().splitlines()]
            self.assertTrue(all(r['readiness'] == 'CANDIDATE_ONLY' for r in rows))
            with self.assertRaises(ValueError):
                module.prepare(root, root / 'output')
            (root / 'val.txt').write_text('image level1 level2 level3 level4\n7.jpg 0 0 0 0\n')
            with self.assertRaisesRegex(ValueError, 'Duplicate annotation'):
                module.prepare(root, out)


if __name__ == '__main__':
    unittest.main()
