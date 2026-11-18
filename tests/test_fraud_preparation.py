"""Read-only split validation and launcher planning; no ML dependencies required."""
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'amazon' / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


verify = load('verify_fraud_split', 'verify_canonical_split.py')
pool = load('fraud_pool', 'run_fraud_rdl_pool.py')


class FakeMask:
    def __init__(self, values):
        self.values = values

    def bool(self):
        return self

    cpu = ravel = bool

    def tolist(self):
        return self.values


class SplitVerificationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        (self.base / 'all_nodes.csv').write_text('uid,event\n2,A\n1,B\n3,C\n')
        (self.base / 'all_edges.csv').write_text('uid,other_uid\n1,3\n')
        (self.base / 'splits').mkdir()
        self.csv = self.base / 'splits/movies_canonical_user_split_seed0.csv'
        self.pt = self.base / 'splits/movies_splits_seed0.pt'

    def run_verify(self, *extra):
        output = io.StringIO()
        with patch.object(sys, 'argv', ['verify', '--processed', str(self.base),
                                       '--dataset', 'movies', '--seeds', '0', *extra]), \
                contextlib.redirect_stdout(output):
            result = verify.main()
        return result, output.getvalue()

    def snapshot(self):
        return {str(p.relative_to(self.base)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in self.base.rglob('*') if p.is_file()}

    def test_canonical_only_no_torch_and_no_mutation(self):
        self.csv.write_text('uid,split\n1,train\n2,val\n3,test\n')
        before = self.snapshot()
        with patch.dict(sys.modules, {'torch': None}):
            self.assertEqual(self.run_verify()[0], 0)
        self.assertEqual(self.snapshot(), before)

    def test_missing_inputs_fail_even_allow_missing_when_zero_verified(self):
        self.assertEqual(self.run_verify()[0], 1)
        self.assertEqual(self.run_verify('--allow-missing')[0], 1)

    def test_requested_missing_seed_requires_explicit_opt_in(self):
        self.csv.write_text('uid,split\n1,train\n2,val\n3,test\n')
        self.assertEqual(self.run_verify('--seeds', '0', '1')[0], 1)
        self.assertEqual(self.run_verify('--seeds', '0', '1', '--allow-missing')[0], 0)

    def test_duplicate_invalid_missing_and_unknown_users_fail(self):
        for contents in ['1,train\n1,val\n2,val\n3,test\n',
                         '1,bogus\n2,val\n3,test\n', '1,train\n2,val\n',
                         '1,train\n2,val\n3,test\n4,train\n']:
            with self.subTest(contents=contents):
                self.csv.write_text('uid,split\n' + contents)
                self.assertEqual(self.run_verify()[0], 1)

    def test_pt_identity_mismatch_and_overlap_are_read_only(self):
        self.pt.write_bytes(b'fake torch archive')
        self.csv.write_text('uid,split\n1,train\n2,val\n3,test\n')
        masks = {'train_mask': FakeMask([True, False, False]),
                 'val_mask': FakeMask([False, True, False]),
                 'test_mask': FakeMask([False, False, True])}
        torch = types.SimpleNamespace(load=lambda *a, **kw: masks)
        before = self.snapshot()
        with patch.dict(sys.modules, {'torch': torch}):
            self.assertEqual(self.run_verify()[0], 0)
            masks['train_mask'] = FakeMask([False, True, False])
            masks['val_mask'] = FakeMask([True, False, False])
            self.assertEqual(self.run_verify()[0], 1)
            masks['val_mask'] = FakeMask([False, True, False])
            self.assertEqual(self.run_verify()[0], 1)
        self.assertEqual(self.snapshot(), before)

    def test_pt_only_does_not_materialize_csv(self):
        self.pt.write_bytes(b'fake torch archive')
        masks = {f'{split}_mask': FakeMask([i == j for i in range(3)])
                 for j, split in enumerate(('train', 'val', 'test'))}
        before = self.snapshot()
        with patch.dict(sys.modules, {'torch': types.SimpleNamespace(load=lambda *a, **kw: masks)}):
            self.assertEqual(self.run_verify()[0], 0)
        self.assertEqual(self.snapshot(), before)


class PoolDryRunTest(unittest.TestCase):
    def test_text_dry_run_prints_every_command_without_subprocess_or_files(self):
        with tempfile.TemporaryDirectory() as temp:
            output = io.StringIO()
            with patch.object(pool, 'REPO', temp), \
                    patch.object(sys, 'argv', ['pool', '--dry_run', '--use_review_text',
                                              '--relgnn_datasets', 'movies,electronics,clothing',
                                              '--relgt_datasets', 'movies,electronics,clothing']), \
                    patch.object(pool.subprocess, 'run', side_effect=AssertionError('subprocess run')), \
                    patch.object(pool.subprocess, 'Popen', side_effect=AssertionError('subprocess Popen')), \
                    contextlib.redirect_stdout(output):
                pool.main()
            text = output.getvalue()
            self.assertEqual(text.count('[prebuild]'), 3)
            self.assertEqual(text.count('  cd '), 18)
            self.assertIn('[missing dependency]', text)
            self.assertIn('relgt-full_text/clothing/s2', text)
            self.assertEqual(list(Path(temp).rglob('*')), [])

    def test_single_arena_text_prebuild_uses_single_directory(self):
        with patch.object(sys, 'argv', ['pool', '--dry_run', '--single_arena', '--use_review_text',
                                       '--models', 'relgt', '--relgt_datasets', 'movies', '--seeds', '0']), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            pool.main()
        self.assertIn('/fraud_detection/relational', output.getvalue())
        self.assertNotIn('/fraud_detection_seed*', output.getvalue())


if __name__ == '__main__':
    unittest.main()
