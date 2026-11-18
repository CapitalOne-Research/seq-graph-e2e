import csv
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/create_negative_sample_100.py'
SPEC = importlib.util.spec_from_file_location('negative_conversion', SCRIPT)
conversion = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(conversion)


class NegativeConversionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'relational_val_negative_sample.csv'
        self.output = self.root / 'relational_val_negative_sample_100.csv'

    def write_rows(self, rows):
        with self.source.open('w', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['uid', 'event', 'other_uid'])
            writer.writerows(rows)

    def read_rows(self):
        with self.output.open(newline='') as source:
            return list(csv.reader(source))

    def test_matches_notebook_block_slices_even_for_repeated_uids(self):
        # Equal UIDs in successive positive blocks must not be grouped together.
        rows = [['7', 'quoted, event\nwith newline', str(i)] for i in range(3000)]
        self.write_rows(rows)
        before = self.source.read_bytes()
        positives = self.root / 'relational_val.csv'
        positives.write_text('uid,other_uid\n7,1\n7,2\n7,3\n')
        self.assertEqual(conversion.convert_file(self.source, self.output, positives=positives), (3000, 300))
        expected = [row for start in range(0, len(rows), 1000) for row in rows[start:start + 100]]
        self.assertEqual(self.read_rows(), [['uid', 'event', 'other_uid']] + expected)
        self.assertEqual(self.source.read_bytes(), before)

    def test_invalid_alignment_leaves_existing_output_unchanged(self):
        self.write_rows([['1', 'r', '10'], ['2', 'r', '11']])
        self.output.write_text('original output\n')
        with self.assertRaisesRegex(ValueError, 'UID changes'):
            conversion.convert_file(self.source, self.output, block_size=2, keep=1, overwrite=True)
        self.assertEqual(self.output.read_text(), 'original output\n')
        self.assertFalse(list(self.root.glob('*.tmp')))

    def test_rejects_partial_block_and_extra_positive(self):
        self.write_rows([['1', 'r', '10']])
        with self.assertRaisesRegex(ValueError, 'complete'):
            conversion.convert_file(self.source, self.output, block_size=2, keep=1)
        self.assertFalse(self.output.exists())
        self.write_rows([['1', 'r', '10'], ['1', 'r', '11']])
        positive = self.root / 'relational_val.csv'
        positive.write_text('uid,other_uid\n1,2\n3,4\n')
        with self.assertRaisesRegex(ValueError, 'fewer negative blocks'):
            conversion.convert_file(self.source, self.output, block_size=2, keep=1, positives=positive)
        self.assertFalse(self.output.exists())

    def test_cli_converts_both_splits_and_refuses_overwrite(self):
        self.write_rows([['1', 'r', str(i)] for i in range(1000)])
        (self.root / 'relational_test_negative_sample.csv').write_bytes(self.source.read_bytes())
        command = [sys.executable, str(SCRIPT), '--data-dir', str(self.root)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertTrue((self.root / 'relational_test_negative_sample_100.csv').exists())
        self.assertEqual(len(self.read_rows()), 101)
        before = self.output.read_bytes()
        second = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(second.returncode, 0)
        self.assertIn('--overwrite', second.stderr)
        self.assertEqual(self.output.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
