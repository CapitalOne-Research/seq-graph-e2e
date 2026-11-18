import contextlib
import io
from pathlib import Path
import os
import runpy
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'brightkite/run_ablation.py'


class AblationCliTest(unittest.TestCase):
    def test_help_and_dry_run_never_start_processes_or_create_outputs(self):
        for option in ['--help', '--dry-run']:
            with self.subTest(option=option), tempfile.TemporaryDirectory() as temp:
                previous = os.getcwd()
                os.chdir(temp)
                try:
                    with patch('sys.argv', [str(SCRIPT), option]), \
                            patch('subprocess.Popen', side_effect=AssertionError('Unexpected child process')), \
                            patch('subprocess.check_output', side_effect=AssertionError('Unexpected GPU query')), \
                            contextlib.redirect_stdout(io.StringIO()) as output:
                        with self.assertRaises(SystemExit) as result:
                            runpy.run_path(str(SCRIPT), run_name='__main__')
                    self.assertEqual(result.exception.code, 0)
                    self.assertTrue(output.getvalue())
                    self.assertEqual(list(Path(temp).iterdir()), [])
                finally:
                    os.chdir(previous)


if __name__ == '__main__':
    unittest.main()
