import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'brightkite/prepare_gowalla_data.py'
SPEC = importlib.util.spec_from_file_location('gowalla_download', MODULE_PATH)
download = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(download)


class FakeCurl:
    def __init__(self, data):
        self.stdout = io.BytesIO(data)
        self.stopped = False

    def kill(self):
        self.stopped = True

    def wait(self):
        return 0


class GowallaDownloadTest(unittest.TestCase):
    def test_configured_url_and_aligned_negative_sampling(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'relational_val.csv').write_text('uid,event_set,other_uid\n1,r,2\n3,r,4\n')
            data = b'uid,event_set,other_uid\n1,r,10\n1,r,11\n1,r,12\n3,r,20\n3,r,21\n3,r,22\n'
            proc = FakeCurl(data)
            with patch.object(download.subprocess, 'Popen', return_value=proc) as popen:
                download.build_split('val', temp, 2, 3, 0, 'https://data.example/tasks')
            self.assertEqual(popen.call_args.args[0][-1],
                             'https://data.example/tasks/relational_val_negative_sample.csv')
            self.assertEqual((root / 'relational_val_negative_sample_2.csv').read_text(),
                             'uid,event_set,other_uid\n1,r,10\n1,r,11\n3,r,20\n3,r,21\n')
            self.assertTrue(proc.stopped)

    def test_requires_base_url_before_creating_output_directory(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'data'
            with patch.dict(download.os.environ, {}, clear=True), patch('sys.argv', ['prepare', '--out_dir', str(output)]):
                with self.assertRaises(SystemExit) as error:
                    download.main()
            self.assertEqual(error.exception.code, 2)
            self.assertFalse(output.exists())

    def test_accepts_url_from_environment(self):
        with tempfile.TemporaryDirectory() as temp:
            with patch.dict(download.os.environ, {'GOWALLA_DATA_URL': 'https://data.example/tasks/'}), \
                    patch('sys.argv', ['prepare', '--out_dir', temp]), \
                    patch.object(download, 'download') as get, patch.object(download, 'build_split') as build:
                download.main()
            self.assertEqual(get.call_count, 4)
            self.assertEqual(get.call_args.args[-1], 'https://data.example/tasks')
            self.assertEqual(build.call_count, 2)


if __name__ == '__main__':
    unittest.main()
