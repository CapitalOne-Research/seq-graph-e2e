import importlib.util
import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import patch

MODULE_PATH = Path(__file__).resolve().parents[1] / 'brightkite/launch_eval.py'
SPEC = importlib.util.spec_from_file_location('evaluation_launcher', MODULE_PATH)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


class EvaluationLauncherTest(unittest.TestCase):
    def test_custom_directory_dataset_seed_and_finished_run_filter(self):
        with tempfile.TemporaryDirectory() as temp:
            checkpoints = Path(temp) / 'custom checkpoints'
            results = Path(temp) / 'custom results'
            checkpoints.mkdir()
            results.mkdir()
            pending = checkpoints / 'gcn_seed0_best.pth'
            for path in [pending, checkpoints / 'gcn_seed1_best.pth',
                         checkpoints / 'gat_seed0_best.pth', checkpoints / 'tokenxat_seed0_best.pth']:
                path.write_bytes(b'fixture')
            (results / 'gat_seed0.json').write_text(json.dumps([{'best_test_mrr': 0.5}]))
            with patch('sys.argv', ['launch_eval', '--dataset', 'gowalla', '--seed', '0',
                                    '--checkpoint-dir', str(checkpoints)]), \
                    patch.object(launcher, 'launch_queue_on_available_gpus') as queue:
                launcher.main()
            commands = queue.call_args.args[0]
            self.assertEqual(len(commands), 1)
            self.assertEqual(shlex.split(commands[0]),
                             ['python', 'joint.py', '--dataset', 'gowalla', '--load_checkpoint_and_eval', str(pending)])


if __name__ == '__main__':
    unittest.main()
