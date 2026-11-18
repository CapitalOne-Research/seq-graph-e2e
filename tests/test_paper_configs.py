"""CPU-only checks: no training imports, no data/network requirements."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('reproduce_paper', ROOT / 'scripts/reproduce_paper.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def flag(job, name):
    return job['command'][job['command'].index('--' + name) + 1]


class PaperConfigsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.registry = runner.load_registry()

    def test_main_rows_and_seeds(self):
        link = runner.build_plan(self.registry, 'link')
        fraud = runner.build_plan(self.registry, 'fraud')
        self.assertEqual(len(link), 5 * 9 * 3)
        self.assertEqual(len(fraud), 3 * 14 * 3)
        self.assertEqual({job['seed'] for job in link + fraud}, {0, 1, 2})

    def test_selected_datasets_methods_seeds(self):
        plan = runner.build_plan(self.registry, 'link', ['gowalla', 'movies'], ['static-gat'], [7, 9])
        self.assertEqual(len(plan), 4)
        self.assertEqual([(job['dataset'], job['seed']) for job in plan],
                         [('gowalla', 7), ('gowalla', 9), ('movies', 7), ('movies', 9)])
        self.assertTrue(all(flag(job, 'train_batch_size') == '4096' for job in plan))
        self.assertEqual(Path(plan[0]['cwd']).name, 'brightkite')
        self.assertEqual(Path(plan[2]['cwd']).name, 'amazon')

    def test_link_bridge_standard_backbones_and_batch(self):
        plan = runner.build_plan(self.registry, 'link', ['brightkite', 'gowalla', 'clothing'], ['bridge-tfconv'], [0])
        self.assertEqual([flag(job, 'batch_size') for job in plan], ['128', '32', '32'])
        self.assertTrue(all(flag(job, 'gnn_layer_types') == 'TransformerConv,TransformerConv' for job in plan))
        self.assertTrue(all(flag(job, 'bert_num_layers') == '1' for job in plan))
        self.assertTrue(all('TokenXAT' not in ' '.join(job['command']) for job in plan))
        self.assertTrue(all('--enable_early_stopping' in job['command'] for job in plan))

    def test_fraud_bridge_table13(self):
        plan = runner.build_plan(self.registry, 'fraud', ['movies', 'electronics', 'clothing'], ['bridge-gcn', 'bridge-gat'], [0])
        self.assertEqual([flag(job, 'tr_ffn_dim') for job in plan], ['512', '512', '256', '256', '256', '256'])
        self.assertEqual([flag(job, 'tr_nhead') for job in plan], ['4', '4', '4', '4', '2', '2'])
        self.assertEqual([flag(job, 'tr_dropout') for job in plan][-2:], ['0.15', '0.2'])
        self.assertTrue(all(flag(job, 'lr') == '0.0003' for job in plan))
        self.assertTrue(all('--bridge' in job['command'] for job in plan))

    def test_temporal_mapping_and_batch(self):
        plan = runner.build_plan(self.registry, 'link', ['brightkite'], ['tgn', 'tcl', 'tpnet'], [0])
        self.assertEqual([job['command'][1] for job in plan], ['temporal_graph.py', 'temporal_graph_new.py', 'temporal_graph_new.py'])
        self.assertEqual([flag(job, 'train_batch_size') for job in plan], ['4096', '1024', '1024'])
        fraud = runner.build_plan(self.registry, 'fraud', ['movies'], ['tgn'], [0])[0]
        self.assertEqual(flag(fraud, 'train_batch_size'), '32768')
        self.assertIn('--use_sequence_embedding', fraud['command'])

    def test_two_stage_and_sequence_only_mapping(self):
        link = runner.build_plan(self.registry, 'link', ['movies'], ['two-stage-gat'], [0])[0]
        self.assertIn('--use_sequence_embedding', link['command'])
        self.assertTrue(link['unresolved'])
        seq = runner.build_plan(self.registry, 'fraud', ['movies'], ['sbert'], [0])[0]
        self.assertEqual(seq['command'][1], 'review_embedding.py')
        self.assertIn('--seq_only', seq['command'])
        self.assertTrue(seq['unresolved'])

    def test_all_registry_commands_match_source_argparse(self):
        for suite in ['link', 'fraud']:
            methods = sorted({row['method'] for row in self.registry['rows'] if row['suite'] == suite})
            runner.build_plan(self.registry, suite, methods=methods, seeds=[0])

    def test_dry_run_never_launches(self):
        for mode in [[], ['--dry-run']]:
            with patch.object(runner.subprocess, 'run') as launch, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(['--suite', 'fraud', '--datasets', 'clothing', '--methods', 'bridge-gcn', '--seeds', '0'] + mode), 0)
                launch.assert_not_called()

    def test_run_refuses_entire_unresolved_selection_before_launch(self):
        plan = runner.build_plan(self.registry, 'fraud', ['movies'], ['bridge-gcn', 'sbert'], [0])
        for job in plan:
            job['required_files'] = []
        with patch.object(runner.subprocess, 'run') as launch:
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                runner.run_plan(plan)
            launch.assert_not_called()

    def test_run_requires_artifacts_and_correct_cwd(self):
        plan = runner.build_plan(self.registry, 'link', ['brightkite'], ['bridge-gcn'], [0])
        with patch.object(runner.subprocess, 'run') as launch:
            with patch.object(Path, 'is_file', return_value=False), self.assertRaisesRegex(ValueError, 'missing required artifact'):
                runner.run_plan(plan)
            launch.assert_not_called()
            with patch.object(Path, 'is_file', return_value=True):
                runner.run_plan(plan)
            launch.assert_called_once_with(plan[0]['command'], cwd=str(ROOT / 'brightkite'), check=True)

    def test_bad_selection_rejected(self):
        for kwargs in [{'datasets':['github']}, {'methods':['TokenXAT']}, {'seeds':[0, 0]}, {'seeds':[-1]}]:
            with self.assertRaises(ValueError):
                runner.build_plan(self.registry, 'link', **kwargs)


if __name__ == '__main__':
    unittest.main()
