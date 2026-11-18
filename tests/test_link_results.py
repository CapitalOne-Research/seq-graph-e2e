import importlib.util
from pathlib import Path
import tempfile
import unittest
import json

MODULE_PATH = Path(__file__).resolve().parents[1] / 'notebooks/link_result_utils.py'
SPEC = importlib.util.spec_from_file_location('link_result_utils', MODULE_PATH)
results = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(results)


def run(config, seed, val, test):
    return {'dataset': 'brightkite', 'type': 'e2e', 'method': 'BRIDGE-GCN',
            'configuration': config, 'seed': seed, 'val_mrr': val,
            'test_metrics': {} if test is None else {'mrr': test}, 'source': f'{config}/seed{seed}'}


class LinkResultTest(unittest.TestCase):
    def test_e2e_uses_best_checkpoint_test_not_validation_or_final_test(self):
        rows = [
            {'epoch': 1, 'evaluation_metrics': {'MRR': .9, 'Hits@1': .8}},
            {'epoch': 2, 'evaluation_metrics': {'MRR': .7}},
            {'test_mrr': .99},
            {'best_test_mrr': .25, 'best_test_hits_at_1': .1},
        ]
        row = results.extract_metrics(rows, 'e2e')
        self.assertEqual(row['val_mrr'], .9)
        self.assertEqual(row['selected_epoch'], 1)
        self.assertEqual(row['test_metrics'], {'mrr': .25, 'hits_at_1': .1})

    def test_last_recorded_best_eval_is_used_without_maximizing_test(self):
        rows = [{'epoch': 2, 'evaluation_metrics': {'MRR': .8}},
                {'best_test_mrr': .7},
                {'best_test_mrr': .6, 'eval_only': True, 'checkpoint_used': 'model_best.pth'}]
        self.assertEqual(results.extract_metrics(rows, 'e2e')['test_metrics']['mrr'], .6)

    def test_static_summary_and_explicit_best_checkpoint_schemas(self):
        rows = [{'epoch': 1, 'val_mrr': .8}, {'epoch': 2, 'val_mrr': .7},
                {'summary': True, 'selection': 'validation_mrr',
                 'test_mrr': .3, 'test_hits_at_1': .2}]
        self.assertEqual(results.extract_metrics(rows, 'static')['test_metrics']['mrr'], .3)
        self.assertEqual(results.extract_metrics(rows, 'temporal')['test_metrics']['mrr'], .3)
        e2e = rows[:2] + [{'test_mrr': .4, 'checkpoint_used': 'run_best.pth', 'eval_only': True}]
        self.assertEqual(results.extract_metrics(e2e, 'e2e')['test_metrics']['mrr'], .4)

    def test_final_epoch_summary_is_not_mislabeled_as_validation_selected(self):
        rows = [{'epoch': 1, 'val_mrr': .8},
                {'summary': True, 'selection': 'final_epoch', 'test_mrr': .3}]
        self.assertEqual(results.extract_metrics(rows, 'static')['test_metrics'], {})

    def test_no_validation_fallback_for_missing_best_test_metrics(self):
        row = results.extract_metrics([{'epoch': 1, 'evaluation_metrics': {'MRR': .8}},
                                       {'test_mrr': .9}], 'e2e')
        self.assertEqual(row['test_metrics'], {})
        with self.assertRaisesRegex(results.IncompleteResultsError, 'No finite validation'):
            results.extract_metrics([{'best_test_mrr': .5, 'eval_only': True}], 'e2e')

    def test_hyperparameters_selected_using_validation_mean_only(self):
        runs = [run('a', s, .9, .1 + .1*s) for s in [0, 1, 2]]
        runs += [run('b', s, .8, .99) for s in [0, 1, 2]]
        summary = results.summarize_runs(runs, metrics=('mrr',))[0]
        self.assertEqual(summary['configuration'], 'a')
        self.assertAlmostEqual(summary['test_mrr_mean'], .2)
        self.assertAlmostEqual(summary['test_mrr_std'], .1)
        self.assertEqual(summary['num_seeds'], 3)

    def test_missing_selected_config_test_does_not_select_an_available_runner_up(self):
        runs = [run('a', s, .9, None if s == 1 else .1) for s in [0, 1, 2]]
        runs += [run('b', s, .8, .99) for s in [0, 1, 2]]
        with self.assertRaisesRegex(results.IncompleteResultsError, 'missing held-out'):
            results.summarize_runs(runs, metrics=('mrr',))

    def test_partial_and_duplicate_seeds_are_not_averaged_as_complete_runs(self):
        with self.assertRaisesRegex(results.IncompleteResultsError, 'missing seeds'):
            results.summarize_runs([run('a', 0, .9, .2)], metrics=('mrr',))
        with self.assertRaisesRegex(results.IncompleteResultsError, 'Duplicate seed'):
            results.summarize_runs([run('a', 0, .9, .2), run('a', 0, .9, .3)], metrics=('mrr',))

    def test_distinct_encoder_configs_and_control_tags_remain_distinct(self):
        a = results.parse_filename('e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs128_lr0.0001_epochs100_seed0.json', 'e2e')
        b = results.parse_filename('e2e_gcn-gcn2x2_bert2x2_hidden-dim128_mean_nngb10_nneg100_bs128_lr0.0001_epochs100_seed1.json', 'e2e')
        c = results.parse_filename('e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs128_lr0.0001_epochs100_seed0_frozenseq_matched.json', 'e2e')
        self.assertEqual(a['method'], 'BRIDGE-GCN')
        self.assertNotEqual(a['configuration'], b['configuration'])
        self.assertNotEqual(a['method'], c['method'])

    def test_collect_and_table_require_all_held_out_metrics(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp) / 'brightkite_results_seeded'
            directory.mkdir()
            for seed in [0, 1, 2]:
                rows = [{'epoch': 1, 'evaluation_metrics': {'MRR': .8}},
                        {f'best_test_{metric}': .2 for metric in results.METRICS}]
                (directory / f'e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs128_lr0.0001_epochs100_seed{seed}.json').write_text(json.dumps(rows))
            summaries = results.summarize_runs(results.collect_runs(temp, ['brightkite']))
            table = results.latex_table(summaries, 'brightkite')
            self.assertIn('0.2000', table)
            self.assertNotIn('0.8000', table)
            self.assertEqual(summaries[0]['validation_mrr_mean'], .8)


if __name__ == '__main__':
    unittest.main()
