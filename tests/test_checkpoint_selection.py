from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CheckpointSelectionTest(unittest.TestCase):
    def test_link_trainers_copy_validation_selected_state(self):
        paths = [
            "brightkite/static_graph.py", "amazon/static_graph.py",
            "brightkite/temporal_graph.py", "amazon/temporal_graph.py",
            "brightkite/temporal_graph_new.py", "amazon/temporal_graph_new.py",
        ]
        for path in paths:
            text = (ROOT / path).read_text()
            with self.subTest(path=path):
                self.assertIn("copy.deepcopy", text)
                self.assertIn("validation_mrr", text)
        for path in ("brightkite/static_graph.py", "amazon/static_graph.py"):
            text = (ROOT / path).read_text()
            self.assertIn('"predictor": copy.deepcopy(predictor.state_dict())', text)
            self.assertIn('predictor.load_state_dict(best_state["predictor"])', text)

    def test_fraud_relational_baselines_select_by_validation_pr_auc(self):
        for path in ("RelGNN-main/examples/relgnn_node_fraud.py", "relgt-main/main_node_ddp.py"):
            text = (ROOT / path).read_text()
            self.assertIn('tune_metric = "average_precision"', text)

    def test_amazon_task_single_worker_path_is_seeded_and_noninteractive(self):
        text = (ROOT / "notebooks/task_amazon.py").read_text()
        self.assertNotIn("breakpoint()", text)
        self.assertIn("candidates = sorted(all_possible_ids - positive_set)", text)
        self.assertIn("if n_jobs == 1:", text)


if __name__ == "__main__":
    unittest.main()
