# Repository map

| Location | Retained purpose |
| --- | --- |
| `brightkite/` | Brightkite/Gowalla training, baselines, evaluation, controls, ablations and timing |
| `amazon/` | Amazon co-review training, fraud relational-table adapters, baseline pools and diagnostics |
| `github/` | GitHub collaboration baselines and their GPU orchestrator |
| `notebooks/` | Raw-data preparation, task splits, fraud training and shared result collection |
| `common/` | One shared temporal-memory implementation and one shared TNCN predictor |
| `TPNet/`, `RelGNN-main/`, `relgt-main/` | Left empty; please download the third-party code separately (see [README](../README.md#third-party-code-not-included)) |
| `relbench_fraud_dataset.py` | Shared adapter for the relational fraud baselines |
| `scripts/`, `tests/` | Source audit/export tools and focused offline regression tests |

[The experiment map](../README.md) lists the supported execution paths. Run task
scripts from their documented directory: neighboring imports and relative input
paths still follow that convention.

Identical temporal-memory and TNCN modules have been consolidated into `common/`.
The shared implementations are byte-identical to the former per-task copies.
Dataset-specific `joint.py`, sequence encoders and attention layers remain separate
because their behavior differs; merging them would change more than file layout.

Earlier next-event forks, unused training copies, exploratory/debug programs,
duplicate table generators, upstream benchmark examples/tutorials/tests, private
cluster tooling and generated result/figure snapshots are absent from the source
snapshot. Local ignored experiment outputs and checkpoints are preserved.
Third-party licenses and citations remain with the retained baseline code.
