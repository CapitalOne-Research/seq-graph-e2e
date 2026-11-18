# Why Freeze the Encoder? A Controlled Study of End-to-End Sequence–Graph Learning for User Behavior

[![Paper](https://img.shields.io/badge/arXiv-2510.25126-b31b1b.svg)](https://arxiv.org/abs/2510.25126)
[![Venue](https://img.shields.io/badge/LoG-2026-4c6ef5.svg)](https://logconference.org/)
[![Python](https://img.shields.io/badge/Python-3.12-3776ab.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/offline_tests-50%20passing-2ea44f.svg)](#verification)

Reference implementation and reproducibility artifact for **BRIDGE**, the
end-to-end sequence–graph model studied in:

> **Why Freeze the Encoder? A Controlled Study of End-to-End Sequence–Graph
> Learning for User Behavior**<br>
> Yuen Chen, Yulun Wu, Samuel Sharpe, Igor Melnyk, Nam H. Nguyen, Furong Huang,
> C. Bayan Bruss, and Rizal Fathony<br>
> *Learning on Graphs Conference (LoG), 2026*

## TL;DR

User behavior combines an asynchronous event sequence for each user with a
persistent user-to-user graph. BRIDGE feeds Transformer sequence representations
into a GNN and trains both modules jointly. The controlled study finds that the
value of end-to-end training depends on where the predictive signal lies:

- On relationship prediction, end-to-end updating consistently improves over
  frozen two-stage pipelines. A matched Brightkite control attributes a
  **10.6-point MRR gain** to updating the encoder.
- On fraud detection, gains shrink when the signal is largely node-local; a
  text-fed relational graph transformer is competitive in this regime.
- Recasting the same data as a temporal graph produces inconsistent gains,
  supporting the native sequence–graph formulation for persistent relations.

See the [paper](https://arxiv.org/abs/2510.25126) for the controlled comparisons,
statistical tests, ablations, and discussion.

## Method

BRIDGE preserves the two native structures in behavioral data:

```text
per-user event history ── Transformer ──┐
                                        ├── user representation ── task head
persistent user relation graph ── GNN ──┘
```

The study holds this sequence-encoder-to-GNN architecture fixed and changes the
optimization strategy:

| Setting | Sequence encoder | Graph model | Purpose |
| --- | --- | --- | --- |
| Graph-only | absent | trained | measures relational signal |
| Frozen two-stage | pretrained, frozen | trained | standard reusable-embedding pipeline |
| BRIDGE | jointly updated | jointly updated | measures end-to-end adaptation |
| Matched frozen control | same initialization as BRIDGE, frozen | trained | isolates the effect of updating the encoder |

Static GCN, GAT, and TransformerConv backbones are evaluated alongside temporal
graph and relational deep-learning alternatives.

## Experimental scope

| Task | Datasets | Primary metrics | Model families |
| --- | --- | --- | --- |
| Relationship prediction | Brightkite, Gowalla, Amazon Movies, Electronics, Clothing | MRR, Hits@K | graph-only, frozen two-stage, BRIDGE, temporal graph |
| Fraud detection | Amazon Movies, Electronics, Clothing | PR-AUC, ROC-AUC | static, sequence, BRIDGE, temporal, RelGNN, RelGT |

The repository also retains PRES GitHub collaboration workflows, camera-ready
controls, no-relation and architecture ablations, timing studies, leakage
diagnostics, and paired significance tests.

## Quick start

### 1. Install

The reference environment uses Linux, Python 3.12, PyTorch 2.5.1, and CUDA 12.4.

```bash
bash setup.sh tgs
conda activate tgs
```

### 2. Inspect the data plan

```bash
python scripts/download_data.py --pres-datasets all --raw-amazon all --dry-run
```

### 3. Inspect a fixed paper configuration

```bash
python scripts/reproduce_paper.py --suite link \
  --datasets brightkite gowalla movies electronics clothing \
  --methods bridge-gcn bridge-gat bridge-tfconv \
  --seeds 0 1 2 --dry-run
```

Dry-run is the default. Training starts only with `--run`, after the runner has
validated the complete selection and its required artifacts.

## Reproducing the paper

The sweep launchers are useful for exploration. Paper reproduction should use
the fixed registry:

- [`configs/paper_experiments.json`](configs/paper_experiments.json) records every
  dataset/method row, its paper table and page, stated hyperparameters, inferred
  retained-code defaults, required files, and unresolved provenance.
- [`scripts/reproduce_paper.py`](scripts/reproduce_paper.py) builds commands
  without importing GPU training modules. It validates the entire plan before
  launching any job.

```bash
# All main relationship-prediction rows, three seeds each.
python scripts/reproduce_paper.py --suite link --dry-run

# All main fraud-detection rows, with provenance and blockers.
python scripts/reproduce_paper.py --suite fraud --json

# Execute only a fully resolved subset.
python scripts/reproduce_paper.py --suite link \
  --datasets brightkite gowalla movies electronics clothing \
  --methods static-gcn static-gat static-tfconv \
            bridge-gcn bridge-gat bridge-tfconv \
  --seeds 0 1 2 --run
```

The paper specifies three independent runs without assigning numeric seeds.
Seeds `0,1,2` follow the retained launcher convention.

### Reproduction boundaries

The registry blocks rows whose exact historical configuration cannot be inferred
responsibly. These include frozen-pretraining artifacts, SBERT-only selection,
RelGT variants, TNCN, and additional temporal fraud head settings. Supported rows
combine paper-stated values with defaults explicitly labeled as inference. They
do not claim to recover original run metadata.

## Data

### Prepared PRES release

The companion dataset paper
[arXiv:2510.11903](https://arxiv.org/abs/2510.11903) identifies the public
PRES release (hosted on Hugging Face; see the paper for the exact dataset
repo id) as the prepared source and the [UCSD Amazon review
collection](https://cseweb.ucsd.edu/~jmcauley/datasets/amazon/links.html) as the
raw Amazon source.

Download all five published PRES datasets and the three raw Amazon categories
used in the BRIDGE experiments:

```bash
python scripts/download_data.py --pres-datasets all --raw-amazon all \
  --pres-repo <org>/PRES --accept-amazon-license
```

`--pres-repo` can also be set via the `PRES_REPO` environment variable; see
the companion paper above for the dataset repo id. The pinned PRES revision is
`6f0a75d92c29586b7179674226ae4bc466a11159`. The complete revision is roughly
132 GB before the raw Amazon archives. Replace `all` with selected dataset names
for a smaller download.

PRES provides prepared Clothing and Electronics tasks. Amazon Movies, review
text labels, and canonical fraud splits require the raw-data reconstruction
below. Movies is an additional BRIDGE input and is not part of the five-dataset
PRES release.

### Reconstruct enriched Amazon data

```bash
for category in movies electronics clothing; do
  python scripts/process_amazon_reviews.py --category "$category"
done

# PRES does not include the Movies task split.
cd notebooks
python task_amazon.py --dataset amazon-movies --seed 42 --n_jobs 1 \
  --neg_sample_cnt_coreview 1000 --neg_sample_cnt_prodrating 1000
cd ..
```

The processor preserves the study thresholds: five shared products define a
Movies co-review relation, while three define Electronics and Clothing
relations. It also derives fraud labels from review helpfulness. A reconstructed
Movies split is deterministic with `--n_jobs 1`, although it cannot establish
byte identity with unavailable historical artifacts.

### Create 100-negative evaluation files

The released evaluation pools contain 1,000 negatives per positive edge. The
paper protocol retains the first 100 from each aligned block:

```bash
python scripts/create_negative_sample_100.py --data-dir \
  data/tasks/amazon-clothing/coreview_prediction \
  data/tasks/amazon-electronics/coreview_prediction \
  data/tasks/brightkite/friend_recommendation \
  data/tasks/gowalla/friend_recommendation
```

Add `data/tasks/amazon-movies/coreview_prediction` after reconstructing Movies.
The converter streams the inputs, validates complete blocks and UID alignment,
and preserves the original negative files.

## Training

### Relationship prediction

The fixed runner above is preferred for paper rows. The original sweep launchers
remain available for exploratory grids.

```bash
# Brightkite and Gowalla
cd brightkite
python launch_exp.py --dataset brightkite --config baseline
python launch_exp_static.py
python launch_exp_tgn.py

# Amazon Movies, Electronics, and Clothing
cd ../amazon
python launch_exp.py --run_baseline
python launch_exp.py
python launch_exp_static.py
python launch_exp_tgn.py
```

All Amazon sweep launchers include the three reported categories.

### Fraud detection

```bash
cd notebooks
python launch_fraud.py --dataset clothing --config bridge
python launch_fraud.py --dataset clothing --config baseline
python launch_fraud.py --dataset clothing --config seq_only
python launch_fraud.py --dataset clothing --config tgn
```

### RelGNN and RelGT fraud baselines

Run the following ordered pipeline from the repository root:

```bash
# Build enriched review nodes and the transductive co-review graph.
for ds in movies electronics clothing; do
  python notebooks/create_fraud_data.py --dataset "$ds" --seed 0
done

# Materialize canonical user-label splits for every seed.
for ds in movies electronics clothing; do
  for seed in 0 1 2; do
    python amazon/splits_pt_to_arena.py --dataset "$ds" --seed "$seed" \
      --processed "data/processed/amazon-$ds" \
      --out "data/tasks/amazon-$ds/fraud_detection_seed$seed"
  done
  python amazon/verify_canonical_split.py \
    --processed "data/processed/amazon-$ds" --dataset "$ds" --seeds 0 1 2
done

# Build relational tables.
for seed in 0 1 2; do
  python amazon/build_relational_tables.py --data_root data/tasks \
    --datasets movies electronics clothing --task "fraud_detection_seed$seed"
done

# Print all planned jobs without probing GPUs or starting training.
python amazon/run_fraud_rdl_pool.py --gpus 0 --models relgnn,relgt \
  --relgnn_datasets movies,electronics,clothing \
  --relgt_datasets movies,electronics,clothing \
  --seeds 0,1,2 --dry_run
```

The canonical helper reuses an existing CSV, recovers a historical temporal
split when available, or persists a seeded 60/20/20 user split. Fresh splits are
reproducible but do not establish historical identity. The fraud graph is
transductive: graph relations are shared across splits, while user labels are
partitioned. RelGNN and RelGT select checkpoints using validation PR-AUC.

For the review-text RelGT control:

```bash
for ds in movies electronics clothing; do
  python amazon/build_user_sbert.py --dataset "$ds" \
    --processed "data/processed/amazon-$ds" \
    --relational_glob "data/tasks/amazon-$ds/fraud_detection_seed*/relational" \
    --max_events 50
done
```

## Evaluation and result collection

Configuration selection uses validation MRR across the same three seeds.
Reported link results come exclusively from the held-out test split of the
selected checkpoint.

```bash
python notebooks/link_result_utils.py --task-dir brightkite \
  --datasets brightkite gowalla --output tables/link_results.json

python notebooks/link_result_utils.py --task-dir amazon \
  --datasets movies electronics clothing --output tables/link_amazon.json
```

The collector fails when a selected run lacks held-out metrics. Static and
temporal trainers deep-copy and restore the full validation-selected state,
including the static link predictor.

Fraud tables and machine-readable summaries are generated with:

```bash
cd notebooks
python get_latex_tables.py
cd ..
python amazon/collect_fraud_results.py
```

## Controls and ablations

| Study | Entry point |
| --- | --- |
| Matched frozen/fine-tuned controls | `brightkite/run_controls.sh` |
| Compressed-representation probe | `brightkite/run_probe.sh` |
| Main ablation grid | `brightkite/run_ablation.py --dry-run` |
| No-relation ablation | `brightkite/run_gowalla_norel.sh` |
| Convergence and wall-clock | `brightkite/run_convergence_wallclock.py` |
| Leakage diagnostics | `brightkite/colocation_leakage.py`, `amazon/leakage_diagnostic.py` |
| Degree-stratified analysis | `brightkite/degree_stratified.py` |
| Paired significance tests | `notebooks/paired_wilcoxon_rr.py`, `notebooks/paired_bootstrap_prauc.py` |

See [`brightkite/README_MLM_TIMING.md`](brightkite/README_MLM_TIMING.md) for
matched checkpoint and timing controls. The Gowalla no-relation protocol is
encoded in `brightkite/run_gowalla_norel.sh` and its Slurm counterpart.

## Repository structure

| Path | Purpose |
| --- | --- |
| `brightkite/` | Brightkite/Gowalla models, launchers, controls, and ablations |
| `amazon/` | Amazon relationship models and relational fraud adapters |
| `notebooks/` | Dataset/task preparation, fraud models, and analysis utilities |
| `github/` | PRES GitHub collaboration baselines |
| `common/` | Shared temporal memory and TNCN implementations |
| `TPNet/`, `RelGNN-main/`, `relgt-main/` | Empty placeholders for third-party baselines; see below |
| `configs/` | Fixed paper-row registry |
| `scripts/` | Download, preprocessing, reproduction, audit, and export tools |
| `tests/` | CPU/offline regression tests |

See [`docs/STRUCTURE.md`](docs/STRUCTURE.md) for the full repository map.

## Third-party code (not included)

Three baselines used by the temporal-link and RelGNN/RelGT fraud launchers
come from other authors' repositories, each under its own license, so we
keep this repository to our own code only. Their folders are left empty
here; please download each one into the matching folder before running the
launchers that depend on it:

| Folder | Upstream repository |
| --- | --- |
| `TPNet/` | <https://github.com/lxd99/TPNet> |
| `RelGNN-main/` | <https://github.com/snap-stanford/RelGNN> |
| `relgt-main/` | <https://github.com/snap-stanford/relgt> |

```bash
git clone https://github.com/lxd99/TPNet.git /tmp/TPNet && cp -r /tmp/TPNet/* TPNet/
git clone https://github.com/snap-stanford/RelGNN.git /tmp/RelGNN && cp -r /tmp/RelGNN/* RelGNN-main/
git clone https://github.com/snap-stanford/relgt.git /tmp/relgt && cp -r /tmp/relgt/* relgt-main/
```

Please check each upstream repository's own license before use.

## Verification

```bash
python scripts/audit_public_tree.py
python -m unittest discover -s tests
python scripts/export_public_source.py \
  --output dist/event-seq-graph-source-new.zip
```

Current offline status:

- **50 regression tests passing**
- **159 source files audited with zero findings**
- Python syntax, JSON registry, notebook metadata, download planning, split
  verification, result selection, and source-export integrity checked

These checks do not substitute for full GPU reruns or numerical reproduction of
the paper tables.

## Citation

```bibtex
@inproceedings{chen2026freeze,
  title     = {Why Freeze the Encoder? A Controlled Study of End-to-End
               Sequence--Graph Learning for User Behavior},
  author    = {Chen, Yuen and Wu, Yulun and Sharpe, Samuel and Melnyk, Igor and
               Nguyen, Nam H. and Huang, Furong and Bruss, C. Bayan and
               Fathony, Rizal},
  booktitle = {Proceedings of the Fifth Learning on Graphs Conference},
  series    = {Proceedings of Machine Learning Research},
  volume    = {338},
  year      = {2026}
}
```