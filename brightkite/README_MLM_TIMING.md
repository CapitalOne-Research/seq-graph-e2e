# Rebuttal Experiment Run Instructions

This document contains the remaining commands that require access to the experiment data and compute environment.

## Fastest Automatic Launch (Recommended)

### Get the camera-ready branch

From an existing checkout of the repository, the collaborator should run:

```bash
git fetch origin
git switch --track origin/camera-ready
```

If the local `camera-ready` branch already exists, use:

```bash
git switch camera-ready
git pull --ff-only
```

From the repository root, run:

```bash
python brightkite/run_rebuttal_experiments.py
```

The launcher detects every GPU with at most 500 MiB used and at least 43,000 MiB free, verifies that each isolated child's logical `cuda:0` maps to the intended physical GPU UUID, starts one worker per eligible GPU, and feeds pending jobs from a shared queue. Consequently, each GPU receives its next checkpoint evaluation as soon as it becomes free rather than waiting for a fixed sequential batch. It runs the CPU-only Amazon overlap diagnostic concurrently and starts the paired statistical analysis after all six prediction dumps are available. Existing outputs are skipped.

Checkpoint evaluation uses batch size 512 and expandable CUDA memory segments. The original eval-only default of 1024 exceeded a 48 GB L40S; 512 retains parallel throughput while leaving adequate memory headroom. The MLM trainer creates `brightkite/stored/` automatically before saving.

Logs are written to `brightkite/rebuttal_run_logs/`; the paired-test report is written to `brightkite/paired_frozen_vs_finetuned.txt`. Do not run the manual commands below at the same time as this launcher.

If a step fails, the final terminal summary identifies the failed job, exit code, full log path, and last 20 log lines. The complete command and working directory are recorded at the top of each log, and Python exceptions are captured there. Missing checkpoints or prediction dumps are printed with their exact paths. Because completed outputs are skipped, correct the reported problem and rerun the same launcher command; successful work will not be repeated.

### If only GPU 0 is used

Each isolated process reports `cuda:0` internally because its assigned physical GPU is the only device visible inside that process. This is expected and does not by itself mean all jobs are using physical GPU 0. Before launching jobs, require a distinct `[binding verified] physical GPU N -> child cuda:0` line for every eligible GPU. The launcher aborts before loading a model if the observed device UUID does not match the intended physical GPU.

Read the launcher's initial preflight output. It prints (1) every GPU visible to `nvidia-smi` and why any GPU was rejected, (2) all pending GPU jobs, and (3) every job-to-GPU assignment. Multiple GPUs can be used only when multiple GPU jobs are pending.

In a new Git worktree, datasets and checkpoints that are untracked or ignored are not copied automatically. Confirm that the six checkpoints are available at:

```text
brightkite/brightkite_checkpoints_seeded/
```

If the launcher prints six `[missing] checkpoint` messages, link the server's existing checkpoint directory into that exact location (or copy it there) before rerunning. If it reports only GPU 0 as visible or eligible, inspect the machine allocation and thresholds with:

```bash
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,memory.total --format=csv,noheader,nounits
```

Useful controls:

```bash
# Limit the scheduler to four otherwise-empty GPUs.
python brightkite/run_rebuttal_experiments.py --max-gpus 4

# Omit a component that is not needed.
python brightkite/run_rebuttal_experiments.py --skip-mlm
python brightkite/run_rebuttal_experiments.py --skip-amazon
```

If the server reserves some GPU memory while otherwise idle, adjust the thresholds deliberately with `--max-used-mib` and `--min-free-mib`. Keep one process per L40S because fine-tuned evaluation has previously used approximately 41 GB.

## Resource Summary

Estimates assume one process per 48 GB NVIDIA L40S. The historical checkpoint evaluations took approximately 55 minutes each, but the MLM estimate remains provisional until the timed run completes.

| Experiment | Number of runs | Expected L40S GPU-hours | Suggested reservation |
|---|---:|---:|---:|
| Brightkite MLM timing | 1 | 1–4 for MLM optimization; approximately 0.5–1 additional for embedding generation | 1 L40S for 5 hours |
| Amazon product-overlap baseline | 1 test sweep, optional validation sweep | 0 (CPU-only) | CPU for up to 1 hour |
| Matched frozen/fine-tuned prediction dumps | 6 checkpoint evaluations | Approximately 5.5 total | 3 L40S for 2–3 hours, or 6 L40S for 1–2 hours |
| Paired bootstrap and Wilcoxon analysis | 1 analysis over all dumps | 0 (CPU-only) | CPU for less than 1 hour |
| **Total** |  | **Approximately 7–11 L40S GPU-hours** | **Reserve 12 L40S GPU-hours** |

The automatic launcher uses all sufficiently empty L40S GPUs by default. With four GPUs, three can evaluate the six checkpoints in two waves while the fourth performs the MLM run. Fine-tuned checkpoint evaluation has previously used approximately 41 GB, so do not colocate two evaluation processes on one L40S.

## 1. Measuring MLM Pretraining Time

**Expected resources:** one L40S; approximately 1–4 GPU-hours for MLM optimization, plus approximately 0.5–1 hour for the subsequent embedding-generation step. Reserve one L40S for five hours. The run itself will replace this estimate with a measured value.

Run the command below from the `brightkite/` directory. It uses the same MLM encoder configuration as the matched frozen-versus-fine-tuned controls and writes to new filenames so existing checkpoints and embeddings are not overwritten.

From the repository root:

```bash
cd brightkite

python sequence_embedding.py \
  --mode train \
  --num_hidden_layers 1 \
  --num_attention_heads 2 \
  --hidden_size 128 \
  --max_events 100 \
  --batch_size 256 \
  --num_epochs 50 \
  --learning_rate 1e-3 \
  --csv_file ../data/tasks/brightkite/friend_recommendation/personal.csv \
  --model_file mlm_brightkite_1L_timed.pth \
  --embedding_file mlm_brightkite_1L_timed_embeddings.pkl \
  2>&1 | tee mlm_brightkite_timing.log
```

## Timing output

The trainer prints two measurements at the end of MLM training:

- `MLM optimization wall-clock`: time spent in the optimization loop.
- `MLM stage wall-clock`: vocabulary/model setup, data preparation, optimization, and checkpoint saving.

The structured timing record is saved to:

```text
stored/mlm_brightkite_1L_timed.pth.timing.json
```

Inspect it with:

```bash
jq . stored/mlm_brightkite_1L_timed.pth.timing.json
```

The checkpoint, generated embeddings, and complete console log are saved as:

```text
stored/mlm_brightkite_1L_timed.pth
stored/mlm_brightkite_1L_timed_embeddings.pkl
mlm_brightkite_timing.log
```

## Reporting

Use `mlm_stage_wall_sec` when reporting the MLM pretraining-stage cost. This measurement excludes the subsequent generation of user embeddings. Record the GPU model separately (for example, with `nvidia-smi --query-gpu=name --format=csv,noheader`) because wall-clock values are comparable only when runs use the same hardware.

## 2. Amazon Product-Overlap Baseline

**Expected resources:** no GPU. Run on CPU; allow up to one hour for the three reported Amazon datasets. The validation sweep, if run, is also CPU-only.

This diagnostic addresses the reviewer concern that Amazon relationship labels are defined using shared products while product IDs also appear in the input sequences. It evaluates a training-free predictor that ranks candidate user pairs by their number of shared observed ASINs.

From the repository root:

```bash
cd amazon
```

### Verify the required files

```bash
for dataset in movies electronics clothing; do
  ls ../data/tasks/amazon-${dataset}/coreview_prediction/personal_observed.csv
  ls ../data/tasks/amazon-${dataset}/coreview_prediction/relational_test.csv
  ls ../data/tasks/amazon-${dataset}/coreview_prediction/relational_test_negative_sample_100.csv
done
```

### Run the test-set diagnostic

```bash
python leakage_diagnostic.py \
  --data_root ../data/tasks \
  --datasets movies electronics clothing \
  --split test \
  --neg_suffix _negative_sample_100 \
  2>&1 | tee amazon_overlap_test_results.txt
```

### Optional validation-set consistency check

```bash
python leakage_diagnostic.py \
  --data_root ../data/tasks \
  --datasets movies electronics clothing \
  --split val \
  --neg_suffix _negative_sample_100 \
  2>&1 | tee amazon_overlap_val_results.txt
```

### Expected output files

```text
amazon/amazon_overlap_test_results.txt
amazon/amazon_overlap_val_results.txt
```

The validation output is optional. Return the test output at minimum.

### Checks before reporting

For every dataset, confirm that the output says `aligned=True`. Report:

- overlap-only MRR;
- Hits@1 and Hits@5;
- percentage of positive pairs sharing at least one observed ASIN;
- mean product overlap for positive and negative pairs; and
- whether the overlap-only metrics approach the corresponding BRIDGE results.

This script addresses the overlap-only-baseline request. It does not perform the separate product-ID-masking experiment.

## 3. Paired Bootstrap and Wilcoxon Tests

**Expected resources:** approximately 5.5 L40S GPU-hours for six checkpoint evaluations (about 55 minutes each), followed by a CPU-only statistical analysis that should take less than one hour. With three L40S GPUs, expect roughly two hours of evaluation wall time; with six, roughly one hour. Allow additional setup and data-loading buffer.

This analysis compares the matched MLM-pretrained frozen and fine-tuned conditions using per-test-query reciprocal ranks. Use the three `bs64` validation-selected checkpoints for each condition. Do not use the stale `bs128`, supervised-frozen, scratch, or final non-`_best.pth` checkpoints.

From the repository root, first verify that all six required checkpoints exist:

```bash
for seed in 0 1 2; do
  ls brightkite/brightkite_checkpoints_seeded/e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs64_lr0.0001_epochs100_seed${seed}_frozenseq_matched_best.pth
  ls brightkite/brightkite_checkpoints_seeded/e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs64_lr0.0001_epochs100_seed${seed}_unfrozen2stage_best.pth
done
```

Generate per-query prediction dumps from the best-validation checkpoints:

```bash
cd brightkite

for seed in 0 1 2; do
  python joint.py \
    --load_checkpoint_and_eval "brightkite_checkpoints_seeded/e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs64_lr0.0001_epochs100_seed${seed}_frozenseq_matched_best.pth" \
    --dump_predictions

  python joint.py \
    --load_checkpoint_and_eval "brightkite_checkpoints_seeded/e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs64_lr0.0001_epochs100_seed${seed}_unfrozen2stage_best.pth" \
    --dump_predictions
done
```

Verify that three prediction dumps were produced for each condition:

```bash
find brightkite_checkpoints_seeded \
  -name '*bs64*_frozenseq_matched_best_test_preds.npz' \
  -print | sort

find brightkite_checkpoints_seeded \
  -name '*bs64*_unfrozen2stage_best_test_preds.npz' \
  -print | sort
```

Run the paired analysis and save its output:

```bash
cd ..

python notebooks/paired_wilcoxon_rr.py \
  --a 'brightkite/brightkite_checkpoints_seeded/*bs64*_unfrozen2stage_best_test_preds.npz' \
  --name-a 'MLM-pretrained + fine-tuned' \
  --b 'brightkite/brightkite_checkpoints_seeded/*bs64*_frozenseq_matched_best_test_preds.npz' \
  --name-b 'MLM-pretrained + frozen' \
  --n-boot 10000 \
  2>&1 | tee brightkite/paired_frozen_vs_finetuned.txt
```

Return these artifacts:

```text
brightkite/brightkite_checkpoints_seeded/*bs64*_best_test_preds.npz
brightkite/paired_frozen_vs_finetuned.txt
```

Before reporting, confirm that the analysis finds seeds 0, 1, and 2, that the number of aligned queries is identical for both models within each seed, and that the MRR reconstructed from each dump matches its corresponding result JSON. Report each seed's MRR gap, paired-bootstrap 95% confidence interval, two-sided Wilcoxon p-value, query count, and the mean $\pm$ SD gap across seeds.
