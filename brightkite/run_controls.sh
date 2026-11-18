#!/usr/bin/env bash
# Decisive fair-controls driver (reviewer R1-4 / R3-1 / AC-3).
# Runs the NEW controls that launch_exp.py does not cover, on a local multi-GPU box.
# TokenXAT is DROPPED: BRIDGE = end-to-end sequence encoder (mean-pooled) + a STANDARD GNN.
# So BRIDGE == the "baseline" config (end-to-end GCN/GAT/TransformerConv), graph-only + static
# frozen two-stage == the "static" config. Do NOT run --config tokenxat.
#   python launch_exp.py --config baseline|static --dataset <ds>
#
# This script adds:
#   (1) MLM pretraining of the sequence encoder  -> stored/mlm_<ds>.pth
#   (2) unfrozen two-stage  = pretrained encoder, fine-tuned jointly with the GNN
#   (3) frozen two-stage (arch-matched) = pretrained (unsupervised MLM) encoder, frozen, + GNN  [variant (i)]
#   (4) supervised-frozen (variant D) = task-supervised sequence-only warm-up, frozen, + GNN  [the decisive D-vs-E]
#
# CRITICAL: the pretraining BERT config MUST equal joint.py's encoder config, or the
# --pretrained_seq_ckpt loader drops weights. We pin both to layers=1 heads=2 hidden=128
# max_events=100. joint.py logs [pretrained-seq] missing/unexpected keys — verify it says
# missing=0 (a few buffer keys like position_ids are fine).
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # reclaim fragmented reserved memory

# Optional dataset arg(s): `bash run_controls.sh gowalla` runs one dataset (for multi-machine
# splits). No args = both.
if [[ $# -gt 0 ]]; then DATASETS=("$@"); else DATASETS=(brightkite gowalla); fi
SEEDS=(0 1 2)
# Detect GPUs with <15GB used (mirror the fraud pool's guard) so a box with a different GPU set / a
# busy RelGT pool doesn't dispatch onto phantom or occupied devices (invalid device ordinal / OOM).
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null | awk -F',' '$2+0<15000{print $1}'))
[[ ${#GPUS[@]} -eq 0 ]] && { echo "[gpu-guard] no free GPUs on this box; aborting"; exit 1; }
NGPU=${#GPUS[@]}
echo "[gpu-guard] using ${NGPU} GPU(s): ${GPUS[*]}"
mkdir -p stored

# GNN backbone (STANDARD GNN -- TokenXAT dropped). Must match the BRIDGE backbone you report,
# and must match the "baseline" config runs. Options mirror BASELINE_GRID:
#   GCN,GCN | GAT,GAT | TransformerConv,TransformerConv
GNN="GCN,GCN"
BERT_ARGS="--bert_num_layers 1 --bert_num_heads 2 --bert_hidden_size 128 --max_events 100"
GNN_ARGS="--gnn_layer_types ${GNN} --encoder_compress mean --gnn_heads 2 --num_neighbors 10 --num_negative_samples 100 --enable_early_stopping --skip_if_exists --resume"

# --- (1) Pretrain the MLM encoder once per dataset (matched config) ---
for ds in "${DATASETS[@]}"; do
  ckpt="mlm_${ds}_1L.pth"
  if [[ -f "stored/${ckpt}" ]]; then
    echo "[pretrain] stored/${ckpt} exists, skipping"
  else
    echo "[pretrain] training MLM encoder for ${ds}"
    python sequence_embedding.py --mode train \
      --num_hidden_layers 1 --num_attention_heads 2 --hidden_size 128 --max_events 100 \
      --csv_file "../data/tasks/${ds}/friend_recommendation/personal.csv" \
      --model_file "${ckpt}"
  fi
done

# --- (2)+(3) Launch controls across GPUs (round-robin, batches of NGPU) ---
job=0
launch() {  # $1=slot (0..NGPU-1) into GPUS[]  $2...=command
  local gpu="${GPUS[$1]}"; shift
  echo "[gpu ${gpu}] $*"
  CUDA_VISIBLE_DEVICES="${gpu}" "$@" &
}

for ds in "${DATASETS[@]}"; do
  ckpt="stored/mlm_${ds}_1L.pth"
  bs=64; [[ "${ds}" == "gowalla" ]] && bs=32   # bs128 OOMs (~38GB + spikes) on 48GB; gowalla denser -> 32
  for seed in "${SEEDS[@]}"; do
    # unfrozen two-stage (the decisive control)
    launch $((job % NGPU)) python joint.py --dataset "${ds}" --seed "${seed}" \
      ${BERT_ARGS} ${GNN_ARGS} --batch_size ${bs} --eval_batch_size $((bs*2)) --pretrained_seq_ckpt "${ckpt}" --run_tag unfrozen2stage
    job=$((job+1)); [[ $((job % NGPU)) -eq 0 ]] && wait

    # frozen two-stage, architecture-matched
    launch $((job % NGPU)) python joint.py --dataset "${ds}" --seed "${seed}" \
      ${BERT_ARGS} ${GNN_ARGS} --batch_size ${bs} --eval_batch_size $((bs*2)) --pretrained_seq_ckpt "${ckpt}" --freeze_seq_encoder --run_tag frozenseq_matched
    job=$((job+1)); [[ $((job % NGPU)) -eq 0 ]] && wait

    # (4) supervised-frozen (variant D, the DECISIVE control): task-supervised sequence-only warm-up
    # on the LINK target (no graph) -> freeze -> train GNN. NOT the MLM ckpt (that is frozenseq_matched
    # above = variant (i)); this is D. Warm up to convergence (watch the "[supervised-frozen warm-up]"
    # BCE in the log) -- a weak D makes an E>D gap unconvincing.
    launch $((job % NGPU)) python joint.py --dataset "${ds}" --seed "${seed}" \
      ${BERT_ARGS} ${GNN_ARGS} --batch_size ${bs} --eval_batch_size $((bs*2)) --supervised_warmup_epochs 100 --run_tag supfrozen_D
    job=$((job+1)); [[ $((job % NGPU)) -eq 0 ]] && wait
  done
done
wait
echo "DONE. Collect with: the result collection notebook  (results in {ds}_results_seeded/*_{unfrozen2stage,frozenseq_matched,supfrozen_D}.json)"

# NOTE: the supervised-frozen control (variant D) is now implemented natively in joint.py via
#   --supervised_warmup_epochs (sequence-only link warm-up on the relationship target -> freeze -> GNN),
