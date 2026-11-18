#!/usr/bin/env bash
# LOG 2026 camera-ready A6: Gowalla no-relation ablation (promised to reviewer orZT).
#
# Re-trains the main-table Gowalla BRIDGE-GCN run for each seed with ONLY --no_relational changed:
# every other training flag is inherited from that seed's main-table checkpoint
# (joint.py --args_from_checkpoint). The main-table run itself is the "with relations" row.
#
#   bash run_gowalla_norel.sh check   # print each seed's effective args + diff vs reference; no training
#   bash run_gowalla_norel.sh run     # launch seeds 0-2 in parallel, one GPU each
#
# No reference checkpoints (e.g. they were lost)? Set MAIN_TABLE_FLAGS=1 to use the main-table Gowalla
# command reconstructed from launch_exp.py (BASELINE_GRID GCN entry + its Gowalla batch-size override);
# every other setting is joint.py's default, unchanged since the Sept 2025 main-table runs:
#   MAIN_TABLE_FLAGS=1 bash run_gowalla_norel.sh check   # prints the effective settings
#   MAIN_TABLE_FLAGS=1 bash run_gowalla_norel.sh run
#
# Env overrides:
#   REF_DIR   dir with the main-table Gowalla checkpoints (default: gowalla_checkpoints_seeded)
#   REF_GLOB  filename glob for one seed's reference checkpoint; SEED is replaced by the seed
#             (default: e2e_gcn-gcn*_bert*_bs32_lr0.0001_epochs100_seedSEED_best.pth)
#   GPUS      space-separated GPU ids, one per seed (default: "0 1 2")
#
# In `check` output, the diff should show include_relational True -> False plus only
# path/GPU/output keys. Anything else (e.g. a key shown as '<absent>' -> default because the
# reference was trained with an older joint.py) must be set explicitly before running.
# Results: gowalla_results_seeded/*_no-relational_*_seed{0,1,2}.json (no collision with the
# main-table JSONs; completed runs are skipped, partial runs resume).
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-check}"
REF_DIR="${REF_DIR:-gowalla_checkpoints_seeded}"
REF_GLOB="${REF_GLOB:-e2e_gcn-gcn*_bert*_bs32_lr0.0001_epochs100_seedSEED_best.pth}"
read -r -a GPUS <<< "${GPUS:-0 1 2}"
SEEDS=(0 1 2)
MAIN_FLAGS=(--dataset gowalla --gnn_layer_types GCN,GCN --num_neighbors 10 --gnn_heads 2
            --enable_early_stopping --eval_batch_size 64 --batch_size 32 --num_negative_samples 100)
mkdir -p norel_logs

ref_for_seed() {  # exactly one match, or abort
  local pattern="${REF_GLOB//SEED/$1}"
  local matches=()
  shopt -s nullglob
  matches=("${REF_DIR}"/${pattern})
  shopt -u nullglob
  if [[ ${#matches[@]} -ne 1 ]]; then
    echo "[error] seed $1: expected 1 reference checkpoint for ${REF_DIR}/${pattern}, found ${#matches[@]}:" >&2
    [[ ${#matches[@]} -gt 0 ]] && printf '  %s\n' "${matches[@]}" >&2
    exit 1
  fi
  echo "${matches[0]}"
}

for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  if [[ "${MAIN_TABLE_FLAGS:-0}" == 1 ]]; then
    ref="main-table flags (no reference checkpoint)"
    cmd=(python joint.py "${MAIN_FLAGS[@]}" --no_relational --seed "${seed}"
         --experiment_name "gowalla-norel_s${seed}" --skip_if_exists --resume)
  else
    ref="$(ref_for_seed "${seed}")"
    cmd=(python joint.py --args_from_checkpoint "${ref}" --no_relational --seed "${seed}"
         --experiment_name "gowalla-norel_s${seed}" --skip_if_exists --resume)
  fi
  case "${MODE}" in
    check)
      echo "===== seed ${seed}: ${ref}"
      if [[ "${MAIN_TABLE_FLAGS:-0}" == 1 ]]; then
        "${cmd[@]}" --print_args | grep -E '^(include_relational|dataset|gnn_layer_types|gnn_num_layers|gnn_heads|num_neighbors|batch_size|eval_batch_size|lr|num_epochs|max_events|patience|eval_every|bert_num_layers|bert_num_heads|bert_hidden_size|encoder_compress|input_path) ='
      else
        "${cmd[@]}" --print_args | sed -n '/^# differences/,$p'
      fi
      ;;
    run)
      gpu="${GPUS[$i]}"
      log="norel_logs/gowalla_norel_seed${seed}.log"
      echo "[gpu ${gpu}] seed ${seed} <- ${ref}  (log: ${log})"
      CUDA_VISIBLE_DEVICES="${gpu}" "${cmd[@]}" > "${log}" 2>&1 &
      ;;
    *)
      echo "usage: $0 check|run" >&2; exit 2 ;;
  esac
done
[[ "${MODE}" == "run" ]] && wait && echo "DONE. Results in gowalla_results_seeded/*_no-relational_*.json"
exit 0
