#!/usr/bin/env bash
# Improved GPU Job Scheduler
# Manages execution of ML experiments across multiple GPUs with dependency tracking

set -o pipefail

# Configuration - Can be overridden via command-line arguments
MAX_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l 2>/dev/null || echo "8")
CHECK_INTERVAL=10
BASE_PATH=""
SEED=42
LOG_DIR="logs"
STORED_DIR="stored"
VERBOSE=0

# Parse command-line arguments
while [[ $# -gt 0 ]]; do
  case "$1" in
    --base_path)    BASE_PATH="$2";     shift 2;;
    --seed)         SEED="$2";          shift 2;;
    --log_dir)      LOG_DIR="$2";       shift 2;;
    --stored_dir)   STORED_DIR="$2";    shift 2;;
    --max_gpus)     MAX_GPUS="$2";      shift 2;;
    --interval)     CHECK_INTERVAL="$2"; shift 2;;
    --verbose)      VERBOSE=1;          shift 1;;
    -h|--help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --base_path PATH    Base path for data files (required)"
      echo "  --seed NUM          Random seed (default: 42)"
      echo "  --log_dir DIR       Directory for logs (default: logs)"
      echo "  --stored_dir DIR    Directory for stored artifacts (default: stored)"
      echo "  --max_gpus NUM      Maximum number of GPUs to use (default: auto-detect)"
      echo "  --interval SEC      Job check interval in seconds (default: 10)"
      echo "  --verbose           Print verbose output"
      echo "  --help              Show this help message"
      exit 0
      ;;
    *) echo "Unknown argument: $1"; exit 1;;
  esac
done

# Validate required arguments
if [[ -z "${BASE_PATH}" ]]; then
  echo "[ERROR] --base_path is required" >&2; exit 1;
fi

# Create directories
mkdir -p "${LOG_DIR}" "${STORED_DIR}"

# Job tracking variables
declare -A running_pids
declare -A job_gpu
declare -A completed
declare -A failed
declare -A runtime

# ---------------------
# Define job functions
# ---------------------

run_job_on_gpu() {
  local job_name="$1"
  local gpu="$2"
  local cmd="$3"

  # Record start time
  start_time=$(date +%s)
  
  (
    export CUDA_VISIBLE_DEVICES="${gpu}"
    echo "[$(date '+%F %T')] START  ${job_name}" > "${LOG_DIR}/${job_name}.log"
    echo "Command: ${cmd}" >> "${LOG_DIR}/${job_name}.log"
    echo "GPU: ${gpu}" >> "${LOG_DIR}/${job_name}.log"
    echo "----------------------------------------" >> "${LOG_DIR}/${job_name}.log"
    
    # Execute the command and capture its exit status
    if eval "${cmd}" >> "${LOG_DIR}/${job_name}.log" 2>&1; then
      echo "[$(date '+%F %T')] FINISH ${job_name} - SUCCESS" >> "${LOG_DIR}/${job_name}.log"
      exit 0
    else
      echo "[$(date '+%F %T')] FINISH ${job_name} - FAILED (exit code $?)" >> "${LOG_DIR}/${job_name}.log"
      exit 1
    fi
  ) &
  
  local pid=$!
  running_pids["$pid"]="$job_name"
  job_gpu["$job_name"]=$gpu
  
  echo "[INFO] Started job '${job_name}' on GPU ${gpu} (PID: ${pid})"
}

# Print current job status
print_status() {
  echo ""
  echo "======== Job Status ========"
  echo "Running: ${#running_pids[@]} | Completed: ${#completed[@]} | Failed: ${#failed[@]} | Remaining: $((total_jobs-${#completed[@]}-${#failed[@]}-${#running_pids[@]}))"
  
  if [[ ${#running_pids[@]} -gt 0 ]]; then
    echo ""
    echo "Currently running jobs:"
    for pid in "${!running_pids[@]}"; do
      job_name="${running_pids[$pid]}"
      gpu="${job_gpu[$job_name]}"
      elapsed=$(($(date +%s) - ${runtime[$job_name]}))
      printf "  %-25s  GPU: %-2s  PID: %-6s  Running for: %02d:%02d:%02d\n" \
        "${job_name}" "${gpu}" "${pid}" $((elapsed/3600)) $(((elapsed%3600)/60)) $((elapsed%60))
    done
  fi

  if [[ ${#failed[@]} -gt 0 ]]; then
    echo ""
    echo "Failed jobs:"
    for job in "${!failed[@]}"; do
      echo "  - ${job}"
    done
  fi
  echo "=========================="
}

check_dependency() {
  local dep="$1"
  
  # Dependency satisfied if job completed successfully
  if [[ -n "${completed[$dep]:-}" ]]; then
    return 0
  fi
  
  # Dependency failed if job failed
  if [[ -n "${failed[$dep]:-}" ]]; then
    echo "[WARNING] Job '$job_name' depends on failed job '$dep'. Marking as failed."
    return 1
  fi
  
  # Dependency not yet satisfied
  return 2
}

# ---------------------
# Define jobs
# ---------------------  

declare -a jobs=(
  "sequence_embedding::python sequence_embedding.py --csv_file ${BASE_PATH}/personal.csv --learning_rate 1e-4 --mask_prob 0.3 --batch_size 4096 --num_epochs 10 --embedding_file user_embedding-m0.3-lr1e-4.pkl --model_file trained_model-m0.3-lr1e-4.pth --gpu_id 0"
  "static_GCN::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --train_batch_size 512 --seed ${SEED}"
  "static_GAT::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --train_batch_size 512 --model GAT --heads 2 --seed ${SEED}"
  "static_GIN::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --train_batch_size 512 --model GIN --seed ${SEED}"
  "static_personal_GCN::python static_graph_personal_nodes.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --experiment_name PG-GCN-2layer --seed ${SEED}"
  "static_personal_GAT::python static_graph_personal_nodes.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --experiment_name PG-GAT-2layer --model GAT --heads 2 --seed ${SEED}"
  "static_personal_GIN::python static_graph_personal_nodes.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --epoch 5 --eval_every 1 --experiment_name PG-GCN-2layer --model GIN --seed ${SEED}"
  # "temporal_TGN::python temporal_graph.py --base_path ${BASE_PATH}/ --epoch 5 --embedding_dim 16 --train_batch_size 512 --eval_every 1 --experiment_name TGN-exp-1 --seed ${SEED}"
  # "temporal_DyRep::python temporal_graph.py --base_path ${BASE_PATH}/ --epoch 5 --embedding_dim 16 --train_batch_size 512 --eval_every 1 --model DyRep --experiment_name DyRep-exp-1 --seed ${SEED}"
  "static_GCN_emb::DEP=sequence_embedding::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --use_sequence_embedding --seq_embedding_filename stored/user_embedding-m0.3-lr1e-4.pkl --epoch 5 --eval_every 1 --train_batch_size 512 --seed ${SEED}"
  "static_GAT_emb::DEP=sequence_embedding::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --use_sequence_embedding --seq_embedding_filename stored/user_embedding-m0.3-lr1e-4.pkl --epoch 5 --eval_every 1 --train_batch_size 512 --model GAT --heads 2 --seed ${SEED}"
  "static_GIN_emb::DEP=sequence_embedding::python static_graph.py --base_path ${BASE_PATH}/ --gpu_id 0 --lr 1e-3 --use_sequence_embedding --seq_embedding_filename stored/user_embedding-m0.3-lr1e-4.pkl --epoch 5 --eval_every 1 --train_batch_size 512 --model GIN --seed ${SEED}"
)
total_jobs=${#jobs[@]}

# Print summary of execution plan
echo "=============================================================================="
echo "Job Scheduler - ${total_jobs} jobs on ${MAX_GPUS} GPUs"
echo "Base path: ${BASE_PATH}"
echo "Log directory: ${LOG_DIR}"
echo "Stored directory: ${STORED_DIR}"
echo "=============================================================================="

# ---------------------
# Scheduler loop
# ---------------------

available_gpus=()
job_index=0
start_timestamp=$(date +%s)

# Handle SIGINT (Ctrl+C) to print status before exiting
trap 'echo ""; echo "Job scheduler interrupted!"; print_status; exit 1;' SIGINT

while [[ ${job_index} -lt ${#jobs[@]} ]] || [[ ${#running_pids[@]} -gt 0 ]]; do
  # Clean up completed jobs
  for pid in "${!running_pids[@]}"; do
    if ! kill -0 "$pid" 2>/dev/null; then
      job_name="${running_pids[$pid]}"
      wait "$pid"
      exit_code=$?
      
      if [[ ${exit_code} -eq 0 ]]; then
        completed["$job_name"]=1
        echo "[SUCCESS] Job '${job_name}' completed successfully"
      else
        failed["$job_name"]=1
        echo "[FAILED] Job '${job_name}' failed with exit code ${exit_code}"
      fi
      
      unset running_pids["$pid"]
       unset job_gpu["$job_name"]
      end_time=$(date +%s)
      duration=$((end_time - runtime["$job_name"]))
      echo "[INFO] Job '${job_name}' ran for $((duration/60))m $((duration%60))s"
    fi
  done

  # Recalculate available GPUs
  busy_gpus=()
  for job in "${!job_gpu[@]}"; do
    busy_gpus+=("${job_gpu[$job]}")
  done
  available_gpus=()
  for gpu in $(seq 0 $((MAX_GPUS - 1))); do
    if [[ ! " ${busy_gpus[*]} " =~ " ${gpu} " ]]; then
      available_gpus+=("${gpu}")
    fi
  done

  # Assign ready jobs
  while [[ ${#available_gpus[@]} -gt 0 && ${job_index} -lt ${#jobs[@]} ]]; do
    # IFS='::' read -r job_name meta cmd <<< "${jobs[$job_index]}"

    entry="${jobs[$job_index]}"

    # 1) job name is everything before the first "::"
    job_name="${entry%%::*}"
    remainder="${entry#*::}"            # drop "<job_name>::"

    # 2) look for an explicit dependency token "DEP=foo"
    dep=""                              # empty string ➜ no dependency
    if [[ "$remainder" == DEP=* ]]; then
        dep="${remainder%%::*}"         # "DEP=foo"
        dep="${dep#DEP=}"               # just "foo"
        cmd="${remainder#*::}"          # what comes after the 2nd "::"
    else
        cmd="$remainder"                # no dependency, whole remainder is command
    fi


    # Handle job dependencies
    dependency_met=true
    if [[ -n "$dep" ]]; then
        check_result=$(check_dependency "$dep")
        case $check_result in
            0) ;;  # Dependency satisfied
            1)  failed["$job_name"]=1
                echo "[SKIPPED] Job '${job_name}' skipped due to failed dependency '${dep}'"
                ((job_index++))
                continue
                ;;
            2)  dependency_met=false ;;  # Dependency not yet satisfied
        esac
    fi

    if [[ "$dependency_met" = true ]]; then
      gpu="${available_gpus[0]}"
      runtime["$job_name"]=$(date +%s)
      run_job_on_gpu "$job_name" "$gpu" "$cmd"
      available_gpus=("${available_gpus[@]:1}")
      ((job_index++))
    else
      # Move to next job if dependency not met
      break
    fi
  done

  # Print status if in verbose mode or every 5 checks
  if [[ $VERBOSE -eq 1 ]] || [[ $((SECONDS % (CHECK_INTERVAL * 5))) -lt $CHECK_INTERVAL ]]; then
    print_status
  fi

  sleep "$CHECK_INTERVAL"
done

# Final status
print_status

# Print summary
echo ""
echo "=============================================================================="
total_duration=$(($(date +%s) - start_timestamp))
echo "[COMPLETED] Job scheduler finished in $((total_duration/3600))h $(((total_duration%3600)/60))m $((total_duration%60))s"
echo "  Success: ${#completed[@]} jobs"
echo "  Failed:  ${#failed[@]} jobs"

if [[ ${#failed[@]} -gt 0 ]]; then
  echo ""
  echo "[WARNING] The following jobs failed:"
  for job in "${!failed[@]}"; do
    echo "  - ${job}"
  done
  exit_code=1
else
  echo ""
  echo "[SUCCESS] All experiments completed successfully."
  exit_code=0
fi

echo ""
echo "Logs are in   → ${LOG_DIR}/"
echo "Model & embedding artifacts live in → ${STORED_DIR}/"
echo "=============================================================================="

exit $exit_code