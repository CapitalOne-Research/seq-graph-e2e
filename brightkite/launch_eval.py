import subprocess
import time
import os
import argparse
import shlex
import math
from collections import deque
import json
# ---------------- Config ----------------
BASE_COMMAND = "python joint.py"
MAX_GPUS_TO_USE = 8
GPU_FREE_MEM_THRESHOLD = 5000  # MiB considered "free"
POLL_INTERVAL_ACTIVE = 30     # seconds between nvidia-smi polls while queue not empty
POLL_INTERVAL_DRAIN  = 60     # seconds once all jobs launched and we're draining


def check_tested(file_path: str):
    with open(file_path) as source:
        json_data = json.load(source)
    rows = json_data if isinstance(json_data, list) else [json_data]
    if any(isinstance(row, dict) and isinstance(row.get('best_test_mrr'), (int, float))
           and math.isfinite(row['best_test_mrr']) for row in rows):
        print(f"File {file_path} has been tested")
        return True

    return False
# -------------- GPU Helpers --------------
def get_free_gpu_ids(max_gpus=MAX_GPUS_TO_USE, free_mem_thresh=GPU_FREE_MEM_THRESHOLD):
    """
    Return a list of GPU ids whose used memory < threshold (MiB).
    Respects MAX_GPUS_TO_USE cap.
    """
    try:
        smi_query = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(smi_query, shell=True, capture_output=True, text=True, check=True)
        free = []
        for line in result.stdout.strip().splitlines():
            if not line:
                continue
            idx_str, used_str = line.split(',')
            idx = int(idx_str.strip())
            used = int(used_str.strip())
            if used < free_mem_thresh:
                free.append(idx)
        return free[:max_gpus]
    except (subprocess.CalledProcessError, FileNotFoundError):
        # No nvidia-smi or failure -> assume no GPUs
        return []

# -------------- Command build --------------
def build_eval_commands(dataset: str, csv_path: str):
    """
    Read test list CSV and build joint.py eval commands for each run_name.
    Expects column 'run_name'. Produces both _best.pth and .pth candidates.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if 'run_name' not in df.columns:
        raise ValueError(f"'run_name' column not found in {csv_path}")

    # Two checkpoint directories (following your pattern)
    best_ckpts = df['run_name'].apply(
        lambda x: os.path.join(f'./{dataset}_checkpoints_seeded', x + '_best.pth')
    )
    plain_ckpts = df['run_name'].apply(
        lambda x: os.path.join(f'./checkpoints_{dataset}_seeded', x + '.pth')
    )

    commands = []
    # Only include paths that actually exist, to avoid immediate failures
    for ckpt_path in list(best_ckpts) + list(plain_ckpts):
        if os.path.isfile(ckpt_path):
            cmd = f"{BASE_COMMAND} --load_checkpoint_and_eval {ckpt_path}"
            commands.append(cmd)
    return commands

# -------------- Launcher --------------
def launch_queue_on_available_gpus(commands):
    """
    Schedule commands onto available GPUs. If no GPUs are present, runs sequentially on CPU.
    """
    run_queue = deque(commands)
    total = len(run_queue)
    if total == 0:
        print("No commands to run. Exiting.")
        return

    active = {}  # gpu_id -> Popen
    completed = 0

    def reap_finished():
        nonlocal completed
        done = []
        for gid, proc in active.items():
            if proc.poll() is not None:
                print(f"[GPU {gid}] finished with code {proc.returncode}")
                done.append(gid)
                completed += 1
        for gid in done:
            del active[gid]

    # GPU mode?
    gpu_ids = get_free_gpu_ids()
    gpu_mode = len(gpu_ids) > 0
    if not gpu_mode:
        print("No GPUs detected or nvidia-smi unavailable. Running sequentially on CPU.")
        while run_queue:
            cmd = run_queue.popleft()
            print(f"[CPU] Running {completed + 1}/{total}: {cmd}")
            proc = subprocess.run(cmd, shell=True)  # blocking
            print(f"[CPU] Finished with code {proc.returncode}")
            completed += 1
        print(f"All {completed}/{total} jobs completed.")
        return

    print(f"Detected free GPUs: {gpu_ids} (max {MAX_GPUS_TO_USE}).")
    while run_queue or active:
        # Reap finished jobs
        reap_finished()

        # Try to launch onto currently free GPUs (and not already occupied by our procs)
        if run_queue:
            free_now = get_free_gpu_ids()
            # exclude GPUs we are already using
            free_now = [g for g in free_now if g not in active]

            for gid in free_now:
                if not run_queue:
                    break
                cmd = run_queue.popleft()
                print("=" * 60)
                print(f"Launching {completed + len(active) + 1}/{total} on GPU {gid}")
                print("CMD:", cmd)
                print("=" * 60)
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gid)
                proc = subprocess.Popen(cmd, shell=True, env=env)
                active[gid] = proc

        # Sleep between cycles
        time.sleep(POLL_INTERVAL_ACTIVE if run_queue else POLL_INTERVAL_DRAIN)

    print(f"\nAll {completed}/{total} jobs completed.")

# -------------- CLI --------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=['brightkite', 'gowalla'], default='brightkite')
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--checkpoint-dir", help="defaults to <dataset>_checkpoints_seeded")
    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir or f"{args.dataset}_checkpoints_seeded"
    ckpt_paths = [os.path.join(checkpoint_dir, p) for p in sorted(os.listdir(checkpoint_dir))
                  if p.endswith('_best.pth') and 'tokenxat' not in p]
    if args.seed is not None:
        ckpt_paths = [p for p in ckpt_paths if f'seed{args.seed}' in p]
    ckpt_paths = [p for p in ckpt_paths if os.path.isfile(p)]
    commands = []
    for checkpoint in ckpt_paths:
        result = checkpoint.replace('checkpoints', 'results').replace('_best.pth', '.json').replace('.pth', '.json')
        if os.path.isfile(result) and check_tested(result):
            continue
        commands.append(f"{BASE_COMMAND} --dataset {args.dataset} --load_checkpoint_and_eval {shlex.quote(checkpoint)}")
    print(f"Total {len(commands)} commands to run.")
    launch_queue_on_available_gpus(commands)


if __name__ == "__main__":
    main()
