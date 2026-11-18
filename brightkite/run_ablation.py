import subprocess
import argparse
import os
import sys
import csv
import time
import datetime
import platform
from collections import deque
from typing import Dict, List, Optional, Tuple

# --- CONFIGURATION ---
DRY_RUN = False           # Set to False to actually run
SEEDS = [0, 1, 2]         # The 3 seeds you should run

# GPU scheduling: a GPU is considered AVAILABLE only if it has at least this much free memory.
MIN_FREE_MB = 5000

# If you want to cap concurrent jobs (e.g., to avoid CPU/RAM pressure), set this.
# By default, we allow up to 1 job per available GPU.
MAX_JOBS_PER_GPU = 1

PYTHON_EXEC = sys.executable
SCRIPT_PATH = os.path.join(os.path.dirname(__file__), "joint.py")

# Paths
INPUT_PATH = "../data/tasks/brightkite/friend_recommendation_5000"
RESULTS_DIR = "./brightkite_results_ablation_seeded"
CHECKPOINTS_DIR = "./brightkite/brightkite_checkpoints_ablation_seeded"
TIMINGS_CSV = os.path.join(RESULTS_DIR, "timings.csv")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")

TIMINGS_FIELDS = [
    "experiment_id",
    "experiment_name",
    "desc",
    "seed",
    "returncode",
    "duration_sec",
    "start_utc",
    "end_utc",
    "log_path",
    "command",
    "hostname",
    "cuda_visible_devices",
    "gpu_free_mb_at_start",
]

# Base Config
BASE_CONFIG = {
    "dataset": "brightkite",
    "debug_subset": "5000",
    "num_neighbors": "10",
    "compress_dim": "2",
    "compress_first": False,
    "gnn_layer_types": "TokenXAT,GCN",
    "bert_num_layers": "2",
    "bert_hidden_size": "128",
    "num_epochs": "100",
}

# --- EXPERIMENTS ---
experiments = [
    # Baseline
    {"id": "EXP-000", "desc": "Baseline", "params": {}},

    # Neighbor Sensitivity
    {"id": "EXP-101", "desc": "Neighbors=2",  "params": {"num_neighbors": "2"}},
    {"id": "EXP-102", "desc": "Neighbors=5",  "params": {"num_neighbors": "5"}},
    {"id": "EXP-103", "desc": "Neighbors=15", "params": {"num_neighbors": "15"}},
    {"id": "EXP-104", "desc": "Neighbors=20", "params": {"num_neighbors": "20"}},

    # Compression Length
    {"id": "EXP-201", "desc": "CompLength=1",  "params": {"compress_dim": "1"}},
    {"id": "EXP-202", "desc": "CompLength=2",  "params": {"compress_dim": "2"}},
    {"id": "EXP-203", "desc": "CompLength=8",  "params": {"compress_dim": "8"}},
    {"id": "EXP-204", "desc": "CompLength=16", "params": {"compress_dim": "16"}},
    {"id": "EXP-205", "desc": "Compress Before", "params": {"compress_first": True}},

    # TF Depth
    {"id": "EXP-301", "desc": "TF_Layers=1", "params": {"bert_num_layers": "1"}},
    {"id": "EXP-302", "desc": "TF_Layers=4", "params": {"bert_num_layers": "4"}},

    # Architecture Variants
    {"id": "EXP-305", "desc": "GCN-GCN (Bridge)", "params": {"gnn_layer_types": "GCN,GCN"}},
    {"id": "EXP-306", "desc": "Token-Token-Token-GCN", "params": {"gnn_layer_types": "TokenXAT,TokenXAT,TokenXAT,GCN"}},
    {"id": "EXP-309", "desc": "Deep GCN Stack", "params": {"gnn_layer_types": "GCN,GCN,GCN,GCN"}},

    # Single Layer
    {"id": "EXP-307", "desc": "1-Layer TokenXATTN", "params": {"gnn_layer_types": "TokenXAT"}},
    {"id": "EXP-308", "desc": "1-Layer GCN", "params": {"gnn_layer_types": "GCN"}},
]

def _append_timing_row(row: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    file_exists = os.path.exists(TIMINGS_CSV)
    with open(TIMINGS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TIMINGS_FIELDS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

def _has_nvidia_smi() -> bool:
    try:
        subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.STDOUT, text=True)
        return True
    except Exception:
        return False

def get_gpu_free_mb() -> Dict[int, int]:
    """
    Returns {gpu_index: free_mb}. Requires nvidia-smi.
    """
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        stderr=subprocess.STDOUT,
        text=True,
    )
    free: Dict[int, int] = {}
    for line in out.strip().splitlines():
        if not line.strip():
            continue
        idx_str, free_str = [x.strip() for x in line.split(",")]
        free[int(idx_str)] = int(free_str)
    return free

def build_command(exp: dict, seed: int) -> Tuple[str, List[str]]:
    exp_id_seeded = f"{exp['id']}_s{seed}"
    config = BASE_CONFIG.copy()
    config.update(exp["params"])

    # Auto-calc layers
    if "gnn_num_layers" not in config:
        config["gnn_num_layers"] = str(len(config["gnn_layer_types"].split(",")))

    cmd = [
        PYTHON_EXEC, SCRIPT_PATH,
        "--input_path", INPUT_PATH,
        "--results_dir", RESULTS_DIR,
        "--checkpoints_dir", CHECKPOINTS_DIR,
        "--experiment_name", exp_id_seeded,
        "--seed", str(seed),
    ]

    for key, val in config.items():
        if val is True:
            cmd.append(f"--{key}")
        elif val is False:
            # skip flags for False
            pass
        else:
            cmd.append(f"--{key}")
            cmd.append(str(val))

    full_command = " ".join(cmd)
    return exp_id_seeded, cmd

class RunningJob:
    def __init__(
        self,
        exp: dict,
        seed: int,
        gpu_id: int,
        proc: subprocess.Popen,
        log_path: str,
        start_ts: str,
        t0: float,
        full_command: str,
        gpu_free_mb_at_start: int,
    ):
        self.exp = exp
        self.seed = seed
        self.gpu_id = gpu_id
        self.proc = proc
        self.log_path = log_path
        self.start_ts = start_ts
        self.t0 = t0
        self.full_command = full_command
        self.gpu_free_mb_at_start = gpu_free_mb_at_start

def launch_job(exp: dict, seed: int, gpu_id: int, gpu_free_mb: int) -> Optional[RunningJob]:
    exp_id_seeded, cmd = build_command(exp, seed)
    full_command = " ".join(cmd)

    if DRY_RUN:
        print(f"[DRY RUN] Launch on GPU {gpu_id} (free={gpu_free_mb}MB): {exp_id_seeded}: {full_command}")
        return None

    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{exp_id_seeded}.log")
    start_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
    t0 = time.perf_counter()

    env = os.environ.copy()
    # Pin this process to a single GPU
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    log_f = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    print(f">>> RUNNING {exp_id_seeded} on GPU {gpu_id} (free={gpu_free_mb}MB) <<<")
    return RunningJob(
        exp=exp,
        seed=seed,
        gpu_id=gpu_id,
        proc=proc,
        log_path=log_path,
        start_ts=start_ts,
        t0=t0,
        full_command=full_command,
        gpu_free_mb_at_start=gpu_free_mb,
    )

def finalize_job(job: RunningJob) -> None:
    returncode = job.proc.poll()
    if returncode is None:
        return  # not done

    t1 = time.perf_counter()
    end_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()

    exp_id_seeded = f"{job.exp['id']}_s{job.seed}"

    _append_timing_row(
        {
            "experiment_id": job.exp["id"],
            "experiment_name": exp_id_seeded,
            "desc": job.exp["desc"],
            "seed": job.seed,
            "returncode": returncode,
            "duration_sec": f"{t1 - job.t0:.3f}",
            "start_utc": job.start_ts,
            "end_utc": end_ts,
            "log_path": job.log_path,
            "command": job.full_command,
            "hostname": platform.node(),
            "cuda_visible_devices": str(job.gpu_id),
            "gpu_free_mb_at_start": str(job.gpu_free_mb_at_start),
        }
    )

    if returncode != 0:
        print(f"FAILED {exp_id_seeded} on GPU {job.gpu_id}: returncode={returncode} (log: {job.log_path})")
    else:
        print(f"DONE   {exp_id_seeded} on GPU {job.gpu_id} (log: {job.log_path})")

def choose_available_gpu(
    gpu_free: Dict[int, int],
    running_counts: Dict[int, int],
) -> Optional[Tuple[int, int]]:
    """
    Return (gpu_id, free_mb) for an available GPU, else None.
    Availability: free_mb >= MIN_FREE_MB and running_counts[gpu_id] < MAX_JOBS_PER_GPU.
    Chooses the GPU with the most free memory.
    """
    candidates = []
    for gid, free_mb in gpu_free.items():
        if free_mb >= MIN_FREE_MB and running_counts.get(gid, 0) < MAX_JOBS_PER_GPU:
            candidates.append((gid, free_mb))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[1], reverse=True)  # most free MB first
    return candidates[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the seeded ablation grid with GPU scheduling.")
    parser.add_argument("--dry-run", action="store_true", help="print commands without starting jobs or creating outputs")
    args = parser.parse_args()
    DRY_RUN = args.dry_run or DRY_RUN
    if DRY_RUN:
        print("!!! DRY RUN MODE: Printing launch plan only !!!")
        for exp in experiments:
            for seed in SEEDS:
                name, command = build_command(exp, seed)
                print(f"[DRY RUN] {name}: {' '.join(command)}")
        sys.exit(0)

    # Build job queue: (exp, seed)
    job_queue = deque()
    for exp in experiments:
        for seed in SEEDS:
            job_queue.append((exp, seed))

    # If nvidia-smi isn't available, fall back to sequential execution.
    if not _has_nvidia_smi():
        print("WARNING: nvidia-smi not found. Falling back to sequential execution.")
        for (exp, seed) in job_queue:
            # sequential, no GPU pinning
            exp_id_seeded, cmd = build_command(exp, seed)
            full_command = " ".join(cmd)
            if DRY_RUN:
                print(f"[DRY RUN] {exp_id_seeded}: {full_command}")
                continue
            os.makedirs(LOGS_DIR, exist_ok=True)
            log_path = os.path.join(LOGS_DIR, f"{exp_id_seeded}.log")
            start_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            t0 = time.perf_counter()
            with open(log_path, "w", encoding="utf-8") as log_f:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)
                returncode = proc.wait()
            t1 = time.perf_counter()
            end_ts = datetime.datetime.now(datetime.timezone.utc).isoformat()
            _append_timing_row(
                {
                    "experiment_id": exp["id"],
                    "experiment_name": exp_id_seeded,
                    "desc": exp["desc"],
                    "seed": seed,
                    "returncode": returncode,
                    "duration_sec": f"{t1 - t0:.3f}",
                    "start_utc": start_ts,
                    "end_utc": end_ts,
                    "log_path": log_path,
                    "command": full_command,
                    "hostname": platform.node(),
                    "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES", ""),
                    "gpu_free_mb_at_start": "",
                }
            )
        sys.exit(0)

    # Parallel scheduler
    running: List[RunningJob] = []
    running_counts: Dict[int, int] = {}

    # Main loop: launch when GPUs are free; reap finished jobs
    while job_queue or running:
        # Reap finished
        still_running: List[RunningJob] = []
        for job in running:
            if job.proc.poll() is not None:
                finalize_job(job)
                running_counts[job.gpu_id] = max(0, running_counts.get(job.gpu_id, 1) - 1)
            else:
                still_running.append(job)
        running = still_running

        # Try launching as many jobs as possible
        launched_any = False
        if job_queue:
            gpu_free = get_gpu_free_mb()
            while job_queue:
                choice = choose_available_gpu(gpu_free, running_counts)
                if choice is None:
                    break
                gpu_id, free_mb = choice
                exp, seed = job_queue.popleft()
                job = launch_job(exp, seed, gpu_id=gpu_id, gpu_free_mb=free_mb)
                launched_any = True
                if job is not None:
                    running.append(job)
                    running_counts[gpu_id] = running_counts.get(gpu_id, 0) + 1
                # refresh GPU free memory snapshot for the next decision (cheap but safer)
                if not DRY_RUN:
                    gpu_free = get_gpu_free_mb()

        # Avoid busy-waiting
        if not launched_any:
            time.sleep(5)
