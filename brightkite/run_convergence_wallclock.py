import argparse
import csv
import glob
import json
import os
import re
import signal
import subprocess
import sys
import time
import threading
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple



_PRINT_LOCK = threading.Lock()

def _print_line(prefix: str, line: str):
    # Prefix each line so interleaved output remains readable.
    with _PRINT_LOCK:
        sys.stdout.write(f"[{prefix}] {line}")
        sys.stdout.flush()

@dataclass(frozen=True)
class MethodConfig:
    method_id: str
    gnn_layer_types: str


@dataclass(frozen=True)
class StaticConfig:
    method_id: str
    model: str
    num_layers: int = 2
    heads: int = 2


@dataclass(frozen=True)
class TemporalConfig:
    method_id: str
    model: str


@dataclass(frozen=True)
class Job:
    kind: str  # "joint" | "static" | "temporal"
    method_id: str
    seed: int
    gnn_layer_types: str = ""
    static_model: str = ""
    static_num_layers: int = 0
    static_heads: int = 0
    temporal_model: str = ""


DEFAULT_METHODS = [
    MethodConfig(method_id="gcn-gcn", gnn_layer_types="GCN,GCN"),
    MethodConfig(method_id="transformerconv-graphsage", gnn_layer_types="TransformerConv,GraphSAGE"),
]

DEFAULT_STATIC = [
    StaticConfig(method_id="static-gcn", model="GCN"),
]

DEFAULT_TEMPORAL = [
    TemporalConfig(method_id="temporal-nat", model="NAT"),
]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_single_results_json(results_dir: str) -> Optional[str]:
    paths = sorted(glob.glob(os.path.join(results_dir, "*.json")))
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def _extract_val_losses(payload: list) -> List[Tuple[int, float]]:
    out = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "epoch" not in item:
            continue
        metrics = item.get("evaluation_metrics")
        if not isinstance(metrics, dict):
            continue
        if "loss" not in metrics:
            continue
        try:
            epoch = int(item["epoch"])
            val_loss = float(metrics["loss"])
        except (TypeError, ValueError):
            continue
        out.append((epoch, val_loss))
    return sorted(out, key=lambda x: x[0])


def _convergence_epoch_by_plateau(
    val_losses: List[Tuple[int, float]],
    patience_evals: int,
    min_delta: float,
) -> Tuple[Optional[int], Optional[int], Optional[float], int]:
    if not val_losses:
        return None, None, None, 0

    best_epoch, best_loss = val_losses[0]
    no_improve = 0
    for epoch, loss in val_losses[1:]:
        if loss < best_loss - min_delta:
            best_loss = loss
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience_evals:
                return epoch, best_epoch, best_loss, no_improve
    return None, best_epoch, best_loss, no_improve


def _sum_wallclock_to_epoch(payload: list, end_epoch: int, include_epoch0: bool) -> float:
    total = 0.0
    for item in payload:
        if not isinstance(item, dict):
            continue
        if "epoch" not in item:
            continue
        try:
            epoch = int(item["epoch"])
        except (TypeError, ValueError):
            continue
        if (not include_epoch0) and epoch == 0:
            continue
        if epoch > end_epoch:
            continue
        try:
            epoch_time = float(item.get("epoch_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            epoch_time = 0.0
        try:
            eval_time = float(item.get("eval_time", 0.0) or 0.0)
        except (TypeError, ValueError):
            eval_time = 0.0
        total += epoch_time + eval_time
    return total


def _append_row(csv_path: str, fieldnames: List[str], row: Dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _has_nvidia_smi() -> bool:
    try:
        subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True, timeout=10)
        return True
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return False


def _parse_cuda_visible_devices_as_ints() -> Optional[List[int]]:
    """
    If CUDA_VISIBLE_DEVICES is set to a comma-separated list of integers (physical indices),
    return those ints. If unset, return None. If set but non-integer format (e.g., UUID/MIG),
    return None (we won't try to interpret).
    """
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None:
        return None
    raw = raw.strip()
    if raw == "":
        return []
    out: List[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except ValueError:
            return None
    return out


def _query_gpu_free_memory_mb_by_index() -> Optional[Dict[int, int]]:
    """
    Best-effort query via nvidia-smi.
    Returns {physical_gpu_index: free_memory_mb} or None if unavailable.
    """
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        return None

    out: Dict[int, int] = {}
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            free = int(float(parts[1]))
        except ValueError:
            continue
        out[idx] = free
    return out


def _auto_detect_physical_gpu_ids(fallback_gpu_id: int) -> List[int]:
    """
    Auto-detect physical GPU indices.
    Priority:
      1) If CUDA_VISIBLE_DEVICES is a list of ints, use it directly.
      2) Else query nvidia-smi for GPU indices.
      3) Else fallback to [fallback_gpu_id].
    """
    visible_ints = _parse_cuda_visible_devices_as_ints()
    if visible_ints is not None:
        if not visible_ints:
            raise SystemExit("CUDA_VISIBLE_DEVICES is set but empty; no GPUs are visible.")
        return visible_ints

    mem_free = _query_gpu_free_memory_mb_by_index()
    if mem_free is not None and mem_free:
        return sorted(mem_free.keys())

    # nvidia-smi unavailable or no GPUs found
    return [fallback_gpu_id]


def _choose_available_gpu_physical(
    candidate_physical_ids: List[int],
    running_counts: Dict[int, int],
    max_jobs_per_gpu: int,
    gpu_free_threshold_mb: int,
    strict_free_check: bool,
) -> Optional[Tuple[int, Optional[int]]]:
    """
    Return (physical_gpu_id, free_mb) for an available GPU, else None.

    If strict_free_check is False (default):
      - schedule on any GPU that has an available slot (running_counts < max_jobs_per_gpu)
      - use memory.free only as a preference/tiebreaker (pick MOST free)

    If strict_free_check is True:
      - require memory.free >= gpu_free_threshold_mb when nvidia-smi is available
    """
    mem_free = _query_gpu_free_memory_mb_by_index()

    candidates: List[Tuple[int, Optional[int]]] = []
    for physical_id in candidate_physical_ids:
        if running_counts.get(physical_id, 0) >= max_jobs_per_gpu:
            continue

        if mem_free is None:
            candidates.append((physical_id, None))
            continue

        free = mem_free.get(physical_id)
        if free is None:
            if not strict_free_check:
                candidates.append((physical_id, None))
            continue

        if strict_free_check:
            if free >= gpu_free_threshold_mb:
                candidates.append((physical_id, free))
        else:
            candidates.append((physical_id, free))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[1] is None, -(x[1] if x[1] is not None else 0), x[0]))
    return candidates[0]


def _popen_pinned(cmd: List[str], cwd: str, physical_gpu_id: int):
    """
    Run-ablation practice:
      - Pin by setting CUDA_VISIBLE_DEVICES to a SINGLE physical GPU index.
      - In that environment, the child should use cuda:0, so we must NOT pass a nonzero gpu_id.
    """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu_id)
    return subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )


def run_one(
    method: MethodConfig,
    seed: int,
    out_root: str,
    python_exec: str,
    physical_gpu_id: int,
    input_path: Optional[str],
    max_epochs: int,
    eval_every: int,
    patience_evals: int,
    min_delta: float, 
    poll_sec: float,
    include_epoch0: bool,
    stop_on_plateau: bool,
    print_prefix: str,
    extra_args: List[str],
) -> Dict:
    run_dir = os.path.join(out_root, method.method_id, f"seed{seed}")
    results_dir = os.path.join(run_dir, "results")
    checkpoints_dir = os.path.join(run_dir, "checkpoints")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "stdout.log")

    cmd = [
        python_exec,
        "joint.py",
        "--dataset",
        "brightkite",
        *(
            ["--input_path", input_path]
            if input_path
            else []
        ),
        "--results_dir",
        results_dir,
        "--checkpoints_dir",
        checkpoints_dir,
        "--experiment_name",
        f"conv_{method.method_id}_seed{seed}",
        "--seed",
        str(seed),
        "--gpu_id",
        "0",  # IMPORTANT: pinned env => single visible GPU => cuda:0
        "--num_epochs",
        str(max_epochs),
        "--eval_every",
        str(eval_every),
        "--gnn_layer_types",
        method.gnn_layer_types,
    ] + extra_args

    start_wall = time.time()
    proc = _popen_pinned(cmd, cwd=os.path.dirname(__file__), physical_gpu_id=physical_gpu_id)

    results_json = None
    last_converged_epoch = None
    last_seen_eval_epoch = None
    best_epoch = None
    best_loss = None
    no_improve_evals = 0
    last_json_mtime = None
    last_check_time = 0.0

    with open(log_path, "w", encoding="utf-8") as log_f:
        assert proc.stdout is not None
        for line in proc.stdout:
            _print_line(print_prefix, line)
            log_f.write(line)

            if results_json is None:
                results_json = _find_single_results_json(results_dir)

            if not stop_on_plateau:
                continue

            now = time.time()
            if results_json is not None and (now - last_check_time) >= poll_sec:
                last_check_time = now
                try:
                    mtime = os.path.getmtime(results_json)
                except OSError:
                    continue
                if last_json_mtime is not None and mtime == last_json_mtime:
                    continue
                last_json_mtime = mtime

                try:
                    payload = _load_json(results_json)
                except Exception:
                    continue
                val_losses = _extract_val_losses(payload)
                if val_losses:
                    current_eval_epoch = val_losses[-1][0]
                    if last_seen_eval_epoch != current_eval_epoch:
                        last_seen_eval_epoch = current_eval_epoch
                        converged_epoch, best_epoch, best_loss, no_improve_evals = _convergence_epoch_by_plateau(
                            val_losses, patience_evals=patience_evals, min_delta=min_delta
                        )
                        if converged_epoch is not None:
                            last_converged_epoch = converged_epoch
                            proc.send_signal(signal.SIGINT)
                            break

    try:
        returncode = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        returncode = proc.wait(timeout=60)

    end_wall = time.time()
    wallclock_sec = end_wall - start_wall

    if results_json is None:
        results_json = _find_single_results_json(results_dir)
    payload = _load_json(results_json) if (stop_on_plateau and results_json) else []
    if stop_on_plateau:
        val_losses = _extract_val_losses(payload)
        converged_epoch, best_epoch2, best_loss2, _ = _convergence_epoch_by_plateau(
            val_losses, patience_evals=patience_evals, min_delta=min_delta
        )
        if best_epoch is None:
            best_epoch = best_epoch2
        if best_loss is None:
            best_loss = best_loss2
        if last_converged_epoch is None:
            last_converged_epoch = converged_epoch
        if last_converged_epoch is None and val_losses:
            last_converged_epoch = val_losses[-1][0]

    train_time_to_conv = None
    if stop_on_plateau and last_converged_epoch is not None:
        train_time_to_conv = _sum_wallclock_to_epoch(payload, end_epoch=last_converged_epoch, include_epoch0=include_epoch0)

    return {
        "family": "joint",
        "method_id": method.method_id,
        "gnn_layer_types": method.gnn_layer_types,
        "seed": seed,
        "returncode": returncode,
        "results_json": results_json or "",
        "stdout_log": log_path,
        "best_val_loss": best_loss if best_loss is not None else "",
        "best_epoch": best_epoch if best_epoch is not None else "",
        "converged_epoch": last_converged_epoch if last_converged_epoch is not None else "",
        "train_time_to_converge_sec": f"{train_time_to_conv:.3f}" if train_time_to_conv is not None else "",
        "wallclock_sec": f"{wallclock_sec:.3f}",
        "patience_evals": patience_evals,
        "min_delta": min_delta,
        "eval_every": eval_every,
        "max_epochs": max_epochs,
        "gpu_id": physical_gpu_id,
    }


_STATIC_EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\s+Loss:.*?\bepoch_time:\s*([0-9]*\.?[0-9]+)s\b")
_STATIC_EVAL_LINE = re.compile(
    r"Epoch\s+(\d+)\s+Evaluation.*?\bVal\s+Loss:\s*([0-9]*\.?[0-9]+)\b.*?\bVal\s+MRR:\s*([0-9]*\.?[0-9]+)\b.*?\beval_time:\s*([0-9]*\.?[0-9]+)s\b"
)

_TEMPORAL_EPOCH_LINE = re.compile(r"Epoch\s+(\d+)\b.*?\bepoch_time:\s*([0-9]*\.?[0-9]+)s\b")
_TEMPORAL_EVAL_LINE = re.compile(
    r"Epoch\s+(\d+)\s+Evaluation.*?\bVal\s+MRR:\s*([0-9]*\.?[0-9]+)\b.*?\bepoch_time:\s*([0-9]*\.?[0-9]+)s\b"
)


def run_one_static(
    cfg: StaticConfig,
    seed: int,
    out_root: str,
    python_exec: str,
    physical_gpu_id: int,
    epochs: int,
    eval_every: int,
    patience_evals: int,
    min_delta: float,
    include_epoch0: bool,
    stop_on_plateau: bool,
    print_prefix: str,
    extra_args: List[str],
) -> Dict:
    run_dir = os.path.join(out_root, cfg.method_id, f"seed{seed}")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "stdout.log")

    cmd = [
        python_exec,
        "static_graph.py",
        "--dataset",
        "brightkite",
        "--model",
        cfg.model,
        "--num_layers",
        str(cfg.num_layers),
        "--heads",
        str(cfg.heads),
        "--epochs",
        str(epochs),
        "--eval_every",
        str(eval_every),
        "--seed",
        str(seed),
        "--gpu_id",
        "0",  # pinned => cuda:0
    ] + extra_args

    start_wall = time.time()
    proc = _popen_pinned(cmd, cwd=os.path.dirname(__file__), physical_gpu_id=physical_gpu_id)

    epoch_time_by_epoch: Dict[int, float] = {}
    eval_time_by_epoch: Dict[int, float] = {}
    val_loss_by_epoch: List[Tuple[int, float]] = []
    val_mrr_by_epoch: List[Tuple[int, float]] = []

    best_loss = None
    best_epoch = None
    best_mrr = None
    no_improve = 0
    converged_epoch = None

    with open(log_path, "w", encoding="utf-8") as log_f:
        assert proc.stdout is not None
        for line in proc.stdout:
            _print_line(print_prefix, line)
            log_f.write(line)

            m = _STATIC_EPOCH_LINE.search(line)
            if m:
                try:
                    epoch_time_by_epoch[int(m.group(1))] = float(m.group(2))
                except ValueError:
                    pass

            m2 = _STATIC_EVAL_LINE.search(line)
            if m2:
                try:
                    epoch = int(m2.group(1))
                    val_loss = float(m2.group(2))
                    val_mrr = float(m2.group(3))
                    eval_time = float(m2.group(4))
                except ValueError:
                    continue
                eval_time_by_epoch[epoch] = eval_time
                val_loss_by_epoch.append((epoch, val_loss))
                val_mrr_by_epoch.append((epoch, val_mrr))

                if not stop_on_plateau:
                    continue

                if best_loss is None or val_loss < best_loss - min_delta:
                    best_loss = val_loss
                    best_epoch = epoch
                    best_mrr = val_mrr
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience_evals:
                        converged_epoch = epoch
                        proc.send_signal(signal.SIGINT)
                        break

    try:
        returncode = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        returncode = proc.wait(timeout=60)

    end_wall = time.time()
    wallclock_sec = end_wall - start_wall

    if stop_on_plateau:
        if converged_epoch is None and val_loss_by_epoch:
            converged_epoch = val_loss_by_epoch[-1][0]
        if best_mrr is None and val_mrr_by_epoch:
            if best_epoch is not None:
                for e, mrr in val_mrr_by_epoch:
                    if e == best_epoch:
                        best_mrr = mrr
                        break
            if best_mrr is None:
                best_mrr = max((mrr for _, mrr in val_mrr_by_epoch), default=None)

    train_time_to_converge = None
    if stop_on_plateau and converged_epoch is not None:
        total = 0.0
        start_epoch = 0 if include_epoch0 else 1
        for e in range(start_epoch, converged_epoch + 1):
            total += float(epoch_time_by_epoch.get(e, 0.0))
            total += float(eval_time_by_epoch.get(e, 0.0))
        train_time_to_converge = total

    return {
        "family": "static_graph",
        "method_id": cfg.method_id,
        "gnn_layer_types": "",
        "seed": seed,
        "returncode": returncode,
        "results_json": "",
        "stdout_log": log_path,
        "best_val_loss": best_loss if best_loss is not None else "",
        "best_val_mrr": best_mrr if best_mrr is not None else "",
        "best_epoch": best_epoch if best_epoch is not None else "",
        "converged_epoch": converged_epoch if converged_epoch is not None else "",
        "train_time_to_converge_sec": f"{train_time_to_converge:.3f}" if train_time_to_converge is not None else "",
        "wallclock_sec": f"{wallclock_sec:.3f}",
        "patience_evals": patience_evals,
        "min_delta": min_delta,
        "eval_every": eval_every,
        "max_epochs": epochs,
        "gpu_id": physical_gpu_id,
    }


def run_one_temporal(
    cfg: TemporalConfig,
    seed: int,
    out_root: str,
    python_exec: str,
    physical_gpu_id: int,
    epochs: int,
    eval_every: int,
    patience_evals: int,
    min_delta: float,
    include_epoch0: bool,
    stop_on_plateau: bool,
    print_prefix: str,
    extra_args: List[str],
) -> Dict:
    run_dir = os.path.join(out_root, cfg.method_id, f"seed{seed}")
    logs_dir = os.path.join(run_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "stdout.log")

    cmd = [
        python_exec,
        "temporal_graph_new.py",
        "--dataset",
        "brightkite",
        "--model",
        cfg.model,
        "--epochs",
        str(epochs),
        "--eval_every",
        str(eval_every),
        "--seed",
        str(seed),
        "--gpu_id",
        "0",  # pinned => cuda:0
        "--experiment_name",
        f"conv_{cfg.method_id}_seed{seed}",
    ] + extra_args

    start_wall = time.time()
    proc = _popen_pinned(cmd, cwd=os.path.dirname(__file__), physical_gpu_id=physical_gpu_id)

    epoch_time_by_epoch: Dict[int, float] = {}
    val_mrr_by_epoch: List[Tuple[int, float]] = []

    best_mrr = None
    best_epoch = None
    no_improve = 0
    converged_epoch = None

    with open(log_path, "w", encoding="utf-8") as log_f:
        assert proc.stdout is not None
        for line in proc.stdout:
            _print_line(print_prefix, line)
            log_f.write(line)

            m = _TEMPORAL_EPOCH_LINE.search(line)
            if m:
                try:
                    epoch_time_by_epoch[int(m.group(1))] = float(m.group(2))
                except ValueError:
                    pass

            m2 = _TEMPORAL_EVAL_LINE.search(line)
            if m2:
                try:
                    epoch = int(m2.group(1))
                    val_mrr = float(m2.group(2))
                except ValueError:
                    continue
                val_mrr_by_epoch.append((epoch, val_mrr))

                if not stop_on_plateau:
                    continue

                if best_mrr is None or val_mrr > best_mrr + min_delta:
                    best_mrr = val_mrr
                    best_epoch = epoch
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience_evals:
                        converged_epoch = epoch
                        proc.send_signal(signal.SIGINT)
                        break

    try:
        returncode = proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        proc.terminate()
        returncode = proc.wait(timeout=60)

    end_wall = time.time()
    wallclock_sec = end_wall - start_wall

    if stop_on_plateau:
        if converged_epoch is None and val_mrr_by_epoch:
            converged_epoch = val_mrr_by_epoch[-1][0]
        if best_mrr is None and val_mrr_by_epoch:
            best_epoch, best_mrr = max(val_mrr_by_epoch, key=lambda x: x[1])

    train_time_to_converge = None
    if stop_on_plateau and converged_epoch is not None:
        total = 0.0
        start_epoch = 0 if include_epoch0 else 1
        for e in range(start_epoch, converged_epoch + 1):
            total += float(epoch_time_by_epoch.get(e, 0.0))
        train_time_to_converge = total

    return {
        "family": "temporal_graph",
        "method_id": cfg.method_id,
        "gnn_layer_types": "",
        "seed": seed,
        "returncode": returncode,
        "results_json": "",
        "stdout_log": log_path,
        "best_val_loss": "",
        "best_val_mrr": best_mrr if best_mrr is not None else "",
        "best_epoch": best_epoch if best_epoch is not None else "",
        "converged_epoch": converged_epoch if converged_epoch is not None else "",
        "train_time_to_converge_sec": f"{train_time_to_converge:.3f}" if train_time_to_converge is not None else "",
        "wallclock_sec": f"{wallclock_sec:.3f}",
        "patience_evals": patience_evals,
        "min_delta": min_delta,
        "eval_every": eval_every,
        "max_epochs": epochs,
        "gpu_id": physical_gpu_id,
    }


def main():
    parser = argparse.ArgumentParser(description="Run convergence-time experiments (wall-clock) with GPU scheduling.")

    parser.add_argument("--input_path", default=None)
    parser.add_argument("--out_root", default="./brightkite_joint_convergence_full")
    parser.add_argument("--python", dest="python_exec", default=sys.executable)
    parser.add_argument("--gpu_id", type=int, default=0, help="Fallback GPU id if auto-detect fails.")

    parser.add_argument("--max_jobs_per_gpu", type=int, default=1)
    parser.add_argument("--gpu_free_threshold_mb", type=int, default=5000)
    parser.add_argument("--strict_free_check", action="store_true", help="Strictly require memory.free >= threshold.")
    parser.add_argument("--scheduler_poll_sec", type=float, default=10.0)

    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--method_ids", type=str, default="gcn-gcn")

    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--patience_evals", type=int, default=3)
    parser.add_argument("--min_delta", type=float, default=0.0)
    parser.add_argument("--include_epoch0", action="store_true")
    parser.add_argument("--poll_sec", type=float, default=5.0)

    parser.add_argument("--run_joint", dest="run_joint", action="store_true", default=True)
    parser.add_argument("--no_run_joint", dest="run_joint", action="store_false")

    parser.add_argument("--run_static", dest="run_static", action="store_true", default=True)
    parser.add_argument("--no_run_static", dest="run_static", action="store_false")

    parser.add_argument("--run_temporal", dest="run_temporal", action="store_true", default=True)
    parser.add_argument("--no_run_temporal", dest="run_temporal", action="store_false")

    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_stop_on_plateau", action="store_true")

    parser.add_argument("--extra_args", type=str, default="")
    parser.add_argument("--static_extra_args", type=str, default="")
    parser.add_argument("--temporal_extra_args", type=str, default="")

    args = parser.parse_args()
    stop_on_plateau = not args.no_stop_on_plateau

    candidate_gpus = _auto_detect_physical_gpu_ids(args.gpu_id)
    if not candidate_gpus:
        raise SystemExit("No GPUs detected.")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    method_ids = [s.strip() for s in args.method_ids.split(",") if s.strip()] if args.method_ids.strip() else []
    extra_args = args.extra_args.split() if args.extra_args.strip() else []
    static_extra_args = args.static_extra_args.split() if args.static_extra_args.strip() else []
    temporal_extra_args = args.temporal_extra_args.split() if args.temporal_extra_args.strip() else []

    methods = DEFAULT_METHODS
    if method_ids:
        wanted = set(method_ids)
        methods = [m for m in DEFAULT_METHODS if m.method_id in wanted]
        missing = sorted(wanted - {m.method_id for m in DEFAULT_METHODS})
        if missing:
            raise SystemExit(f"Unknown --method_ids: {','.join(missing)}")
        if not methods:
            raise SystemExit("No methods selected.")

    jobs: List[Job] = []
    if args.run_joint:
        for method in methods:
            for seed in seeds:
                jobs.append(Job(kind="joint", method_id=method.method_id, seed=seed, gnn_layer_types=method.gnn_layer_types))

    if args.run_static:
        for cfg in DEFAULT_STATIC:
            for seed in seeds:
                jobs.append(
                    Job(
                        kind="static",
                        method_id=cfg.method_id,
                        seed=seed,
                        static_model=cfg.model,
                        static_num_layers=cfg.num_layers,
                        static_heads=cfg.heads,
                    )
                )

    if args.run_temporal:
        for cfg in DEFAULT_TEMPORAL:
            for seed in seeds:
                jobs.append(Job(kind="temporal", method_id=cfg.method_id, seed=seed, temporal_model=cfg.model))

    if args.dry_run:
        print(
            "Dry-run only (no execution). "
            f"Auto-detected GPUs={candidate_gpus}. "
            f"(strict_free_check={args.strict_free_check}, free_threshold={args.gpu_free_threshold_mb}MB, max_jobs_per_gpu={args.max_jobs_per_gpu})\n"
        )
        for i, job in enumerate(jobs, start=1):
            physical_gpu_id = candidate_gpus[(i - 1) % len(candidate_gpus)]
            print(f"[{i}] {job.kind}:{job.method_id} seed={job.seed} -> physical_gpu={physical_gpu_id}")
        return

    summary_csv = os.path.join(args.out_root, "summary.csv")
    fields = [
        "family",
        "method_id",
        "gnn_layer_types",
        "seed",
        "returncode",
        "best_val_loss",
        "best_val_mrr",
        "best_epoch",
        "converged_epoch",
        "train_time_to_converge_sec",
        "wallclock_sec",
        "eval_every",
        "patience_evals",
        "min_delta",
        "max_epochs",
        "gpu_id",
        "results_json",
        "stdout_log",
    ]
    append_lock = threading.Lock()

    def _run_job(job: Job, physical_gpu_id: int) -> Dict:
        prefix = f"gpu{physical_gpu_id}|{job.kind}|{job.method_id}|s{job.seed}"
        if job.kind == "joint":
            method = MethodConfig(method_id=job.method_id, gnn_layer_types=job.gnn_layer_types)
            return run_one(
                method=method,
                seed=job.seed,
                out_root=args.out_root,
                python_exec=args.python_exec,
                physical_gpu_id=physical_gpu_id,
                input_path=args.input_path,
                max_epochs=args.max_epochs,
                eval_every=args.eval_every,
                patience_evals=args.patience_evals,
                min_delta=args.min_delta,
                poll_sec=args.poll_sec,
                include_epoch0=args.include_epoch0,
                stop_on_plateau=stop_on_plateau,
                extra_args=extra_args,
                print_prefix=prefix,
            )
        if job.kind == "static":
            cfg = StaticConfig(
                method_id=job.method_id,
                model=job.static_model,
                num_layers=job.static_num_layers,
                heads=job.static_heads,
            )
            return run_one_static(
                cfg=cfg,
                seed=job.seed,
                out_root=args.out_root,
                python_exec=args.python_exec,
                physical_gpu_id=physical_gpu_id,
                epochs=args.max_epochs,
                eval_every=args.eval_every,
                patience_evals=args.patience_evals,
                min_delta=args.min_delta,
                include_epoch0=args.include_epoch0,
                stop_on_plateau=stop_on_plateau,
                extra_args=static_extra_args,
                print_prefix=prefix,
            )
        cfg = TemporalConfig(method_id=job.method_id, model=job.temporal_model)
        return run_one_temporal(
            cfg=cfg,
            seed=job.seed,
            out_root=args.out_root,
            python_exec=args.python_exec,
            physical_gpu_id=physical_gpu_id,
            epochs=args.max_epochs,
            eval_every=args.eval_every,
            patience_evals=args.patience_evals,
            min_delta=args.min_delta,
            include_epoch0=args.include_epoch0,
            stop_on_plateau=stop_on_plateau,
            print_prefix=prefix,
            extra_args=temporal_extra_args,
        )

    running: Dict = {}
    running_counts: Dict[int, int] = {}  # keyed by physical gpu id
    next_job_idx = 0

    def _try_schedule_available(executor: ThreadPoolExecutor):
        nonlocal next_job_idx
        while next_job_idx < len(jobs):
            choice = _choose_available_gpu_physical(
                candidate_physical_ids=candidate_gpus,
                running_counts=running_counts,
                max_jobs_per_gpu=args.max_jobs_per_gpu,
                gpu_free_threshold_mb=args.gpu_free_threshold_mb,
                strict_free_check=args.strict_free_check,
            )
            if choice is None:
                return
            physical_gpu_id, free_mb = choice
            job = jobs[next_job_idx]
            next_job_idx += 1
            free_msg = "" if free_mb is None else f" (free={free_mb}MB)"
            print(f"[scheduler] start {job.kind}:{job.method_id} seed={job.seed} on physical_gpu={physical_gpu_id}{free_msg}")
            running_counts[physical_gpu_id] = running_counts.get(physical_gpu_id, 0) + 1
            running[executor.submit(_run_job, job, physical_gpu_id)] = (job, physical_gpu_id)

    with ThreadPoolExecutor(max_workers=len(candidate_gpus)) as executor:
        while next_job_idx < len(jobs) or running:
            _try_schedule_available(executor)

            if not running:
                time.sleep(args.scheduler_poll_sec)
                continue

            done, _ = wait(list(running.keys()), timeout=args.scheduler_poll_sec, return_when=FIRST_COMPLETED)
            for fut in done:
                job, physical_gpu_id = running.pop(fut)
                running_counts[physical_gpu_id] = max(0, running_counts.get(physical_gpu_id, 1) - 1)
                try:
                    row = fut.result()
                except Exception as e:
                    row = {
                        "family": job.kind,
                        "method_id": job.method_id,
                        "gnn_layer_types": job.gnn_layer_types,
                        "seed": job.seed,
                        "returncode": 999,
                        "stdout_log": "",
                        "results_json": "",
                        "best_val_loss": "",
                        "best_val_mrr": "",
                        "best_epoch": "",
                        "converged_epoch": "",
                        "train_time_to_converge_sec": "",
                        "wallclock_sec": "",
                        "eval_every": args.eval_every,
                        "patience_evals": args.patience_evals,
                        "min_delta": args.min_delta,
                        "max_epochs": args.max_epochs,
                        "gpu_id": physical_gpu_id,
                    }
                    print(f"[scheduler] FAILED {job.kind}:{job.method_id} seed={job.seed} gpu={physical_gpu_id}: {e}", file=sys.stderr)

                with append_lock:
                    _append_row(summary_csv, fields, row)
                print(f"[scheduler] done {job.kind}:{job.method_id} seed={job.seed} on physical_gpu={physical_gpu_id} (returncode={row.get('returncode')})")

    print(f"\nWrote: {summary_csv}")


if __name__ == "__main__":
    main()