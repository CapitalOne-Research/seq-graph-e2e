#!/usr/bin/env python3
"""Run pending rebuttal experiments on all sufficiently empty GPUs.

GPU jobs are pulled from a shared queue, so a GPU receives another job as soon
as its current job finishes. The CPU-only Amazon overlap diagnostic runs in
parallel. Paired statistics run only after all six prediction dumps exist.
"""

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
import traceback


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CKPT_DIR = HERE / "brightkite_checkpoints_seeded"
LOG_DIR = HERE / "rebuttal_run_logs"

BASE = "e2e_gcn-gcn2x2_bert1x2_hidden-dim128_mean_nngb10_nneg100_bs64_lr0.0001_epochs100"
TAGS = ("frozenseq_matched", "unfrozen2stage")
SEEDS = (0, 1, 2)


def available_gpus(min_free_mib, max_used_mib, max_gpus):
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.used,memory.free,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError("nvidia-smi was not found; run this launcher on the GPU server.") from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "no diagnostic output").strip()
        raise RuntimeError(f"nvidia-smi failed: {detail}") from exc
    found = []
    print("[gpu preflight] GPUs visible to nvidia-smi:")
    for line in result.stdout.splitlines():
        index, name, gpu_uuid, used, free, total = [part.strip() for part in line.split(",", 5)]
        used_mib, free_mib, total_mib = int(used), int(free), int(total)
        eligible = used_mib <= max_used_mib and free_mib >= min_free_mib
        reasons = []
        if used_mib > max_used_mib:
            reasons.append(f"used>{max_used_mib}")
        if free_mib < min_free_mib:
            reasons.append(f"free<{min_free_mib}")
        status = "eligible" if eligible else "rejected: " + ", ".join(reasons)
        print(
            f"  GPU {index}: {name}; used={used_mib} MiB; "
            f"free={free_mib}/{total_mib} MiB; {status}"
        )
        if eligible:
            found.append((index, name, gpu_uuid, used_mib, free_mib, total_mib))
    return found[:max_gpus] if max_gpus else found


def verify_gpu_bindings(gpus):
    """Prove that each isolated child's cuda:0 is the intended physical GPU."""
    probe = (
        "import json, torch; "
        "n=torch.cuda.device_count(); "
        "p=torch.cuda.get_device_properties(0) if n else None; "
        "print(json.dumps({'count':n,'name':getattr(p,'name',None),"
        "'uuid':str(getattr(p,'uuid',''))}))"
    )
    verified = []
    for gpu in gpus:
        index, name, expected_uuid, used_mib, free_mib, total_mib = gpu
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = index
        result = subprocess.run(
            [sys.executable, "-c", probe], text=True, capture_output=True, env=env
        )
        if result.returncode:
            detail = (result.stderr or result.stdout or "no output").strip()
            raise RuntimeError(f"CUDA binding probe failed for physical GPU {index}: {detail}")
        try:
            observed = json.loads(result.stdout.strip().splitlines()[-1])
        except (IndexError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"Could not parse CUDA binding probe for physical GPU {index}: {result.stdout!r}"
            ) from exc
        if observed["count"] != 1:
            raise RuntimeError(
                f"Binding physical GPU {index} exposed {observed['count']} CUDA devices; expected 1."
            )
        observed_uuid = observed.get("uuid", "").removeprefix("GPU-").lower()
        normalized_expected = expected_uuid.removeprefix("GPU-").lower()
        if observed_uuid and observed_uuid != normalized_expected:
            raise RuntimeError(
                f"GPU binding mismatch: requested physical GPU {index} ({expected_uuid}), "
                f"but child cuda:0 resolved to {observed.get('uuid')}."
            )
        identity = observed.get("uuid") or "UUID unavailable in this PyTorch build"
        print(
            f"[binding verified] physical GPU {index} -> child cuda:0; "
            f"{observed.get('name')}; {identity}"
        )
        verified.append(gpu)
    return verified


def run_logged(name, cmd, cwd, log_path, env=None):
    start = time.perf_counter()
    print(f"[start] {name}: {' '.join(map(str, cmd))}", flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"command: {' '.join(map(str, cmd))}\n")
        log.write(f"cwd: {cwd}\n")
        log.flush()
        try:
            proc = subprocess.run(
                [str(x) for x in cmd],
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
            returncode = proc.returncode
        except Exception:
            traceback.print_exc(file=log)
            returncode = 127
        elapsed = time.perf_counter() - start
        log.write(f"\nlauncher_wall_sec: {elapsed:.3f}\n")
        log.write(f"exit_code: {returncode}\n")
    print(f"[done] {name}: exit={returncode}, wall={elapsed / 3600:.2f} h", flush=True)
    return returncode


def print_log_tail(log_path, lines=20):
    """Print a compact failure excerpt while retaining the complete log on disk."""
    try:
        content = Path(log_path).read_text(errors="replace").splitlines()
    except OSError as exc:
        print(f"    Could not read log: {exc}")
        return
    print(f"    Last {min(lines, len(content))} log lines:")
    for line in content[-lines:]:
        print(f"    | {line}")


def checkpoint(seed, tag):
    return CKPT_DIR / f"{BASE}_seed{seed}_{tag}_best.pth"


def prediction_dump(seed, tag):
    return CKPT_DIR / f"{BASE}_seed{seed}_{tag}_best_test_preds.npz"


def build_gpu_tasks(skip_mlm):
    tasks = []
    timing_json = HERE / "stored/mlm_brightkite_1L_timed.pth.timing.json"
    mlm_data = ROOT / "data/tasks/brightkite/friend_recommendation/personal.csv"
    if not skip_mlm and not timing_json.exists():
        if mlm_data.exists():
            tasks.append((
                "mlm_brightkite_timing",
                [
                    sys.executable, "sequence_embedding.py", "--mode", "train",
                    "--num_hidden_layers", "1", "--num_attention_heads", "2",
                    "--hidden_size", "128", "--max_events", "100",
                    "--batch_size", "256", "--num_epochs", "50",
                    "--learning_rate", "1e-3", "--csv_file", str(mlm_data),
                    "--model_file", "mlm_brightkite_1L_timed.pth",
                    "--embedding_file", "mlm_brightkite_1L_timed_embeddings.pkl",
                ],
                LOG_DIR / "mlm_brightkite_timing.log",
            ))
        else:
            print(f"[skip] MLM data missing: {mlm_data}")

    for tag in TAGS:
        for seed in SEEDS:
            ckpt = checkpoint(seed, tag)
            dump = prediction_dump(seed, tag)
            if dump.exists():
                print(f"[skip] prediction dump exists: {dump.name}")
            elif ckpt.exists():
                tasks.append((
                    f"eval_{tag}_seed{seed}",
                    [
                        sys.executable, "joint.py",
                        "--load_checkpoint_and_eval", str(ckpt),
                        "--dump_predictions",
                        "--eval_batch_size", "512",
                    ],
                    LOG_DIR / f"eval_{tag}_seed{seed}.log",
                ))
            else:
                print(f"[missing] checkpoint: {ckpt}")
    return tasks


def start_amazon_overlap(skip_amazon):
    if skip_amazon:
        return None
    amazon_dir = ROOT / "amazon"
    required = []
    for dataset in ("movies", "electronics", "clothing"):
        task = ROOT / f"data/tasks/amazon-{dataset}/coreview_prediction"
        required.extend([
            task / "personal_observed.csv",
            task / "relational_test.csv",
            task / "relational_test_negative_sample_100.csv",
        ])
    missing = [path for path in required if not path.exists()]
    if missing:
        print("[skip] Amazon overlap data missing:")
        for path in missing:
            print(f"  {path}")
        return None

    log_path = amazon_dir / "amazon_overlap_test_results.txt"
    cmd = [
        sys.executable, "leakage_diagnostic.py", "--data_root", "../data/tasks",
        "--datasets", "movies", "electronics", "clothing", "--split", "test",
        "--neg_suffix", "_negative_sample_100",
    ]
    holder = {}

    def target():
        holder["returncode"] = run_logged("amazon_overlap", cmd, amazon_dir, log_path)

    thread = threading.Thread(target=target, name="amazon-overlap")
    thread.start()
    return thread, holder


def run_gpu_queue(gpus, tasks):
    work = queue.Queue()
    for task in tasks:
        work.put(task)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        index, name, gpu_uuid, used_mib, free_mib, total_mib = gpu
        print(f"[worker] GPU {index}: {name}, used {used_mib} MiB, free {free_mib}/{total_mib} MiB")
        while True:
            try:
                task_name, cmd, log_path = work.get_nowait()
            except queue.Empty:
                return
            env = os.environ.copy()
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = index
            env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
            print(f"[assign] {task_name} -> physical GPU {index}", flush=True)
            code = run_logged(task_name, cmd, HERE, log_path, env)
            if code:
                with lock:
                    failures.append((task_name, code, log_path))
            work.task_done()

    workers = [threading.Thread(target=worker, args=(gpu,), name=f"gpu-{gpu[0]}") for gpu in gpus]
    for thread in workers:
        thread.start()
    for thread in workers:
        thread.join()
    return failures


def run_statistics():
    expected = [prediction_dump(seed, tag) for tag in TAGS for seed in SEEDS]
    missing = [path for path in expected if not path.exists()]
    if missing:
        print("[skip] paired statistics; missing prediction dumps:")
        for path in missing:
            print(f"  {path}")
        return 1
    cmd = [
        sys.executable, "notebooks/paired_wilcoxon_rr.py",
        "--a", str(CKPT_DIR / "*bs64*_unfrozen2stage_best_test_preds.npz"),
        "--name-a", "MLM-pretrained + fine-tuned",
        "--b", str(CKPT_DIR / "*bs64*_frozenseq_matched_best_test_preds.npz"),
        "--name-b", "MLM-pretrained + frozen", "--n-boot", "10000",
    ]
    return run_logged(
        "paired_statistics", cmd, ROOT,
        HERE / "paired_frozen_vs_finetuned.txt",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-free-mib", type=int, default=43000,
                        help="Only use GPUs with at least this much free memory.")
    parser.add_argument("--max-used-mib", type=int, default=500,
                        help="Only use GPUs with at most this much allocated memory.")
    parser.add_argument("--max-gpus", type=int, default=0,
                        help="Maximum GPUs to use; 0 means all eligible GPUs.")
    parser.add_argument("--skip-mlm", action="store_true")
    parser.add_argument("--skip-amazon", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    tasks = build_gpu_tasks(args.skip_mlm)
    print(f"[scheduler] pending GPU jobs ({len(tasks)}):")
    for task_name, _, _ in tasks:
        print(f"  {task_name}")
    failures = []
    if tasks:
        gpus = available_gpus(args.min_free_mib, args.max_used_mib, args.max_gpus)
        if not gpus:
            raise SystemExit(
                f"No GPU has at least {args.min_free_mib} MiB free and at most "
                f"{args.max_used_mib} MiB used."
            )
        print(f"[scheduler] eligible GPUs: {[gpu[0] for gpu in gpus]}")
        gpus = verify_gpu_bindings(gpus)
    else:
        print("[scheduler] no pending GPU jobs")

    amazon = start_amazon_overlap(args.skip_amazon)
    if tasks:
        failures = run_gpu_queue(gpus, tasks)

    if amazon:
        amazon[0].join()
        if amazon[1].get("returncode"):
            failures.append(("amazon_overlap", amazon[1]["returncode"], ROOT / "amazon/amazon_overlap_test_results.txt"))

    stats_code = run_statistics()
    if stats_code:
        failures.append(("paired_statistics", stats_code, HERE / "paired_frozen_vs_finetuned.txt"))

    if failures:
        print("[failed] One or more steps failed. Full logs and excerpts follow:")
        for name, code, log in failures:
            print(f"  {name}: exit={code}; log={log}")
            print_log_tail(log)
        raise SystemExit(1)
    print("[complete] all available rebuttal experiments finished successfully")


if __name__ == "__main__":
    main()
