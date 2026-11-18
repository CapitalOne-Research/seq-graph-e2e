#!/usr/bin/env python3
"""Idempotent GPU-pool launcher for the fraud RDL comparison (RelGNN + RelGT) across N GPUs.

Keeps every GPU busy until the queue drains (jobs finish at different times, so a shared pool
beats one static loop per GPU). Safe to re-run: skips any job whose result JSON already exists,
and skips (with a warning) any job whose relational-table dir hasn't been built on this box.

RelGT is DDP; each job is launched with `torchrun --standalone --nproc_per_node=1`, which picks
its own ephemeral rendezvous port, so concurrent RelGT jobs never collide.

Outputs (recognized later by amazon/collect_fraud_results.py):
  RelGNN  -> RelGNN-main/examples/results/relgnn_fraud/amazon-<ds>/fraud/relgnn_seed<seed>.json
  RelGT   -> relgt-main/results/<fraud_default|fraud>/amazon-<ds>/fraud/<seed>.json

Examples:
  python amazon/run_fraud_rdl_pool.py --gpus 0,1,2,3 --dry_run
  python amazon/run_fraud_rdl_pool.py --gpus 0,1,2,3
  python amazon/run_fraud_rdl_pool.py --gpus 0,1,2,3 --models relgnn
  python amazon/run_fraud_rdl_pool.py --gpus 0,1 --relgt_cfg matched --relgt_datasets clothing
"""
import argparse
import os
import subprocess
import shlex
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # amazon/ -> repo root


def rel_dir(ds, seed=None):
    # per-seed arena (fraud_detection_seed<seed>) so RelGT matches the baselines' seed-specific
    # 60/20/20 split; seed=None uses the legacy single arena.
    sub = "fraud_detection" if seed is None else f"fraud_detection_seed{seed}"
    return f"{REPO}/data/tasks/amazon-{ds}/{sub}/relational"


def relgnn_job(ds, seed, epochs, per_seed=True):
    rd = rel_dir(ds, seed if per_seed else None)
    out = f"{REPO}/RelGNN-main/examples/results/relgnn_fraud/amazon-{ds}/fraud/relgnn_seed{seed}.json"
    cmd = (f"cd {REPO}/RelGNN-main/examples && "
           f"FRAUD_REL_DIR={rd} HDF5_USE_FILE_LOCKING=FALSE "
           f"python relgnn_node_fraud.py --dataset amazon-{ds} --task fraud "
           f"--no_temporal_sampling --epochs {epochs} --seed {seed}")
    return f"relgnn/{ds}/s{seed}", out, cmd


def relgt_job(ds, seed, cfg, epochs, per_seed=True, use_text=False):
    sub = "fraud_default" if cfg == "full" else "fraud"
    tag = "_text" if use_text else ""              # keep text vs no-text results/caches fully separate
    sub_out = (f"{sub}_canon" if per_seed else sub) + tag
    extra = "" if cfg == "full" else "--channels 128 --num_neighbors 64 "
    rd = rel_dir(ds, seed if per_seed else None)
    # Per-seed token cache: RelGT keys its HDF5 precompute by (dataset, task) ONLY, so all seeds would
    # otherwise collide on one cache (and the old-arena cache gets reused). An isolated cache_dir per seed
    # keeps each seed's split correct and lets same-dataset jobs run concurrently (no shared-cache crash).
    # The text variant needs its OWN cache root too, or it reuses the no-text materialized/precomputed tokens.
    cache_root = "token_cache_text_canon" if use_text else "token_cache_canon"
    cache = f"--cache_dir {REPO}/relgt-main/{cache_root}/seed{seed} " if per_seed else ""
    text_flag = "--use_review_text " if use_text else ""
    out = f"{REPO}/relgt-main/results/{sub_out}/amazon-{ds}/fraud/{seed}.json"
    cmd = (f"cd {REPO}/relgt-main && "
           f"FRAUD_REL_DIR={rd} HDF5_USE_FILE_LOCKING=FALSE "
           f"torchrun --standalone --nproc_per_node=1 main_node_ddp.py "
           f"--dataset amazon-{ds} --task fraud --epochs {epochs} --seed {seed} "
           f"{extra}{text_flag}{cache}--out_dir results/{sub_out}")
    return f"relgt-{cfg}{tag}/{ds}/s{seed}", out, cmd


def build_jobs(args):
    seeds = [int(s) for s in args.seeds.split(",")]
    models = args.models.split(",")
    per_seed = not args.single_arena
    jobs = []
    if "relgnn" in models:
        for ds in args.relgnn_datasets.split(","):
            for s in seeds:
                jobs.append(relgnn_job(ds, s, args.epochs, per_seed))
    if "relgt" in models:
        cfgs = ["full", "matched"] if args.relgt_cfg == "both" else [args.relgt_cfg]
        for cfg in cfgs:
            for ds in args.relgt_datasets.split(","):
                for s in seeds:
                    jobs.append(relgt_job(ds, s, cfg, args.epochs, per_seed, args.use_review_text))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--models", default="relgnn,relgt")
    ap.add_argument("--relgnn_datasets", default="clothing,electronics,musical")
    ap.add_argument("--relgt_datasets", default="clothing,electronics,movies")
    ap.add_argument("--relgt_cfg", default="full", choices=["full", "matched", "both"])
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--poll", type=int, default=5, help="seconds between reap/fill cycles")
    ap.add_argument("--single_arena", action="store_true",
                    help="use the legacy single fraud_detection arena instead of per-seed "
                         "fraud_detection_seed<seed> (per-seed is the fair default)")
    ap.add_argument("--use_review_text", action="store_true",
                    help="RelGT only: feed the per-user pooled SBert review-text vector as a node "
                         "embedding feature (mirrors static+SBert). Writes to results/*_text and a "
                         "separate token cache; prebuilds user_sbert.npy first.")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    prebuilds = []
    if args.use_review_text and "relgt" in args.models.split(","):
        for ds in args.relgt_datasets.split(","):
            proc = f"{REPO}/data/processed/amazon-{ds}"
            arena = "fraud_detection" if args.single_arena else "fraud_detection_seed*"
            rglob = f"{REPO}/data/tasks/amazon-{ds}/{arena}/relational"
            prebuilds.append(["python", f"{REPO}/amazon/build_user_sbert.py",
                              "--dataset", ds, "--processed", proc, "--relational_glob", rglob])

    # A dry run is a complete, read-only plan, including commands blocked by missing
    # data. Return before embedding/model processes or even GPU discovery.
    if args.dry_run:
        for cmd in prebuilds:
            print("[prebuild] " + shlex.join(cmd))
            if not os.path.isfile(os.path.join(cmd[5], "all_nodes.csv")):
                print(f"[missing dependency] {cmd[5]}/all_nodes.csv (including reviewText)")
        jobs = build_jobs(args)
        print(f"\n[dry run] {len(jobs)} planned trainer command(s); requested GPUs {args.gpus}:")
        for name, out, cmd in jobs:
            ds, seed = name.split("/")[1:]
            rd = rel_dir(ds, None if args.single_arena else int(seed[1:]))
            if os.path.exists(out):
                status = "skip: done"
            elif not os.path.isdir(rd):
                status = f"missing dependency: {rd}"
            else:
                status = "ready"
            print(f"[{status}] {name}\n  {cmd}")
        return

    for cmd in prebuilds:
        print("[prebuild] " + shlex.join(cmd))
        subprocess.run(cmd, check=True)

    gpus = [int(g) for g in args.gpus.split(",")]
    # Boxes differ in GPU count (relgt-0/iccd-0/ddp-0); drop indices this box lacks, else jobs on
    # phantom GPUs die at NCCL init with "ProcessGroupNCCL is only supported with GPUs, no GPUs found".
    try:
        _smi = subprocess.run("nvidia-smi --query-gpu=index --format=csv,noheader,nounits",
                              shell=True, capture_output=True, text=True, check=True)
        _have = {int(x) for x in _smi.stdout.split()}
        _bad = [g for g in gpus if g not in _have]
        if _bad:
            print(f"[gpu-guard] requested GPUs absent on this box, dropping {_bad} (have {sorted(_have)})")
            gpus = [g for g in gpus if g in _have]
        if not gpus:
            raise SystemExit(f"[gpu-guard] none of the requested GPUs exist here (have {sorted(_have)})")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[gpu-guard] nvidia-smi unavailable; using requested GPUs as-is")
    all_jobs = build_jobs(args)

    queue = []
    for name, out, cmd in all_jobs:
        if os.path.exists(out):
            print(f"[skip: done]      {name}")
        elif not os.path.isdir(rel_dir(name.split('/')[1],
                                       None if args.single_arena else int(name.split('/')[2][1:]))):
            print(f"[skip: no data]   {name}  (build per-seed relational tables first)")
        else:
            queue.append((name, out, cmd))

    print(f"\n{len(queue)} job(s) to run on GPUs {gpus}:")
    for name, _, _ in queue:
        print(f"  - {name}")
    if not queue:
        print("Nothing to run.")
        return

    active = {}  # gpu -> (proc, name)
    while queue or active:
        for g, (p, nm) in list(active.items()):
            if p.poll() is not None:
                tag = "ok" if p.returncode == 0 else f"FAIL rc={p.returncode}"
                print(f"[gpu {g}] finished [{tag}]: {nm}")
                del active[g]
        for g in gpus:
            if g in active or not queue:
                continue
            name, out, cmd = queue.pop(0)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(g)
            env["WANDB_MODE"] = "disabled"  # public wandb is 403-blocked on the boxes; no-op it
            log = f"{REPO}/pool_{name.replace('/', '_')}.log"
            print(f"[gpu {g}] launch: {name}   (log: {os.path.basename(log)})")
            with open(log, "w") as lf:
                p = subprocess.Popen(cmd, shell=True, env=env, stdout=lf, stderr=subprocess.STDOUT)
            active[g] = (p, name)
        time.sleep(args.poll)

    print("\nAll jobs done. Collect with:  python amazon/collect_fraud_results.py "
          "--relgt relgt-main/results RelGNN-main/examples/results")


if __name__ == "__main__":
    main()
