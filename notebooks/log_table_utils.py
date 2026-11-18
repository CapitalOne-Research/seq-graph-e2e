"""Utility helpers for turning local experiment logs into LaTeX-ready tables."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import math

# keep this ONCE, near the top
_BRIDGE_RX = re.compile(
    r'(?i)^bridge[-_](tokenxat)[-_](gatv2|gat|gcn|transformerconv)(?=$|[_-])'
)

def _map_bridge_method_from_text(txt: str) -> Optional[str]:
    if not isinstance(txt, str):
        return None
    m = _BRIDGE_RX.match(txt.strip())
    if not m:
        return None
    base = m.group(2).lower()
    base_disp = "GAT" if base.startswith("gat") else ("GCN" if base == "gcn" else "TransformerConv")
    return f"Bridge-TokenXAT+{base_disp}"

def _fmt_pct_mean_std(mean: Any, std: Any, *, places: int = 1, mult: float = 100.0) -> str:
    """Format as $ (mean*100) ± (std*100) $ with one decimal by default."""
    if mean is None or pd.isna(mean):
        return "--"
    m = f"{float(mean) * mult:.{places}f}"
    if std is None or pd.isna(std):
        return f"${m}$"
    s = f"{float(std) * mult:.{places}f}"
    return f"${m}\\,{{\\scriptstyle \\pm {s}}}$"


def _latex_escape(s: str) -> str:
    """Escape LaTeX special chars in text cells."""
    repl = {
        '\\': r'\textbackslash{}',
        '{': r'\{', '}': r'\}',
        '#': r'\#', '$': r'\$', '%': r'\%', '&': r'\&',
        '_': r'\_', '^': r'\^{}', '~': r'\~{}',
    }
    out = []
    for ch in str(s):
        out.append(repl.get(ch, ch))
    return ''.join(out)


def _resolve_metric_label_maps(df: pd.DataFrame, label_map: Dict[str, str]):
    """
    Map from any of the following to display labels:
      test_<key>_mean/_std
      test_<key>_last_mean/_std, test_<key>_best_mean/_std
      val_<key>_mean/_std
      val_<key>_last_mean/_std, val_<key>_best_mean/_std
    """
    mean_map, std_map = {}, {}

    # label_map looks like: {"test_max_f1_mean": "Max F1", ...}
    wanted = []
    for src, disp in label_map.items():
        m = re.match(r"test_(.+)_mean$", src)
        if m:
            wanted.append((m.group(1), disp))  # e.g. ("max_f1", "Max F1")

    mean_suffixes = ["_mean", "_last_mean", "_best_mean"]
    std_suffixes  = ["_std",  "_last_std",  "_best_std"]

    for key, disp in wanted:
        # try test_* means
        for suf in mean_suffixes:
            col = f"test_{key}{suf}"
            if col in df.columns:
                mean_map[col] = disp
                break
        else:
            # fallback to val_* means
            for suf in mean_suffixes:
                col = f"val_{key}{suf}"
                if col in df.columns:
                    mean_map[col] = disp
                    break

        # std: mirror the mean source if present; otherwise search all std candidates
        # try test_* std
        found_std = False
        for suf in std_suffixes:
            col = f"test_{key}{suf}"
            if col in df.columns:
                std_map[col] = f"{disp}_std"
                found_std = True
                break
        if not found_std:
            for suf in std_suffixes:
                col = f"val_{key}{suf}"
                if col in df.columns:
                    std_map[col] = f"{disp}_std"
                    break

    return mean_map, std_map


def _fmt_mean_std(mean: Any, std: Any, *, places: int = 4) -> str:
    """Format as $m\,{\scriptstyle \pm s}$; fallback to $m$ when std is missing."""
    if mean is None or (isinstance(mean, float) and not math.isfinite(mean)) or pd.isna(mean):
        return "--"
    m = f"{float(mean):.{places}f}"
    if std is None or (isinstance(std, float) and not math.isfinite(std)) or pd.isna(std):
        return f"${m}$"
    s = f"{float(std):.{places}f}"
    return f"${m}\\,{{\\scriptstyle \\pm {s}}}$"

def _is_close(a: float, b: float, tol: float = 1e-12) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return abs(a - b) <= tol

VAL_SELECTION_PRIORITY: Sequence[str] = (
    "val_max_f1_best",
    "val_max_f1_last",
    "val_max_f1",
    "val_pr_auc_best",
    "val_pr_auc",
    "val_auc_best",
    "val_auc",
)

VAL_RECORD_PRIORITY: Sequence[str] = (
    "val_max_f1",
    "val_pr_auc",
    "val_auc",
    "val_f1",
    "val_accuracy",
)

MINIMIZE_PREFIXES: Sequence[str] = ("val_loss", "train_loss")

METRIC_LABELS: Dict[str, str] = {
    "test_max_f1_mean": "Max F1",
    "test_pr_auc_mean": "PR AUC",
    "test_precision_at_10_percent_mean": "Precision@10%",
    "test_recall_at_10_percent_mean": "Recall@10%",
}

DEFAULT_HPARAM_COLUMNS: Sequence[str] = (
    "lr",
    "batch_size",
    "heads",
    "num_neighbors",
    "compress_dim",
    "use_sequence_embedding",
)

TABLE_STRUCTURE: Dict[str, List[str]] = {
    "Graph Only": ["GAT", "GCN", "TransformerConv"],
    "Temporal Graph Methods": ["DyRep", "TGN"],
    "Two-Stage": [
        "GCN (Two-Stage)",
        "GAT (Two-Stage)",
        "TransformerConv (Two-Stage)",
        "TGN (Two-Stage)",
        "DyRep (Two-Stage)",
    ],
    "Sequence Only": ["SBert"],
    "Sequence + Graph (Ours)": ["SBert + GCN", "SBert + GAT", "SBert + TransformerConv", "SBert + TokenXAT"],
    # Bridge-* end-to-end methods (populate dynamically in generate_latex_table)
    "Bridge Methods (Advanced)": ["Bridge-TokenXAT+GAT", "Bridge-TokenXAT+TransformerConv"],
}

MethodBuilder = Callable[[pd.Series], str]

__all__ = [
    "process_local_logs_to_dataframe",
    "process_gnn_results_from_logs",
    "process_sbert_results_from_logs",
    "process_sbert_results_from_csv",
    "summarize_runs",
    "generate_latex_table",
    "process_dataset_directly",
]


def _safe_search(pattern: str, text: str) -> Optional[str]:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def _to_int(value: Optional[str], default: int = -1) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def flatten_dict(data: Dict[str, Any], prefix: str = "", sep: str = "_") -> Dict[str, Any]:
    items: Dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{prefix}{sep}{key}" if prefix else str(key)
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key, sep=sep))
        else:
            items[new_key] = value
    return items


def _parse_filename_metadata(path: Path) -> Dict[str, Any]:
    stem = path.stem
    base_name = stem.replace("_best", "")
    metadata: Dict[str, Any] = {
        "name": base_name,
        "run_group_name": re.sub(r"_seed\d+", "", base_name),
        "seed": _to_int(_safe_search(r"_seed(\d+)", base_name)),
        "source_path": str(path),
        "source_dir": str(path.parent),
    }

    # Infer whether this run used sequence (SBERT) embeddings from the filename.
    # Static/temporal graph logs that use SBERT are saved with a *_sbert.json suffix.
    metadata["use_sequence_embedding"] = stem.endswith("_sbert")

    lr_value = _safe_search(r"_lr([\d.eE+-]+)", base_name)
    metadata["lr"] = float(lr_value) if lr_value else None
    metadata["learning_rate"] = metadata["lr"]
    metadata["batch_size"] = _to_int(_safe_search(r"_bs(\d+)", base_name))
    metadata["epochs"] = _to_int(_safe_search(r"_epochs(\d+)", base_name))
    metadata["heads"] = _to_int(_safe_search(r"_heads(\d+)", base_name), default=1)
    metadata["num_neighbors"] = _to_int(_safe_search(r"_nngb(\d+)", base_name))
    metadata["compress_dim"] = _to_int(_safe_search(r"_cdim(\d+)", base_name))
    metadata["clf_layers"] = _to_int(_safe_search(r"_clf(\d+)", base_name))

    if base_name.startswith("bridge_"):
        layer_spec = _safe_search(r"^bridge_([^_]+?)(?:_heads|_seed|$)", base_name)
        if layer_spec:
            metadata["gnn_layer_types"] = layer_spec.replace("-", ",")
            metadata["bridge_method"] = layer_spec.split("-")[0]
    else:
        layer_from_layers = _safe_search(r"^(.*?)_layers", base_name)
        if layer_from_layers:
            metadata["gnn_layer_types"] = layer_from_layers.replace("-", ",")
        else:
            head_prefix = _safe_search(r"^(.*?_heads)", base_name)
            if head_prefix:
                metadata["gnn_layer_types"] = head_prefix.replace("_heads", "").replace("-", ",")

    return metadata


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        print(f"Warning: Failed to parse JSON from {path}: {err}")
        return []

    if isinstance(payload, dict):
        payload = [payload]

    return [flatten_dict(entry) for entry in payload if isinstance(entry, dict)]


def _pick_target_record_metric(records: List[Dict[str, Any]]) -> Optional[str]:
    for metric in VAL_RECORD_PRIORITY:
        if any(metric in record for record in records):
            return metric
    return None


def _extract_epoch_summaries(records: List[Dict[str, Any]]):
    if not records:
        return {}, {}, {}, {}, {}, None

    target_key = _pick_target_record_metric(records)
    if target_key:
        best_entry = max(records, key=lambda item: item.get(target_key, float("-inf")))
    else:
        best_entry = records[-1]

    last_entry = records[-1]
    val_entry = next((item for item in reversed(records) if any(k.startswith("val_") for k in item)), {})
    test_entry = next((item for item in reversed(records) if any(k.startswith("test_") for k in item)), {})
    train_entry = next((item for item in reversed(records) if any(k.startswith("train_") for k in item)), {})
    return best_entry, last_entry, val_entry, test_entry, train_entry, target_key


def _build_log_row(path: Path) -> Optional[Dict[str, Any]]:
    records = _load_json_records(path)
    if not records:
        return None

    best_entry, last_entry, val_entry, test_entry, train_entry, target_key = _extract_epoch_summaries(records)
    metadata = _parse_filename_metadata(path)

    row: Dict[str, Any] = dict(metadata)
    row["best_epoch"] = int(best_entry.get("epoch", metadata.get("best_epoch", -1)))
    if target_key:
        row["target_metric_key"] = target_key

    for key, value in best_entry.items():
        if key.startswith("val_"):
            row[f"{key}_best"] = value

    for key, value in val_entry.items():
        if key.startswith("val_"):
            row[f"{key}_last"] = value

    for key, value in train_entry.items():
        if key.startswith("train_"):
            row[f"{key}_last"] = value

    for key, value in test_entry.items():
        if key.startswith("test_"):
            row[key] = value

    # Report every method at its val-selected checkpoint: when a run logged best_test_* (static/BRIDGE/
    # SBert do; temporal already reloads its best-val model; RelGT is val-selected), use it as test_*.
    for key, value in test_entry.items():
        if key.startswith("best_test_"):
            row[key[len("best_"):]] = value  # best_test_pr_auc -> test_pr_auc

    for key in ("epoch", "_step", "_runtime", "_timestamp"):
        if key in last_entry and key not in row:
            row[key] = last_entry[key]

    return row


def process_local_logs_to_dataframe(log_dir: str) -> pd.DataFrame:
    log_path = Path(log_dir)
    print(f"Scanning for local JSON logs in: {log_path}")
    if not log_path.is_dir():
        print(f"Warning: Log directory not found at '{log_dir}'.")
        return pd.DataFrame()

    rows = [row for row in (_build_log_row(path) for path in sorted(log_path.glob("*.json"))) if row]
    if not rows:
        print(f"No JSON logs processed in '{log_dir}'.")
        return pd.DataFrame()

    frame = pd.DataFrame(rows)
    numeric_candidates = [
        column
        for column in frame.columns
        if column.startswith(("val_", "test_", "train_")) or column in {"lr", "learning_rate"}
    ]
    for column in numeric_candidates:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _candidate_csv_paths(log_dir: str) -> List[Path]:
    stripped = log_dir.rstrip("/\\")
    raw_name = stripped.replace("logs/", "")
    candidates = {
        Path(f"{raw_name}-local.csv"),
        Path("notebooks") / f"{raw_name}-local.csv",
        Path(f"{Path(stripped).name}-local.csv"),
        Path("notebooks") / f"{Path(stripped).name}-local.csv",
    }
    return [path for path in candidates if path.name != "-local.csv"]


def _load_sources(log_directories: Iterable[str], allow_csv_fallback: bool = False) -> List[pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for log_dir in log_directories:
        frame = process_local_logs_to_dataframe(log_dir)
        if not frame.empty:
            frames.append(frame)
            continue

        if not allow_csv_fallback:
            continue

        for csv_path in _candidate_csv_paths(log_dir):
            if csv_path.is_file():
                try:
                    csv_frame = pd.read_csv(csv_path)
                    csv_frame["source_path"] = str(csv_path)
                    frames.append(csv_frame)
                    print(f"Loaded {len(csv_frame)} rows from fallback CSV: {csv_path}")
                    break
                except Exception as exc:
                    print(f"Warning: Failed to read fallback CSV '{csv_path}': {exc}")
    return frames


def _is_minimization_key(column: str) -> bool:
    return any(column.startswith(prefix) for prefix in MINIMIZE_PREFIXES)


def _select_selection_column(frame: pd.DataFrame, priority: Iterable[str]) -> Tuple[Optional[str], bool]:
    for base in priority:
        for candidate in (f"{base}_mean", base):
            if candidate in frame.columns:
                return candidate, not _is_minimization_key(candidate)
    return None, True


def _build_gnn_method(row: pd.Series) -> str:
    run_group = str(row.get("run_group_name", "") or "")
    token = run_group.split("_")[0].upper() if run_group else ""
    use_seq = bool(row.get("use_sequence_embedding", False))

    mapping = {
        "TRANSFORMERCONV": "TransformerConv",
        "DYREP": "DyRep",
        "GAT": "GAT",
        "GCN": "GCN",
        "TGN": "TGN",
    }
    if not token:
        base = "Unknown"
    else:
        base = mapping.get(token, token)  # keeps exact uppercase for GAT/GCN/TGN

    if use_seq:
        return f"{base} (Two-Stage)"
    return base



def _build_sbert_method(row: pd.Series) -> str:
    run_group = str(row.get("run_group_name", "") or "")
    gnn_layers = str(row.get("gnn_layer_types", "") or "")

    if run_group.startswith("bridge_"):
        layers = gnn_layers.replace(",", "+") if gnn_layers else run_group.replace("bridge_", "")
        return f"Bridge-{layers}"

    if run_group.startswith("seq_only"):
        return "SBert"

    # if not gnn_layers:
    #     return "SBert"

    lowered_layers = gnn_layers.lower()
    if "tokenxat" in lowered_layers:
        return "SBert + TokenXAT"

    if "transformerconv" in lowered_layers or "transformerconv" in run_group.lower():
        return "SBert + TransformerConv"

    base = gnn_layers.split(",")[0].strip()
    mapping = {
        "GATV2": "GATV2",
        "GAT": "GAT",
        "GCN": "GCN",
        "TRANSFORMERCONV": "TransformerConv",
        "TOKENXAT": "TokenXAT",
    }
    pretty = mapping.get(base.upper(), base.title())
    label = f"SBert + {pretty}".strip()
    return label



def summarize_runs(
    df: pd.DataFrame,
    method_builder: MethodBuilder,
    metric_labels: Dict[str, str] = METRIC_LABELS,
    target_priority: Iterable[str] = VAL_SELECTION_PRIORITY,
    seed_bounds: Optional[Tuple[int, int]] = (0, 100),
    keep_hparams: Sequence[str] = DEFAULT_HPARAM_COLUMNS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = df.copy()
    if "seed" not in work.columns:
        work["seed"] = -1
    work["seed"] = pd.to_numeric(work["seed"], errors="coerce").fillna(-1).astype(int)

    if seed_bounds is not None:
        lower, upper = seed_bounds
        work = work[(work["seed"] >= lower) & (work["seed"] <= upper)]
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()

    if "run_group_name" not in work.columns:
        work["run_group_name"] = work.get("name", work.index).astype(str)

    metric_cols = [col for col in work.columns if col.startswith(("val_", "test_"))]
    for col in metric_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    hyperparam_cols = sorted(
        set(work.columns)
        - set(metric_cols)
        - {"seed", "name", "run_group_name", "target_metric_key", "source_path", "source_dir"}
    )

    agg_map: Dict[str, Any] = {col: ["mean", "std"] for col in metric_cols}
    for col in hyperparam_cols:
        agg_map[col] = "first"
    agg_map["seed"] = "count"

    grouped = work.groupby("run_group_name").agg(agg_map).reset_index()
    grouped.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns.values
    ]
    if "seed_count" in grouped.columns:
        grouped.rename(columns={"seed_count": "Seeds Averaged"}, inplace=True)

    for col in hyperparam_cols:
        first_key = f"{col}_first"
        if first_key in grouped.columns:
            grouped.rename(columns={first_key: col}, inplace=True)

    min_seed_df = grouped[grouped.get("Seeds Averaged", 0) > 1].copy()
    if min_seed_df.empty:
        return pd.DataFrame(), grouped

    min_seed_df["Method"] = min_seed_df.apply(method_builder, axis=1)
    selection_col, maximize = _select_selection_column(min_seed_df, target_priority)
    if selection_col is None:
        print("Warning: No validation metric available for selection; returning aggregated data only.")
        return pd.DataFrame(), grouped

    reducer = min_seed_df.groupby("Method")[selection_col]
    idx = reducer.idxmax() if maximize else reducer.idxmin()
    best_configs = min_seed_df.loc[idx].copy()

    # final_df = best_configs.rename(columns=metric_labels)
    mean_map, std_map = _resolve_metric_label_maps(best_configs, metric_labels)
    final_df = best_configs.rename(columns={**mean_map, **std_map})
    std_labels = {
        original.replace("_mean", "_std"): f"{label}_std"
        for original, label in metric_labels.items()
        if original.endswith("_mean")
    }
    final_df.rename(columns=std_labels, inplace=True)

    keep_cols = ["Method"]
    if "Seeds Averaged" in final_df.columns:
        keep_cols.append("Seeds Averaged")
    keep_cols.extend(metric_labels.values())
    keep_cols.extend(std_labels.values())

    for col in keep_hparams:
        if col in final_df.columns and col not in keep_cols:
            keep_cols.append(col)

    for col in keep_cols:
        if col not in final_df.columns:
            final_df[col] = np.nan

    final_df = final_df[keep_cols].sort_values("Method").reset_index(drop=True)
    return final_df, grouped.reset_index(drop=True)


def process_gnn_results_from_logs(
    log_directories: Iterable[str],
    seed_bounds: Tuple[int, int] = (0, 100),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = _load_sources(log_directories, allow_csv_fallback=False)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return summarize_runs(
        combined,
        method_builder=_build_gnn_method,
        metric_labels=METRIC_LABELS,
        target_priority=VAL_SELECTION_PRIORITY,
        seed_bounds=seed_bounds,
        keep_hparams=DEFAULT_HPARAM_COLUMNS,
    )


def process_gnn_results(filenames=["static-graph-fraud-electronics.csv"], model_type='Static Graph Methods'):
    if isinstance(filenames, str):
        filenames = [filenames]

    df_list = []
    for filename in filenames:
        try:
            df_part = pd.read_csv(filename)
            print(f"Successfully loaded {len(df_part)} runs from '{filename}'.")
            df_list.append(df_part)
        except FileNotFoundError:
            print(f"Warning: Could not find '{filename}'. Skipping this file.")
        except EmptyDataError:
            print(f"Warning: '{filename}' is empty. Skipping this file.")
        except Exception as e:
            print(f"Warning: Failed to read '{filename}': {e}")

    if not df_list:
        # always return a tuple to avoid unpack errors
        return pd.DataFrame(), pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    # choose target validation metric (prefer last then mean)
    target_metric = 'val_max_f1_last' if 'val_max_f1_last' in df.columns else 'val_max_f1_mean'
    if target_metric not in df.columns:
        print(f"Error: Key metric '{target_metric}' not found.")
        print(df.columns)
        return pd.DataFrame(), pd.DataFrame()

    # keep only runs with a valid validation metric
    valid_df = df[df[target_metric].notna()].copy()
    # coerce seed
    valid_df['seed'] = pd.to_numeric(valid_df['seed'], errors='coerce').fillna(-1).astype(int)
    filtered_df = valid_df[valid_df['seed'].between(0, 5)].copy()
    print(f"Found {len(filtered_df)} valid GNN-only runs with seeds 0-5.")

    if filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if 'name' in filtered_df.columns:
        filtered_df['run_group_name'] = filtered_df['name'].str.replace(r'_seed\d+', '', regex=True)
    else:
        print("Error: 'name' column not found in GNN results.")
        return pd.DataFrame(), pd.DataFrame()

    # identify metric and hyperparam columns
    metric_cols = [col for col in filtered_df.columns if col.startswith(('val_', 'test_'))]
    hyperparam_cols = [col for col in filtered_df.columns if col not in metric_cols + ['seed', 'name', 'run_group_name']]

    # Ensure metric columns are numeric so mean/std work correctly
    for c in metric_cols:
        filtered_df[c] = pd.to_numeric(filtered_df[c], errors='coerce')

    # aggregation functions: mean and std for metrics, first for hyperparams, count seeds
    agg_funcs = {col: ['mean', 'std'] for col in metric_cols}
    agg_funcs.update({col: 'first' for col in hyperparam_cols})
    agg_funcs['seed'] = 'count'

    averaged_df = filtered_df.groupby('run_group_name', dropna=False).agg(agg_funcs).reset_index()
    # flatten multiindex columns
    averaged_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col for col in averaged_df.columns.values]
    # rename seed_count for readability
    averaged_df.rename(columns={'seed_count': 'Seeds Averaged', 'run_group_name_': 'run_group_name'}, inplace=True)

    # normalize hyperparam first-* back to original names if present
    for col in hyperparam_cols:
        first_col = f'{col}_first'
        if first_col in averaged_df.columns:
            averaged_df.rename(columns={first_col: col}, inplace=True)

    # filter groups with at least one seed
    min_seed_df = averaged_df[averaged_df['Seeds Averaged'] > 1].copy()
    if min_seed_df.empty:
        return pd.DataFrame(), averaged_df

    # Create standardized Method column BEFORE selecting best per method
    # run_group_name may be present as column name; ensure we reference correctly
    rg_col = 'run_group_name'
    if rg_col not in min_seed_df.columns:
        # try to find the column if flattening created a variant
        candidates = [c for c in min_seed_df.columns if c.lower().startswith('run_group')]
        rg_col = candidates[0] if candidates else rg_col

    min_seed_df['Method'] = min_seed_df[rg_col].astype(str).str.split('_').str[0].str.upper()
    min_seed_df['Method'] = min_seed_df['Method'].replace('TRANSFORMERCONV', 'TransformerConv')
    min_seed_df['Method'] = min_seed_df['Method'].replace('DYREP', 'DyRep')

    # best per Method according to the chosen validation metric mean
    sel_col = f'{target_metric}_mean'
    if sel_col not in min_seed_df.columns:
        print(f"Selection column {sel_col} missing in aggregated data. Available cols: {min_seed_df.columns.tolist()}")
        return pd.DataFrame(), averaged_df

    best_idx = min_seed_df.groupby('Method')[sel_col].idxmax()
    best_configs = min_seed_df.loc[best_idx].copy()
    # breakpoint()

    # At this point best_configs contains *_mean and *_std columns for metric_cols
    # Prepare a user-friendly final DataFrame that includes both mean and std for test metrics
    # We'll keep Method and Seeds Averaged plus test means/stds (if present)

    # Identify test mean and std columns
    test_mean_cols = [c for c in best_configs.columns if c.startswith('test_') and c.endswith('_mean')]
    test_std_cols = [c for c in best_configs.columns if c.startswith('test_') and c.endswith('_std')]

    # Build mapping from mean col -> display name
    metric_mapping = {c: c.replace('_mean', '') for c in test_mean_cols}
    # std mapping
    std_mapping = {c: c for c in test_std_cols}

    # Final columns to keep
    keep_cols = ['Method', 'Seeds Averaged'] + test_mean_cols + test_std_cols
    for c in keep_cols:
        if c not in best_configs.columns:
            best_configs[c] = np.nan

    final_df = best_configs[keep_cols].copy()

    # Optionally, rename mean columns to user-friendly names (e.g., test_max_f1_mean -> test_max_f1)
    rename_map = {m: metric_mapping[m] for m in test_mean_cols}
    # keep std names as-is or append _std label if you prefer
    final_df = final_df.rename(columns=rename_map)

    # Return (best per Method with means+stds, full averaged per run_group_name)
    return final_df.reset_index(drop=True), averaged_df.reset_index(drop=True)


# def process_sbert_results(filenames=["fraud-electronics-sweep2.csv"]):
#     if isinstance(filenames, str):
#         filenames = [filenames]

#     df_list = []
#     for filename in filenames:
#         try:
#             df_part = pd.read_csv(filename)
#             print(f"Successfully loaded {len(df_part)} runs from '{filename}'.")
#             df_list.append(df_part)
#         except Exception as e:
#             print(e)
#             print(f"Warning: Could not parse '{filename}'. Skipping this file.")

#     if not df_list:
#         print("Error: No valid files were loaded. Returning empty DataFrame.")
#         return pd.DataFrame(), pd.DataFrame()

#     df = pd.concat(df_list, ignore_index=True)

#     # ---- pick by VALIDATION metric; adjust to your CSVs ('val_max_f1' vs 'val_max_f1_best')
#     target_metric = 'val_max_f1_last' if 'val_max_f1_last' in df.columns else 'val_max_f1'
#     if target_metric not in df.columns:
#         print(f"Error: Key metric '{target_metric}' not found in the combined data.")
#         print(df.columns)
#         return pd.DataFrame(), pd.DataFrame()

#     # ---- basic filtering
#     valid_df = df[df[target_metric].notna()].copy()
#     valid_df['seed'] = pd.to_numeric(valid_df['seed'], errors='coerce')
#     valid_df = valid_df.dropna(subset=['seed'])
#     valid_df['seed'] = valid_df['seed'].astype(int)
#     filtered_df = valid_df[valid_df['seed'].between(0, 5)].copy()
#     if filtered_df.empty:
#         return pd.DataFrame(), pd.DataFrame()

#     # ---- create run_group_name EARLY (needed for grouping below)
#     if 'name' not in filtered_df.columns:
#         print("Error: 'name' column not found in SBert results.")
#         return pd.DataFrame(), pd.DataFrame__()
#     filtered_df['run_group_name'] = filtered_df['name'].str.replace(r'_seed\d+$', '', regex=True)

#     # ---- coerce all metrics to numeric
#     metric_cols = [c for c in filtered_df.columns if c.startswith(('val_', 'test_'))]
#     for c in metric_cols:
#         filtered_df[c] = pd.to_numeric(filtered_df[c], errors='coerce')

#     # ---- prefer the final row that actually has test_* (per run_group_name, seed)
#     test_cols = [c for c in metric_cols if c.startswith('test_')]

#     def pick_final_with_test(g):
#         has_test = g[test_cols].notna().any(axis=1) if test_cols else pd.Series(False, index=g.index)
#         if has_test.any():
#             return g.loc[has_test].tail(1)
#         return g.tail(1)

#     filtered_df = (
#         filtered_df
#         .sort_values(by=['run_group_name', 'seed', 'name'])
#         .groupby(['run_group_name', 'seed'], as_index=False, group_keys=False)
#         .apply(pick_final_with_test)
#     )

#     # ---- aggregate per run_group_name
#     hyperparam_cols = [c for c in filtered_df.columns
#                        if c not in metric_cols + ['seed', 'name', 'run_group_name', '_runtime', '_timestamp', '_step']]
#     agg_funcs = {c: ['mean', 'std'] for c in metric_cols}
#     for c in hyperparam_cols:
#         agg_funcs[c] = 'first'
#     agg_funcs['seed'] = 'count'

#     averaged_df = filtered_df.groupby('run_group_name', dropna=False).agg(agg_funcs).reset_index()
#     averaged_df.columns = ['_'.join(t).strip('_') if isinstance(t, tuple) and t[1] else t[0]
#                            for t in averaged_df.columns.values]
#     averaged_df = averaged_df.rename(columns={'seed_count': 'Seeds Averaged'})

#     # ---- normalize hyperparam name columns back from *_first
#     for c in hyperparam_cols:
#         fc = f'{c}_first'
#         if fc in averaged_df.columns:
#             averaged_df = averaged_df.rename(columns={fc: c})

#     # ---- build Method (robust, case-insensitive)
#     averaged_df['gnn_layer_types'] = averaged_df.get('gnn_layer_types', pd.Series(index=averaged_df.index)).fillna('None')
#     base_gnn = averaged_df['gnn_layer_types'].astype(str).str.split(',').str[0]

#     averaged_df['Method'] = 'SBert + ' + base_gnn
#     averaged_df.loc[averaged_df['gnn_layer_types'].str.contains('tokenxat', case=False, na=False), 'Method'] = 'SBert + TokenXAT'
#     averaged_df.loc[averaged_df['run_group_name'].str.contains('transformerconv-transformerconv', case=False, na=False), 'Method'] = 'SBert + TransformerConv'
#     averaged_df.loc[averaged_df['run_group_name'].str.startswith('seq_only', na=False), 'Method'] = 'SBert'
#     averaged_df['Method'] = (averaged_df['Method'].astype(str)
#                              .str.replace(r'(?i)\btransformerconv\b', 'TransformerConv', regex=True)
#                              .str.replace(r'(?i)\btokenxat\b', 'TokenXAT', regex=True)
#                              .str.replace(r'(?i)\bgcn\b', 'GCN', regex=True)
#                              .str.replace(r'(?i)\bgat\b', 'GAT', regex=True)
#                              .str.strip())

#     bridge_from_rg = averaged_df['run_group_name'].astype(str).map(_map_bridge_method_from_text)
#     bridge_from_m  = averaged_df['Method'].astype(str).map(_map_bridge_method_from_text)
#     # prefer explicit run_group_name mapping; fallback to current Method text
#     bridge_final   = bridge_from_rg.combine_first(bridge_from_m)
#     mask_bridge    = bridge_final.notna()
#     averaged_df.loc[mask_bridge, 'Method'] = bridge_final[mask_bridge]
#     # ---- select best per Method by VALIDATION metric, report TEST metrics
#     min_seed_df = averaged_df[averaged_df['Seeds Averaged'] >= 1].copy()
#     if min_seed_df.empty:
#         return pd.DataFrame(), averaged_df

#     sel_col = f'{target_metric}_mean'  # e.g., 'val_max_f1_mean'
#     if sel_col not in min_seed_df.columns:
#         print(f"Selection column {sel_col} missing. Available: {min_seed_df.columns.tolist()}")
#         return pd.DataFrame(), averaged_df

#     # SBert-only best
#     sbert_only_df = min_seed_df[min_seed_df['Method'] == 'SBert'].copy()
#     best_sbert = pd.DataFrame()
#     if not sbert_only_df.empty:
#         best_sbert = sbert_only_df.loc[[sbert_only_df[sel_col].idxmax()]]

#     # Hybrids best per method
#     other_df = min_seed_df[min_seed_df['Method'] != 'SBert'].copy()
#     best_other = pd.DataFrame()
#     if not other_df.empty:
#         best_other = other_df.loc[other_df.groupby('Method')[sel_col].idxmax()]

#     best_configs = pd.concat([best_sbert, best_other], ignore_index=True)

#     # ---- rename TEST means to final labels
#     metric_mapping = {
#         'test_max_f1_mean': 'Max F1',
#         'test_pr_auc_mean': 'PR AUC',
#         'test_precision_at_10_percent_mean': 'Precision@10%',
#         'test_recall_at_10_percent_mean': 'Recall@10%',
#     }
#     std_metric_mapping = {k.replace('_mean', '_std'): v + '_std' for k, v in metric_mapping.items()}

#     final_df = best_configs.rename(columns={**metric_mapping, **std_metric_mapping})
#     final_cols = ['Method', 'Seeds Averaged'] + list(metric_mapping.values()) + list(std_metric_mapping.values())
#     for c in final_cols:
#         if c not in final_df.columns:
#             final_df[c] = np.nan

#     return final_df[final_cols], averaged_df


def process_sbert_results(filenames=["fraud-electronics-sweep2.csv"]):
    if isinstance(filenames, str):
        filenames = [filenames]

    df_list = []
    for filename in filenames:
        try:
            df_part = pd.read_csv(filename)
            print(f"Successfully loaded {len(df_part)} runs from '{filename}'.")
            df_list.append(df_part)
        except Exception as e:
            print(e)
            print(f"Warning: Could not parse '{filename}'. Skipping this file.")

    if not df_list:
        print("Error: No valid files were loaded. Returning empty DataFrame.")
        return pd.DataFrame(), pd.DataFrame()

    df = pd.concat(df_list, ignore_index=True)

    # ---- choose validation metric
    target_metric = 'val_max_f1_last' if 'val_max_f1_last' in df.columns else 'val_max_f1'
    if target_metric not in df.columns:
        print(f"Error: Key metric '{target_metric}' not found in the combined data.")
        print(df.columns)
        return pd.DataFrame(), pd.DataFrame()

    # ---- basic filtering
    valid_df = df[df[target_metric].notna()].copy()
    valid_df['seed'] = pd.to_numeric(valid_df['seed'], errors='coerce')
    valid_df = valid_df.dropna(subset=['seed'])
    valid_df['seed'] = valid_df['seed'].astype(int)
    filtered_df = valid_df[valid_df['seed'].between(0, 5)].copy()
    if filtered_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    # ---- create run_group_name early
    if 'name' not in filtered_df.columns:
        print("Error: 'name' column not found in SBert results.")
        return pd.DataFrame(), pd.DataFrame()  # FIXED: no double underscore

    filtered_df['run_group_name'] = filtered_df['name'].str.replace(r'_seed\d+$', '', regex=True)

    # ---- coerce all val_/test_ metrics to numeric
    metric_cols = [c for c in filtered_df.columns if c.startswith(('val_', 'test_'))]
    for c in metric_cols:
        filtered_df[c] = pd.to_numeric(filtered_df[c], errors='coerce')

    # ---- prefer the final row that actually has test_* per (run_group_name, seed)
    test_cols = [c for c in metric_cols if c.startswith('test_')]
    val_cols  = [c for c in metric_cols if c.startswith('val_')]

    # breakpoint()
    def pick_final_with_primary(g: pd.DataFrame) -> pd.DataFrame:
        # Any non-NaN test metric in this group's rows?
        has_test_any = g[test_cols].notna().any(axis=1) if test_cols else pd.Series(False, index=g.index)
        if has_test_any.any():
            # keep the last row among those that actually have a test_* value
            return g.loc[has_test_any].tail(1)

        # Else fall back to any non-NaN val metric
        has_val_any = g[val_cols].notna().any(axis=1) if val_cols else pd.Series(False, index=g.index)
        if has_val_any.any():
            return g.loc[has_val_any].tail(1)

        # If neither test nor val exists (all NaN), just take the last row
        return g.tail(1)
    def pick_final_with_test(g: pd.DataFrame) -> pd.DataFrame:
        if test_cols:
            has_test_any = g[test_cols].notna().any(axis=1)
            if has_test_any.any():
                return g.loc[has_test_any].tail(1)
        return g.tail(1)

    filtered_df = (
        filtered_df
        .sort_values(by=['run_group_name', 'seed', 'name'])
        .groupby(['run_group_name', 'seed'], as_index=False, group_keys=False)
        .apply(pick_final_with_primary)
    )

    # ---- aggregate per run_group_name
    hyperparam_cols = [c for c in filtered_df.columns
                       if c not in metric_cols + ['seed', 'name', 'run_group_name', '_runtime', '_timestamp', '_step']]
    agg_funcs = {c: ['mean', 'std'] for c in metric_cols}
    for c in hyperparam_cols:
        agg_funcs[c] = 'first'
    agg_funcs['seed'] = 'count'

    averaged_df = filtered_df.groupby('run_group_name', dropna=False).agg(agg_funcs).reset_index()
    averaged_df.columns = [
        '_'.join(t).strip('_') if isinstance(t, tuple) and t[1] else t[0]
        for t in averaged_df.columns.values
    ]
    averaged_df = averaged_df.rename(columns={'seed_count': 'Seeds Averaged'})

    # ---- normalize hyperparam names back from *_first
    for c in hyperparam_cols:
        fc = f'{c}_first'
        if fc in averaged_df.columns:
            averaged_df = averaged_df.rename(columns={fc: c})

    # ---- build Method (robust, case-insensitive)
    averaged_df['gnn_layer_types'] = averaged_df.get('gnn_layer_types', pd.Series(index=averaged_df.index)).fillna('None')
    base_gnn = averaged_df['gnn_layer_types'].astype(str).str.split(',').str[0]

    averaged_df['Method'] = 'SBert + ' + base_gnn
    averaged_df.loc[
        averaged_df['gnn_layer_types'].str.contains('tokenxat', case=False, na=False),
        'Method'
    ] = 'SBert + TokenXAT'
    averaged_df.loc[
        averaged_df['run_group_name'].str.contains('transformerconv-transformerconv', case=False, na=False),
        'Method'
    ] = 'SBert + TransformerConv'
    averaged_df.loc[
        averaged_df['run_group_name'].str.startswith('seq_only', na=False),
        'Method'
    ] = 'SBert'

    averaged_df['Method'] = (
        averaged_df['Method'].astype(str)
        .str.replace(r'(?i)\btransformerconv\b', 'TransformerConv', regex=True)
        .str.replace(r'(?i)\btokenxat\b', 'TokenXAT', regex=True)
        .str.replace(r'(?i)\bgcn\b', 'GCN', regex=True)
        .str.replace(r'(?i)\bgat\b', 'GAT', regex=True)
        .str.strip()
    )

    # ---- force canonical Bridge names (prefer run_group_name → fallback Method)
    bridge_from_rg = averaged_df['run_group_name'].astype(str).map(_map_bridge_method_from_text)
    bridge_from_m  = averaged_df['Method'].astype(str).map(_map_bridge_method_from_text)
    bridge_final   = bridge_from_rg.combine_first(bridge_from_m)
    mask_bridge    = bridge_final.notna()
    averaged_df.loc[mask_bridge, 'Method'] = bridge_final[mask_bridge]


    # ---- select best per Method by VALIDATION metric (report TEST metrics)
    min_seed_df = averaged_df[averaged_df['Seeds Averaged'] > 1].copy()
    if min_seed_df.empty:
        return pd.DataFrame(), averaged_df

    sel_col = f'{target_metric}_mean'  # e.g., 'val_max_f1_mean'
    if sel_col not in min_seed_df.columns:
        print(f"Selection column {sel_col} missing. Available: {min_seed_df.columns.tolist()}")
        return pd.DataFrame(), averaged_df

    # SBert-only best row (optional special-casing)
    sbert_only_df = min_seed_df[min_seed_df['Method'] == 'SBert'].copy()
    best_sbert = sbert_only_df.loc[[sbert_only_df[sel_col].idxmax()]] if not sbert_only_df.empty else pd.DataFrame()

    # Other methods: best per Method
    other_df = min_seed_df[min_seed_df['Method'] != 'SBert'].copy()
    best_other = other_df.loc[other_df.groupby('Method')[sel_col].idxmax()] if not other_df.empty else pd.DataFrame()

    best_configs = pd.concat([best_sbert, best_other], ignore_index=True)

    # ---- CHANGED: resolve display columns from test_* (preferred) or val_* (fallback)
    mean_map, std_map = _resolve_metric_label_maps(best_configs, METRIC_LABELS)
    final_df = best_configs.rename(columns={**mean_map, **std_map})

    # Build the final column order dynamically based on what we actually resolved
    resolved_labels = list(METRIC_LABELS.values())  # ['Max F1', 'PR AUC', 'Precision@10%', 'Recall@10%']
    resolved_std_labels = [f"{l}_std" for l in resolved_labels]

    keep_cols = ["Method"]
    if "Seeds Averaged" in final_df.columns:
        keep_cols.append("Seeds Averaged")
    keep_cols += [c for c in resolved_labels if c in final_df.columns]
    keep_cols += [c for c in resolved_std_labels if c in final_df.columns]

    for c in keep_cols:
        if c not in final_df.columns:
            final_df[c] = np.nan

    final_df = final_df[keep_cols].reset_index(drop=True)
    # breakpoint()
    return final_df, averaged_df.reset_index(drop=True)


# def process_sbert_results_from_logs(
#     log_directories: Iterable[str],
#     seed_bounds: Tuple[int, int] = (0, 100),
# ) -> Tuple[pd.DataFrame, pd.DataFrame]:
#     frames = _load_sources(log_directories, allow_csv_fallback=True)
#     if not frames:
#         return pd.DataFrame(), pd.DataFrame()
#     combined = pd.concat(frames, ignore_index=True)
#     return summarize_runs(
#         combined,
#         method_builder=_build_sbert_method,
#         metric_labels=METRIC_LABELS,
#         target_priority=VAL_SELECTION_PRIORITY,
#         seed_bounds=seed_bounds,
#         keep_hparams=DEFAULT_HPARAM_COLUMNS,
#     )



def _build_sbert_method_with_bridge(row: pd.Series) -> str:
    """
    Wraps your existing _build_sbert_method, but overrides to canonical Bridge names
    when the run_group_name (or the initial method) indicates a Bridge run.
    """
    base = _build_sbert_method(row)  # uses your current logic (SBert, SBert+GAT, etc.)
    # Prefer mapping from run_group_name; fallback to the already-built method label
    via_rg = _map_bridge_method_from_text(str(row.get("run_group_name", "")))
    via_m  = _map_bridge_method_from_text(str(base))
    return via_rg or via_m or base

def _canonicalize_bridge_methods_inplace(df: pd.DataFrame) -> None:
    """
    Safety net: if any Bridge-like labels remain unnormalized in 'Method',
    canonicalize them in-place based on 'run_group_name' and 'Method'.
    """
    if df.empty or "Method" not in df.columns:
        return
    rg = df.get("run_group_name", pd.Series(index=df.index, dtype=object)).astype(str)
    bridge_from_rg = rg.map(_map_bridge_method_from_text)
    bridge_from_m  = df["Method"].astype(str).map(_map_bridge_method_from_text)
    bridge_final   = bridge_from_rg.combine_first(bridge_from_m)
    mask           = bridge_final.notna()
    if mask.any():
        df.loc[mask, "Method"] = bridge_final[mask]

def process_sbert_results_from_logs(
    log_directories: Iterable[str],
    seed_bounds: Tuple[int, int] = (0, 100),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames = _load_sources(log_directories, allow_csv_fallback=True)
    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Use the bridge-aware method builder so grouping/selection is done under canonical names.
    final_df, averaged_df = summarize_runs(
        combined,
        method_builder=_build_sbert_method_with_bridge,  # <— key change
        metric_labels=METRIC_LABELS,
        target_priority=VAL_SELECTION_PRIORITY,
        seed_bounds=seed_bounds,
        keep_hparams=DEFAULT_HPARAM_COLUMNS,
    )

    # Extra safety: canonicalize any leaked Bridge variants in both outputs
    _canonicalize_bridge_methods_inplace(final_df)
    _canonicalize_bridge_methods_inplace(averaged_df)

    return final_df, averaged_df

def process_sbert_results_from_csv(
    filenames: Iterable[str],
    seed_bounds: Tuple[int, int] = (0, 5),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    frames: List[pd.DataFrame] = []
    for filename in filenames:
        try:
            df = pd.read_csv(filename)
            frames.append(df)
            print(f"Loaded {len(df)} rows from '{filename}'.")
        except Exception as exc:
            print(f"Warning: Failed to read '{filename}': {exc}")

    if not frames:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    return summarize_runs(
        combined,
        method_builder=_build_sbert_method,
        metric_labels=METRIC_LABELS,
        target_priority=VAL_SELECTION_PRIORITY,
        seed_bounds=seed_bounds,
        keep_hparams=DEFAULT_HPARAM_COLUMNS,
    )

def generate_three_dataset_table(
    data_by_dataset: dict,
    order=("Amazon-Movies", "Amazon-Electronics", "Amazon-Clothing"),
    sections=(
        ("Graph Only (Static)", ["GCN", "GAT", "TransformerConv"]),
        ("Temporal Graphs", ["TGN", "DyRep"]),
        ("Sequence Only (Review Text Embedding)", ["SBert", "SBert Seq"]),
        (r"\ourmodel", [r"\ourmodel-GCN", r"\ourmodel-GAT", r"\ourmodel-TransformerConv"]),
        ("Relational Deep Learning (RDL)", ["RelGT (full)", "RelGT (matched)", "RelGNN"]),
    ),
    metrics=("Max F1", "PR AUC"),
    decimals=1,  # one decimal place
):
    def _bridge_to_ourmodel(m):
        # Fold end-to-end BRIDGE runs into the \ourmodel block; drop the retired TokenXAT variant.
        ml = str(m).lower()
        if "tokenxat" in ml:
            return None
        if ml.startswith(("bridge-gcn", "bridge_gcn")):
            return r"\ourmodel-GCN"
        if ml.startswith(("bridge-gat", "bridge_gat")):
            return r"\ourmodel-GAT"
        if ml.startswith(("bridge-transformerconv", "bridge_transformerconv")):
            return r"\ourmodel-TransformerConv"
        return m

    def _first_present(name_list, available):
        for n in name_list:
            if n in available:
                return n
        return None

    # Build normalized per-dataset frames and per-dataset best means (for bolding)
    ds_frames = {}
    per_ds_bests = {}
    all_methods = set()
    for ds in order:
        df = data_by_dataset.get(ds, pd.DataFrame())
        if df is None or df.empty:
            ds_frames[ds] = pd.DataFrame().set_index(pd.Index([], name="Method"))
            per_ds_bests[ds] = {m: float("nan") for m in metrics}
            continue
        work = df.copy()
        work["Method"] = work["Method"].astype(str).map(_bridge_to_ourmodel)
        work = work[work["Method"].notna()]   # drop retired TokenXAT rows
        work = work.drop_duplicates(subset="Method", keep="first").set_index("Method")
        ds_frames[ds] = work
        per_ds_bests[ds] = {}
        for m in metrics:
            per_ds_bests[ds][m] = pd.to_numeric(work.get(m, pd.Series(dtype=float)), errors="coerce").max(skipna=True)
        all_methods.update(work.index.astype(str).tolist())

    # Derive dynamic method groups present in any dataset
    two_stage_methods = sorted([m for m in all_methods if m.endswith("(Two-Stage)")])
    bridge_methods = sorted([m for m in all_methods if m.startswith("Bridge-")])

    # Dynamic cmidrules
    cmids = "".join(
        rf"\cmidrule(lr){{{2*i}-{2*i+1}}}"
        for i in range(1, len(order) + 1)
    )

    lines = []
    lines += [
        r"\begin{table*}[ht!]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\caption{Fraud detection performance across three Amazon categories. "
        r"Values are mean$\pm$std over three seeds (\%); higher is better. All methods share one "
        r"canonical 60/20/20 user split; PR-AUC is $\mathrm{auc}(\mathrm{recall},\mathrm{precision})$ "
        r"and every method is reported at its validation-PR-AUC-selected checkpoint. "
        r"`--' marks a method not run on a dataset.}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{l" + "cc" * len(order) + "}",
        r"\toprule",
        " & ".join(
            [""] + [rf"\multicolumn{{2}}{{c}}{{\textbf{{{ds}}}}}" for ds in order]
        ) + r" \\",
        cmids,
        r"\textbf{Method} & " + " & ".join([r"\textbf{Max F1} & \textbf{PR AUC}"] * len(order)) + r" \\",
        r"\midrule",
    ]

    def row_for_method(method_label: str) -> str:
        cells = [method_label if str(method_label).startswith("\\") else _latex_escape(method_label)]
        for ds in order:
            work = ds_frames[ds]
            if work.empty or method_label not in work.index:
                cells += ["--", "--"]
                continue

            # means
            m1 = work.at[method_label, metrics[0]] if metrics[0] in work.columns else np.nan
            m2 = work.at[method_label, metrics[1]] if metrics[1] in work.columns else np.nan
            # stds (optional)
            s1 = work.at[method_label, f"{metrics[0]}_std"] if f"{metrics[0]}_std" in work.columns else np.nan
            s2 = work.at[method_label, f"{metrics[1]}_std"] if f"{metrics[1]}_std" in work.columns else np.nan

            # bold if mean equals dataset-wise max (tiny tolerance)
            def maybe_bold(cell_str, mean_val, best_val):
                if pd.notna(mean_val) and pd.notna(best_val) and abs(float(mean_val) - float(best_val)) <= 1e-12:
                    return r"\textbf{" + cell_str + "}"
                return cell_str

            cell1 = _fmt_pct_mean_std(m1, s1, places=decimals)
            cell2 = _fmt_pct_mean_std(m2, s2, places=decimals)
            cells += [
                maybe_bold(cell1, m1, per_ds_bests[ds][metrics[0]]),
                maybe_bold(cell2, m2, per_ds_bests[ds][metrics[1]]),
            ]
        return " & ".join(cells) + r" \\"

    for sec_title, method_list in sections:
        lines.append(rf"\multicolumn{{{1 + 2*len(order)}}}{{l}}{{\textbf{{\textit{{{sec_title}}}}}}} \\")
        # Dynamically populate Two-Stage and Bridge sections if requested
        if sec_title.startswith("Two-Stage"):
            current_methods = two_stage_methods
        elif sec_title.startswith("Bridge"):
            current_methods = bridge_methods
        else:
            current_methods = method_list

        for m in current_methods:
            # Only print rows that exist in at least one dataset
            if any((not ds_frames[ds].empty) and (m in ds_frames[ds].index) for ds in order):
                lines.append(row_for_method(m))
        lines.append(r"\addlinespace[0.8ex]")

    if lines[-1] == r"\addlinespace[0.8ex]":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    lines += [
        r"\end{tabular}",
        r"}",
        r"\label{tab:fraud_three}",
        r"\end{table*}",
    ]

    return "\n".join(lines)


def generate_latex_table(
    combined_df: pd.DataFrame,
    dataset: Optional[str] = None,
    metric_order: Sequence[str] = ("Max F1", "PR AUC", "Precision@10%", "Recall@10%"),
    bold_tolerance: float = 1e-12,
    decimals: int = 4,
) -> Tuple[str, pd.DataFrame]:
    """
    Create a LaTeX table string plus a DataFrame preview of the selected rows.
    Expects columns:
      - 'Method'
      - metric means: names in `metric_order`
      - metric stds:  corresponding '<metric>_std'
    """

    if combined_df.empty:
        return "", pd.DataFrame()

    df = combined_df.copy()

    # Keep only requested metrics in the given order if present
    metric_columns = [m for m in metric_order if m in df.columns]
    if not metric_columns:
        # Nothing to print
        return "", pd.DataFrame()

    # Use PR AUC to sort if present (better readability)
    if "PR AUC" in metric_columns:
        df = df.sort_values("PR AUC", ascending=False)

    # One row per method (take first if duplicates exist)
    df = df.drop_duplicates(subset="Method", keep="first").reset_index(drop=True)

    # Compute column maxima for bolding (ignore NaN)
    max_values: Dict[str, float] = {}
    for col in metric_columns:
        col_max = pd.to_numeric(df[col], errors="coerce").max(skipna=True)
        max_values[col] = col_max

    # Collect any end-to-end Bridge methods (Method starting with 'Bridge-')
    all_methods_list = df["Method"].astype(str).tolist()
    bridge_methods_all = sorted({m for m in all_methods_list if m.startswith("Bridge-")})

    # Build LaTeX
    num_cols = 1 + len(metric_columns)  # Method + metrics
    lines: List[str] = []
    lines += [
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{6pt}",
        r"\begin{tabular}{" + "l" + "c" * len(metric_columns) + "}",
        r"\toprule",
        "Method & " + " & ".join([_latex_escape(m) for m in metric_columns]) + r" \\",
        r"\midrule",
    ]

    # Assemble rows by TABLE_STRUCTURE sections (fallback to alphabetical for missing)
    used_methods = set()
    rows_for_preview: List[Dict[str, Any]] = []

    def _append_row(method_name: str):
        nonlocal rows_for_preview, used_methods
        # Try to get the row for this method; else produce placeholders
        row_mask = df["Method"].astype(str) == method_name
        if not row_mask.any():
            lines.append(_latex_escape(method_name) + " & " + " & ".join(["--"] * len(metric_columns)) + r" \\")
            # preview entry with NaNs
            rows_for_preview.append({"Section": current_section, "Method": method_name, **{m: float('nan') for m in metric_columns}})
            return

        row = df.loc[row_mask].iloc[0]
        used_methods.add(method_name)
        cells: List[str] = []
        preview_row: Dict[str, Any] = {"Section": current_section, "Method": method_name}

        for m in metric_columns:
            mean_val = row.get(m)
            std_val = row.get(f"{m}_std") if f"{m}_std" in df.columns else None
            # Store for preview DF (raw numbers)
            preview_row[m] = mean_val

            cell = _fmt_mean_std(mean_val, std_val, places=decimals)
            # Bold if this equals the column max within tolerance
            mv = pd.to_numeric(pd.Series([mean_val]), errors="coerce").iloc[0]
            if pd.notna(max_values[m]) and pd.notna(mv) and _is_close(mv, max_values[m], tol=bold_tolerance):
                # Wrap entire cell (including math) in \textbf{}
                cell = r"\textbf{" + cell + "}"
            cells.append(cell)

        rows_for_preview.append(preview_row)
        lines.append(_latex_escape(method_name) + " & " + " & ".join(cells) + r" \\")

    # 1) Known sections, in order
    for current_section, methods in TABLE_STRUCTURE.items():
        lines.append(
            rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{\textit{{{_latex_escape(current_section)}}}}}}} \\"
        )
        # For Bridge section, populate dynamically with all Bridge-* methods if present
        if current_section == "Bridge Methods (Advanced)" and bridge_methods_all:
            methods_iter = bridge_methods_all
        else:
            methods_iter = methods
        for method in methods_iter:
            _append_row(method)
        lines.append(r"\addlinespace[0.6ex]")

    # 2) Any remaining methods not covered in TABLE_STRUCTURE (optional)
    remaining = [m for m in df["Method"].astype(str).tolist() if m not in used_methods]
    if remaining:
        lines[-1] = r"\midrule"
        current_section = "Other"
        lines.append(
            rf"\multicolumn{{{num_cols}}}{{l}}{{\textbf{{\textit{{{_latex_escape(current_section)}}}}}}} \\"
        )
        for method in sorted(remaining):
            _append_row(method)
        lines.append(r"\addlinespace[0.6ex]")

    # Footer
    if lines[-1] == r"\addlinespace[0.6ex]":
        lines[-1] = r"\bottomrule"
    else:
        lines.append(r"\bottomrule")

    caption_suffix = f" on Amazon--{dataset.capitalize()}" if dataset else ""
    lines += [
        rf"\caption{{Fraud Detection{caption_suffix}. Values are mean$\pm$std over seeds; higher is better.}}",
        r"\label{tab:model_comparison}",
        r"\end{tabular}",
        r"\end{table*}",
    ]

    preview_df = pd.DataFrame(rows_for_preview)
    # Order preview columns: Section, Method, metrics
    if not preview_df.empty:
        preview_df = preview_df[["Section", "Method"] + list(metric_columns)]

    return "\n".join(lines), preview_df



def process_rdl_results_from_logs(dirs):
    """Aggregate RelGT/RelGNN LocalLogger-format logs (notebooks/logs_rdl/fraud-<ds>/) into the
    (Method, 'PR AUC', 'PR AUC_std', 'Max F1', 'Max F1_std') schema the fraud tables expect.

    Values are FRACTIONS (0-1); the table formatter multiplies by 100. RDL emits no Max F1
    (relbench metrics = average_precision + roc_auc), so 'Max F1' is NaN -> renders as '--'.
    """
    import glob as _glob
    import os

    def _label(basename):
        b = basename.lower()
        if b.startswith("relgnn"):
            return "RelGNN"
        if "fraud_default" in b:
            return "RelGT (full)"
        if b.startswith("relgt"):
            return "RelGT (matched)"
        return None

    recs: Dict[str, Dict[str, list]] = {}
    for d in dirs:
        for p in _glob.glob(os.path.join(d, "*.json")):
            label = _label(os.path.basename(p))
            if label is None:
                continue
            try:
                payload = json.loads(Path(p).read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if not isinstance(payload, list):
                continue
            tests = [e for e in payload if isinstance(e, dict) and "test_pr_auc" in e]
            if not tests:
                continue
            last = tests[-1]
            recs.setdefault(label, {"pr": [], "f1": []})
            recs[label]["pr"].append(last.get("test_pr_auc"))
            recs[label]["f1"].append(last.get("test_max_f1"))  # absent for RDL -> None

    rows = []
    for label, vals in recs.items():
        pr = [v for v in vals["pr"] if v is not None]
        f1 = [v for v in vals["f1"] if v is not None]
        if not pr:
            continue
        rows.append({
            "Method": label,
            "PR AUC": float(np.mean(pr)),
            "PR AUC_std": float(np.std(pr, ddof=1)) if len(pr) > 1 else 0.0,
            "Max F1": float(np.mean(f1)) if f1 else np.nan,
            "Max F1_std": (float(np.std(f1, ddof=1)) if len(f1) > 1 else 0.0) if f1 else np.nan,
        })
    return pd.DataFrame(rows)


def process_dataset_directly(dataset: str):
    """Convenience wrapper to process all experiment families for a dataset."""
    print(f"\n==================== Processing Dataset: {dataset} ====================")
    static_dirs = [f"logs_canonical/static-graph-fraud-{dataset}/"]  # canonical rerun only (plain + SBert static)
    temporal_dirs = [f"logs/temporal-graph-fraud-{dataset}/",
                     f"logs_tuned_rev/temporal-graph-fraud-{dataset}/",
                     f"logs_tuned-2/temporal-graph-fraud-{dataset}/"]
    sbert_dirs = [f"logs_canonical/fraud-{dataset}-rating/"]  # canonical BRIDGE rerun only

    gnn_results, _ = process_gnn_results_from_logs(static_dirs)
    temporal_results, _ = process_gnn_results_from_logs(temporal_dirs)
    sbert_results, _ = process_sbert_results_from_logs(sbert_dirs)

    # canonical rerun: static/BRIDGE come ONLY from logs_canonical above. The old wandb-CSV exports hold
    # pre-canonical-split numbers, so drop them. Temporal is spared, so keep its CSV export.
    gnn_results2 = pd.DataFrame()
    temporal_results2, _ = process_gnn_results(filenames=[f"temporal-graph-fraud-{dataset}-local.csv"])
    sbert_results2 = pd.DataFrame()
    # if dataset == 'movies':
    #     print(sbert_results2)
    #     breakpoint()
    # concate the results


    parts = [frame for frame in (gnn_results, temporal_results, sbert_results, gnn_results2, temporal_results2, sbert_results2) if not frame.empty]
    # parts = [frame for frame in (gnn_results2, temporal_results2, sbert_results2) if not frame.empty]
    # parts = [frame for frame in (gnn_results, temporal_results, sbert_results) if not frame.empty]
    if not parts:
        print(f"[{dataset}] No data available — skipping.")
        return pd.DataFrame(), ("", pd.DataFrame())

    combined_results = pd.concat(parts, ignore_index=True)
    latex_str, best_df = generate_latex_table(combined_results, dataset=dataset)
    # RDL rows are appended by get_latex_tables.py from the explicit --rdl_root (fraud_default_canon),
    # so don't also pull them from logs_rdl here (that dir mixes old + canonical RelGT logs -> double-count).
    return combined_results, (latex_str, best_df)


def main():
    # Single-dataset table for Clothing (your existing thing)
    for ds in ["movies","electronics","clothing"]:
        combined_results, (latex_str, best_df) = process_dataset_directly(ds)
        Path("./tables").mkdir(parents=True, exist_ok=True)
        with open(f"./tables/{ds}_table.tex", "w") as f:
            f.write(latex_str)
        print(latex_str)

    # Build per-dataset frames for the 3-dataset table
    movies_df, _ = process_dataset_directly("movies")        # key: Amazon-Movies
    electronics_df, _ = process_dataset_directly("electronics")  # key: Amazon-Electronics
    clothing_df, _ = process_dataset_directly("clothing")    # key: Amazon-Clothing

    table_tex = generate_three_dataset_table(
        {
            "Amazon-Movies": movies_df,
            "Amazon-Electronics": electronics_df,
            "Amazon-Clothing": clothing_df,
        },
        order=("Amazon-Movies", "Amazon-Electronics", "Amazon-Clothing"),
    )

    with open("./tables/fraud_three.tex", "w") as f:
        f.write(table_tex)


if __name__ == "__main__":
    main()