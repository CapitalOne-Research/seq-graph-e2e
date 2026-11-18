import subprocess
import itertools
import time
import os
from collections import deque
import argparse
# import wandb




# --- Configuration ---
# BASE_COMMAND = "python review_embedding.py"


# --- Configuration ---
TOKENXAT_BASE_COMMAND = "python review_embedding.py"
BASELINE_BASE_COMMAND = "python static_graph_clf.py"  # As requested
SEQ_ONLY_BASE_COMMAND = "python review_embedding.py"
TGN_BASE_COMMAND = "python temporal_graph.py"

COMMON_ARGS = {
    # "--use_wandb": "", # This makes it a flag
    "--num_neighbors": 10,
    "--log_dir": "logs_canonical",  # fresh dir for the canonical rerun; avoids false-skip vs old logs
}




TOKENXAT_GRID = {
    '--dataset': ['movies', 'electronics','clothing'],
    # '--dataset' :['musical'],
    # '--gnn_layer_types': ['TransformerConv,TransformerConv'],
    '--gnn_layer_types': ['GCN,GCN', 'GAT,GAT','TransformerConv,TransformerConv','TokenXAT,GCN', 'TokenXAT,GAT'],
    '--seed': [0,1,2],
    '--lr': [1e-4,],
    '--compress_dim': [1,2],
    # '--compress_first': [False],
    # TokenXAT specific settings can be hardcoded in the command generator
    '--compress_mode': ['learned'], 
    # '--compress_mode': ['mlp'], 
    '--aggregation_mode': ['gat'],
    '--residual': [True],
    '--batch_size': [512],
    '--bridge': [True],
    
    # '--pre_agg_proj': [True, False],
    # '--post_agg_proj': [True, False],
    # '--bridge': [True],
    
}

# Config for baseline models like GCN, GAT, etc.
BASELINE_GRID = {
    '--dataset': ['clothing','electronics','movies'],
    '--model': ['GAT', 'GCN','TransformerConv'],
    '--seed': [0,1,2],
    '--lr': [1e-3],
    '--use_sequence_embedding': [True, False],
}


# Config for temporal graph methods (unified):
#   - TGN-style memory models: TGN, DyRep (and optionally TNCN)
#   - TPNet-family temporal encoders: TPNet, DyGFormer, GraphMixer, TGAT, TCL, NAT, CTAN
# All are launched through this grid; the script (temporal_graph.py vs temporal_graph_tpnet.py)
# is chosen per-model inside generate_run_commands.
TGN_GRID = {
    '--dataset': ['movies','clothing','electronics'],
    # '--model': ['DyRep', 'TGN'],#, 'TPNet', 'DyGFormer', 'GraphMixer', 'TGAT', 'TCL', 'NAT', 'CTAN'],
    '--model': ['TGN', 'DyRep', 'TPNet', 'DyGFormer', 'GraphMixer', 'TGAT', 'TCL', 'NAT', 'CTAN'],
    '--seed': [0, 1, 2],
    '--lr': [1e-4, 1e-3],
    '--use_sequence_embedding': [True],
    '--enable_early_stopping': [True],
    '--embed_batch_size': [2048],
    '--eval_every': [5],
    '--patience': [10]
}


# Config for sequence-only models (no GNN)
SEQ_ONLY_GRID = {
    '--dataset': ['clothing', 'electronics', 'movies'],
    '--seed': [0, 1, 2],
    '--lr': [5e-5, 1e-4],
    '--seq_only': [True],
    '--bridge': [True],
}

# Clean BRIDGE grid for the fraud table (Bridge-{GCN,GAT,TfConv}); no TokenXAT, no compress_dim sweep.
BRIDGE_GRID = {
    '--dataset': ['clothing', 'electronics', 'movies'],
    '--gnn_layer_types': ['GCN,GCN', 'GAT,GAT', 'TransformerConv,TransformerConv'],
    '--seed': [0, 1, 2],
    '--lr': [1e-4],
    '--compress_dim': [1],
    '--compress_mode': ['learned'],
    '--aggregation_mode': ['gat'],
    '--residual': [True],
    '--batch_size': [512],
    '--bridge': [True],
}

# Supervised-frozen control (variant ii): same Bridge config, but the sequence encoder is warmed up
# on the task and then frozen before the graph module (isolates joint optimization from the objective).
SUPFROZEN_GRID = {
    '--dataset': ['clothing', 'electronics', 'movies'],
    '--gnn_layer_types': ['GCN,GCN', 'GAT,GAT', 'TransformerConv,TransformerConv'],
    '--seed': [0, 1, 2],
    '--lr': [1e-4],
    '--compress_dim': [1],
    '--compress_mode': ['learned'],
    '--aggregation_mode': ['gat'],
    '--residual': [True],
    '--batch_size': [512],
    '--bridge': [True],
    '--freeze_encoder': [True],
    '--encoder_warmup_epochs': [50],
}

# Clothing regularization probe (ceiling check). The seed-0 val PR-AUC curve peaks at ~0.362
# (epoch ~21) then overfits down to ~0.32, so the reported ~35 is a converged best-val number,
# NOT under-training. This grid gives BRIDGE a fairer shot on the noisiest/lowest-signal set via
# stronger regularization + lower LR. If it beats the ~35 ceiling, we must sweep RelGT-clothing on
# the SAME budget before reporting (else it is cherry-picking). Run with a distinct project dir:
#   python launch_fraud.py --config bridge_clothing_reg --suffix reg   -> logs_canonical/fraud-clothing-reg/
BRIDGE_CLOTHING_REG_GRID = {
    '--dataset': ['clothing'],
    '--gnn_layer_types': ['GCN,GCN', 'GAT,GAT', 'TransformerConv,TransformerConv'],
    '--seed': [0, 1, 2],
    '--lr': [5e-5],
    '--dropout': [0.7],
    '--tr_dropout': [0.2],
    '--compress_dim': [1],
    '--compress_mode': ['learned'],
    '--aggregation_mode': ['gat'],
    '--residual': [True],
    '--batch_size': [512],
    '--bridge': [True],
}
# --- GPU Configuration ---
# Memory threshold (in MiB) to consider a GPU "free".
GPU_FREE_MEM_THRESHOLD = 15000

def get_free_gpu_ids():
    """
    Parses nvidia-smi to find GPUs with memory usage below a threshold.
    Returns a list of integer IDs for the free GPUs.
    """
    try:
        smi_query = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(smi_query, shell=True, capture_output=True, text=True, check=True)
        
        free_gpus = []
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            index, mem_used = line.split(',')
            index, mem_used = int(index.strip()), int(mem_used.strip())
            
            if mem_used < GPU_FREE_MEM_THRESHOLD:
                free_gpus.append(index)
        return free_gpus
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not run nvidia-smi. Assuming no GPUs available. Error: {e}")
        return []


def parse_command_to_args(command_str: str) -> dict:
    """A simple parser to convert a command string back into a dictionary."""
    parts = command_str.split()
    args = {}
    # Start after the script name (e.g., "python", "review_embedding.py")
    i = 2 
    while i < len(parts):
        arg = parts[i]
        if arg.startswith('--'):
            # Handle flags (arguments without a value)
            if i + 1 >= len(parts) or parts[i+1].startswith('--'):
                args[arg] = True
                i += 1
            # Handle key-value arguments
            else:
                # Remove quotes if they exist
                value = parts[i+1].strip('\'"')
                args[arg] = value
                i += 2
        else:
            i += 1
    return args


                                
def generate_run_commands(base_cmd, grid):
    """
    Generates all possible command-line strings from a given hyperparameter grid.
    This function is now general and doesn't need special logic for different models.
    """
    keys = grid.keys()
    values = grid.values()
    
    # Create all combinations of hyperparameter values
    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))

        # Decide which base script to call (for temporal methods we may override).
        cmd = base_cmd
        model_name = params.get('--model', '')

        # Handle special parameters for TokenXAT if it's a TokenXAT run
        if 'TokenXAT' in params.get('--gnn_layer_types', ''):
            params['--add_self_loop'] = True

        # For unified temporal config: choose script by model.
        # TGN / DyRep / TNCN -> temporal_graph.py
        # TPNet-family          -> temporal_graph_tpnet.py
        if "temporal_graph.py" in base_cmd:
            if model_name in ['TGN', 'DyRep', 'TNCN']:
                cmd = "python temporal_graph.py"
            elif model_name in ['TPNet', 'DyGFormer', 'GraphMixer', 'TGAT', 'TCL', 'NAT', 'CTAN', 'CAWN']:
                cmd = "python temporal_graph_tpnet.py"

        # Adjust train/eval batch sizes for temporal models that are heavier.
        if model_name in ['GraphMixer', 'TCL', 'NAT', 'TGAT', 'TPNet', 'DyGFormer']:
            params['--train_batch_size'] = 1024
            params['--eval_batch_size'] = 4096
        elif model_name == 'CAWN':
            params['--train_batch_size'] = 128
            params['--eval_batch_size'] = 512
        if model_name in ['DyGFormer']:
            params['--embed_batch_size'] = 512

        # Some models (TGN/DyRep/TNCN) don't know about embed_batch_size; skip it for them.
        skip_args = set()
        if model_name in ['TGN', 'DyRep', 'TNCN']:
            skip_args.add('--embed_batch_size')

        # Build the command string from the dictionary
        for arg, val in params.items():
            if arg in skip_args:
                continue
            # For boolean flags, only add the flag if it's True
            if isinstance(val, bool):
                if val:
                    cmd += f" {arg}"
            # For other arguments, add the argument and its value
            else:
                # Add quotes around values with commas (like 'TokenXAT,GAT')
                cmd += f' {arg} "{val}"' if ',' in str(val) else f" {arg} {val}"
        yield cmd
                             
                             

def main():
    """
    Main function to orchestrate the hyperparameter sweep with config selection.
    """
    # --- 1. Set up Argument Parser ---

    parser = argparse.ArgumentParser(description="Hyperparameter Sweep Manager")
    parser.add_argument(
        '--config', 
        type=str, 
        default='tokenxat', 
        choices=['tokenxat', 'bridge', 'baseline', 'seq_only', 'tgn', 'tpnet_temporal', 'supfrozen', 'bridge_clothing_reg'],
        help="Select the configuration grid to run."
    )
    parser.add_argument(
        '--suffix', 
        type=str, 
        default='', 
        help="suffix to wandb project"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=['musical', 'movies', 'clothing', 'electronics'], 
        default='movies',
        help="The name of the dataset to fetch runs for."
    )
    args = parser.parse_args()
    if args.suffix:
        COMMON_ARGS['--suffix'] = args.suffix
    # --- 2. Select the Active Grid based on the argument ---
    active_grid = None
    print("\n" + "="*50)
    if args.config == 'tokenxat':
        print("🚀 Mode: Running TokenXAT Sweep")
        active_grid = TOKENXAT_GRID
        base_command = TOKENXAT_BASE_COMMAND
    elif args.config == 'baseline':
        print("📊 Mode: Running Baseline GNN Sweep")
        active_grid = BASELINE_GRID
        base_command = BASELINE_BASE_COMMAND
    elif args.config == 'seq_only':
        print("📜 Mode: Running Sequence-Only Sweep")
        active_grid = SEQ_ONLY_GRID
        base_command = SEQ_ONLY_BASE_COMMAND
    elif args.config == 'bridge':
        print("Mode: Running BRIDGE Sweep")
        active_grid = BRIDGE_GRID
        base_command = "python review_embedding.py"
    elif args.config == 'supfrozen':
        print("Mode: Running Supervised-Frozen Control Sweep")
        active_grid = SUPFROZEN_GRID
        base_command = "python review_embedding.py"
    elif args.config == 'bridge_clothing_reg':
        print("Mode: Running BRIDGE Clothing Regularization Probe (ceiling check)")
        active_grid = BRIDGE_CLOTHING_REG_GRID
        base_command = "python review_embedding.py"
    elif args.config == 'tgn':
        print("⏱️ Mode: Running Temporal Graph Sweep")
        active_grid = TGN_GRID
        base_command = TGN_BASE_COMMAND
    elif args.config == 'tpnet_temporal':
        # Alias to the unified temporal grid so older commands still work.
        print("⏱️ Mode: Running Temporal Graph Sweep (tpnet_temporal alias)")
        active_grid = TGN_GRID
        base_command = TGN_BASE_COMMAND
    print("="*50 + "\n")
    config_type = args.config

    # Override dataset in the active grid (when present) with CLI choice
    # if '--dataset' in active_grid:
    #     active_grid['--dataset'] = [args.dataset]
    
    # api = wandb.Api()

    # Define project paths based on arguments
    WANDB_PROJECT_PATH = [f"fraud-{args.dataset}-{args.suffix}"]


    # --- 3. Generate Commands and Run the Sweep (existing logic) ---
    print("--- Starting Hyperparameter Sweep ---")
    
    # Generate all command configurations from the selected grid
    full_command_list = []
    
    for cmd_base in generate_run_commands(base_command, active_grid):
        final_cmd = cmd_base
        for arg, value in COMMON_ARGS.items():
            final_cmd += f" {arg} {value}"
        full_command_list.append(final_cmd)
        
    run_commands = deque(full_command_list)
    total_runs = len(run_commands)
    
    if total_runs == 0:
        print("No runs to execute for the selected configuration.")
        return
        
    print(f"Generated {total_runs} unique configurations to run.")
    
    # --- ✨ NEW: Filter out completed runs before execution ---
    print("\n--- Checking for previously completed runs in W&B ---")
    runs_to_execute = []
    for command in full_command_list[::-1]:
        runs_to_execute.append(command)

    
    # Use the filtered list for the execution queue
    run_commands = deque(runs_to_execute)
    total_runs = len(run_commands)
    
    if total_runs == 0:
        print("\n🎉 All runs for the selected configuration have already been completed!")
        return
        
    print(f"\nFound {total_runs} new configurations to run.")
    
    active_processes = {}
    completed_runs = 0
    while run_commands or active_processes:
        # --- Check for finished processes ---
        finished_gpus = []
        for gpu_id, process in active_processes.items():
            if process.poll() is not None:
                if process.returncode == 0:
                    print(f"GPU {gpu_id}: Run finished successfully.")
                else:
                    print(f"GPU {gpu_id}: Run FAILED with exit code {process.returncode}.")
                finished_gpus.append(gpu_id)
                completed_runs += 1
        
        for gpu_id in finished_gpus:
            del active_processes[gpu_id]
            
        # --- Find free GPUs and launch new runs ---
        if run_commands:
            available_gpus = get_free_gpu_ids()
            gpus_to_launch_on = [gpu for gpu in available_gpus if gpu not in active_processes]

            for gpu_id in gpus_to_launch_on:
                if not run_commands:
                    break
                    
                command_to_run = run_commands.popleft()
                
                print("\n" + "="*50)
                print(f"Found free GPU: {gpu_id}")
                print(f"Launching run {completed_runs + len(active_processes) + 1}/{total_runs}")
                print(f"COMMAND: {command_to_run}")
                print("="*50 + "\n")
                
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                process = subprocess.Popen(command_to_run, shell=True, env=env)
                active_processes[gpu_id] = process

        # --- Wait before the next cycle ---
        time.sleep(0)

    print("\n--- Hyperparameter Sweep Complete ---")
    print(f"Successfully completed {completed_runs} runs.")
    
    
if __name__ == "__main__":
    main()