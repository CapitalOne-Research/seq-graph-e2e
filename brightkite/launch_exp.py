import subprocess
import time
import itertools
import os
import argparse
from collections import deque

# --- Configuration ---

# 1. Define the base command to run your training script.
#    The script will append the hyperparameter arguments to this.
BASE_COMMAND = "python joint.py"

# 2. Define the grid of hyperparameters you want to sweep.
#    Use the exact command-line argument as the key.
#    For boolean flags, use [True, False]. The script will add the flag if True.
HYPERPARAM_GRID = {
    '--gnn_layer_types': ['TokenXAT,GCN'],
    '--add_self_loops': [True], #best
    '--num_neighbors': [10],
    '--gcn_edge_weight': [True], #best
    '--aggregation_mode': ['fixed'],
    '--qk_norm': [False],
    '--pre_agg_proj': [False], #best
    '--post_agg_proj': [False], #best
    '--msg_mode': ['gat'],
    '--compress_dim': [2, 1],
    '--compress_first': [False],
    '--seed':[0,1,2],
    '--enable_early_stopping': [True],
    '--eval_batch_size': [256]
}


BASELINE_GRID = {
    '--gnn_layer_types': ['TransformerConv,TransformerConv', 'GCN,GCN','GAT,GAT'],
    '--num_neighbors': [10],
    '--gnn_heads': [2],
    '--seed':[0,1,2],
    '--enable_early_stopping': [True]
}


BASE_COMMAND_STATIC = "python static_graph.py"
STATIC_GRID = {
    '--model': ['TransformerConv', 'GCN','GAT'],
    '--num_neighbors': [10],
    '--heads': [2],
    '--seed':[0,1,2],
    '--enable_early_stopping': [True],
    '--lr': [1e-3]
}


# 3. GPU Configuration
#    Maximum number of GPUs you want the script to use.
MAX_GPUS_TO_USE = 8
#    Memory threshold (in MiB) to consider a GPU "free".
#    Adjust this based on your model's idle memory footprint.
GPU_FREE_MEM_THRESHOLD = 500


# --- NEW: Function to generate result filename ---
def get_filename_for_config(config):
    """
    Mirrors the filename generation logic from your joint.py script for
    both TokenXAT and baseline models.
    """
    # Use .get() to provide default values for keys that might not be in every config
    gnn_layer_types = config.get('--gnn_layer_types', 'TokenXAT,GCN') # Default for main sweep
    gnn_num_layers = config.get('--gnn_num_layers', 2)
    gnn_heads = config.get('--gnn_heads', 1)
    bert_num_layers = config.get('--bert_num_layers', 1)
    bert_num_heads = config.get('--bert_num_heads', 2)
    bert_hidden_size = config.get('--bert_hidden_size', 128)
    num_neighbors = config.get('--num_neighbors', 2)
    num_negative_samples = config.get('--num_negative_samples', 100)
    # batch_size = config.get('--batch_size', 128) 
    # lr = config.get('--lr', 1e-5)
    lr = config.get('--lr', 0.0001)
    epochs = config.get('--epochs', 100)
    debug_subset = config.get('--debug_subset', None)
    seed = config.get('--seed', 10)

    gnn_layer_config_str = gnn_layer_types.lower().replace(',', '-')
    
    # --- MODIFICATION: Check if this is a TokenXAT run or a baseline run ---

    # This block is for your main TokenXAT sweep
    add_self_loops = config.get('--add_self_loops', False)
    qk_norm = config.get('--qk_norm', False)
    pre_agg_proj = config.get('--pre_agg_proj', False)
    post_agg_proj = config.get('--post_agg_proj', False)
    aggregation_mode = config.get('--aggregation_mode', 'fixed')
    compress_first = config.get('--compress_first', False)
    compress_dim = config.get('--compress_dim', 4)
    msg_mode = config.get('--msg_mode', 'gat')
    layer_norm = config.get('--layer_norm', False)
    gcn_edge_weight = config.get('--gcn_edge_weight', False)
    encoder_compress = 'stack'
    gnn_layer_types = config.get('--gnn_layer_types', 'TokenXAT,GCN') # Default for main sweep
    gnn_num_layers = config.get('--gnn_num_layers', 2)
    gnn_heads = config.get('--gnn_heads', 1)
    bert_num_layers = config.get('--bert_num_layers', 1)
    bert_num_heads = config.get('--bert_num_heads', 2)
    bert_hidden_size = config.get('--bert_hidden_size', 128)
    num_neighbors = config.get('--num_neighbors', 2)
    num_negative_samples = config.get('--num_negative_samples', 100)
    batch_size = config.get('--batch_size', 128)
    # lr = config.get('--lr', 1e-5)
    lr = config.get('--lr', 0.0001)
    epochs = config.get('--epochs', 100)
    debug_subset = config.get('--debug_subset', None)
    seed = config.get('--seed', 10)
    if "xat" in gnn_layer_config_str:
        name = (
            f"e2e_{gnn_layer_config_str}{gnn_num_layers}x{gnn_heads}"
            f"{'_self-loop' if add_self_loops else ''}"
            f"{'_qknorm' if qk_norm else ''}"
            f"{'_pre-agg-proj' if pre_agg_proj else ''}"
            f"{'_post-agg-proj' if post_agg_proj else ''}"
            f"_{aggregation_mode}"
            f"{'_compress-first' if compress_first else ''}"
            f"_compress-dim{compress_dim}"
            f"{'_msg-mode-gcn' if msg_mode == 'gcn' else ''}"
            f"{'_layer-norm' if layer_norm else ''}"
            f"{'_gcn-edge-weight' if gcn_edge_weight else ''}"
            f"_bert{bert_num_layers}x{bert_num_heads}"
            f"_hidden-dim{bert_hidden_size}_"
            f"{encoder_compress}_nngb{num_neighbors}_nneg{num_negative_samples}_"
            f"bs{batch_size}_lr{lr}_epochs{epochs}"
        )
    else:
        # This block is for your baseline models (GCN, GAT, etc.)
        encoder_compress = 'mean' # Baselines typically use mean encoding
        name = (
            f"e2e_{gnn_layer_config_str}{gnn_num_layers}x{gnn_heads}"
            f"_bert{bert_num_layers}x{bert_num_heads}_hidden-dim{bert_hidden_size}_"
            f"{encoder_compress}_nngb{num_neighbors}_nneg{num_negative_samples}_"
            f"bs{batch_size}_lr{lr}_epochs{epochs}"
        )

    if debug_subset:
        name += f"_debug{debug_subset}"
    
    name += f"_seed{seed}"
    
    return name + ".json"

# --- Helper Functions ---

def get_free_gpu_ids():
    """
    Parses nvidia-smi to find GPUs with memory usage below a threshold.
    Returns a list of integer IDs for the free GPUs.
    """
    try:
        # Execute nvidia-smi command
        smi_query = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        result = subprocess.run(smi_query, shell=True, capture_output=True, text=True, check=True)
        
        free_gpus = []
        # Parse the output line by line
        for line in result.stdout.strip().split('\n'):
            if not line:
                continue
            index, mem_used = line.split(',')
            index, mem_used = int(index.strip()), int(mem_used.strip())
            
            if mem_used < GPU_FREE_MEM_THRESHOLD:
                free_gpus.append(index)

        # Respect the user-defined maximum number of GPUs
        return free_gpus[:MAX_GPUS_TO_USE]
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Warning: Could not run nvidia-smi. Assuming no GPUs available. Error: {e}")
        return []

def generate_run_commands(base_cmd, grid, existing_files):
    """
    Generates all possible command-line strings from the hyperparameter grid.
    """
    keys = grid.keys()
    values = grid.values()
    
    # Create all combinations of hyperparameter values
    for combination in itertools.product(*values):
        args_dict = dict(zip(keys, combination))
        # --- MODIFIED: Check if file exists before yielding command ---
        expected_filename = get_filename_for_config(args_dict)
        # if expected_filename in existing_files:
        #     # print(f"Skipping existing run: {expected_filename}")
        #     continue

        # --- MODIFIED: Apply conditional logic for the sweep ---
        # Rule: Only run --msg_mode=gcn when --compress_first=True and --compress_dim=1
        # if args_dict.get('--msg_mode') == 'gcn':
        #     if not (args_dict.get('--compress_first') is True and args_dict.get('--compress_dim') == 1):
        #         continue # Skip this invalid combination
        # --- END MODIFICATION ---
        cmd = base_cmd
        for arg, val in args_dict.items():
            # For boolean flags, only add the flag if it's True
            if isinstance(val, bool):
                if val:
                    cmd += f" {arg}"
            # For other arguments, add the argument and its value
            else:
                cmd += f" {arg} {val}"
        # cmd += f" --use_wandb"
        # breakpoint()
        if ("--num_neighbors 10" in cmd) and ('TokenXAT' not in cmd):
            if "gowalla" in cmd:   # denser graph -> smaller batch to fit 48GB
                cmd += " --eval_batch_size 64 --batch_size 32 --num_negative_samples 100"
            else:
                cmd += " --eval_batch_size 128 --batch_size 64 --num_negative_samples 100"
        if ("--num_neighbors 5" in cmd) and ('TokenXAT'  not in cmd) and ("--aggregation_mode learned" in cmd):
            cmd += " --eval_batch_size 512 --batch_size 64"
        if "joint.py" in cmd:   # resume from last checkpoint on relaunch (safe: fresh start if none)
            cmd += " --resume"
        yield cmd

# --- Main Execution Logic ---

def main():
    """
    Main function to orchestrate the hyperparameter sweep.
    """
    
    global BASE_COMMAND
    # --- MODIFIED: Setup argument parser to fix specific hyperparameters ---
    parser = argparse.ArgumentParser(description="GPU Hyperparameter Sweep Manager")
    parser.add_argument('--config', type=str, choices=['tokenxat','baseline','static','static_bert','tgn'], default='tokenxat')
    parser.add_argument("--dataset", type=str, choices=['brightkite','gowalla'], default='brightkite')

    
    # Parse only the --run_baseline flag first
    args, remaining_argv = parser.parse_known_args()
    
    if args.config == 'baseline':
        print("--- Mode: Running Baselines ---")
        active_grid = BASELINE_GRID
    elif args.config == 'tokenxat':
        print("--- Mode: Running Main Hyperparameter Sweep ---")
        active_grid = HYPERPARAM_GRID
    elif args.config == 'static':
        BASE_COMMAND = BASE_COMMAND_STATIC
        active_grid = STATIC_GRID
    elif args.config == 'tgn':
        active_grid = tgn_GRID
        
    

    # --- MODIFIED: Automatically generate parser arguments from the grid ---
    arg_map = {}
    for grid_key, grid_values in active_grid.items():
        # e.g., grid_key '--add_self_loops' becomes dest 'fix_add_self_loops'
        arg_name = grid_key.lstrip('-')
        dest_name = f"fix_{arg_name}"
        arg_flag = f"--{dest_name}"
        arg_map[dest_name] = grid_key

        value_type = type(grid_values[0])
        
        if value_type == bool:
            parser.add_argument(arg_flag, type=str, choices=['True', 'False'], dest=dest_name,
                                help=f"Fix the value for {grid_key}")
        else:
            parser.add_argument(arg_flag, type=value_type, dest=dest_name,
                                help=f"Fix the value for {grid_key}")

    args = parser.parse_args()
    args_dict = vars(args)

    # --- MODIFIED: Dynamically update the grid based on parsed arguments ---
    for dest_name, grid_key in arg_map.items():
        fixed_value = args_dict.get(dest_name)
        if fixed_value is not None:
            # Handle boolean conversion from string
            if isinstance(active_grid[grid_key][0], bool):
                active_grid[grid_key] = [fixed_value == 'True']
            else:
                active_grid[grid_key] = [fixed_value]
            print(f"Fixing hyperparameter {grid_key} = {active_grid[grid_key][0]}")
    RESULTS_DIR = f"./{args.dataset}_results_seeded/"
    # --- MODIFIED: Scan for existing results ---
    print(f"Checking for existing results in {RESULTS_DIR}...")
    try:
        existing_files = set(os.listdir(RESULTS_DIR))
        print(f"Found {len(existing_files)} existing result files.")
    except FileNotFoundError:
        existing_files = set()
        print("Results directory not found. Starting all runs from scratch.")
    
    
    print("--- Starting Hyperparameter Sweep ---")
    
    # Generate all command configurations and put them in a queue
    # BASE_COMMAND += f" --dataset {args.dataset}"
    run_commands = deque(generate_run_commands(BASE_COMMAND + f" --dataset {args.dataset}", active_grid, existing_files))
    # breakpoint()
    # for cmd in run_commands:
    #     cmd += f" --dataset {args.dataset}"
    total_runs = len(run_commands)
    print(f"Generated {total_runs} unique configurations to run.")
    
    # Dictionary to keep track of active processes on each GPU
    # Format: {gpu_id: subprocess.Popen_object}
    active_processes = {}
    completed_runs = 0
    # Loop as long as there are commands to run or processes still running
    while run_commands or active_processes:
        # --- 1. Check for and clean up finished processes ---
        finished_gpus = []
        for gpu_id, process in active_processes.items():
            # poll() returns None if the process is still running, otherwise the exit code
            if process.poll() is not None:
                print(f"GPU {gpu_id}: Run finished with exit code {process.returncode}.")
                finished_gpus.append(gpu_id)
                completed_runs += 1
        
        # Remove finished processes from the active dictionary
        for gpu_id in finished_gpus:
            del active_processes[gpu_id]
            
        # --- 2. Find free GPUs and launch new runs ---
        if run_commands:
            available_gpus = get_free_gpu_ids()
            
            # Find GPUs that are available AND not currently running a process
            gpus_to_launch_on = [gpu for gpu in available_gpus if gpu not in active_processes]

            for gpu_id in gpus_to_launch_on:
                # If there are no more commands to run, break the loop
                if not run_commands:
                    break
                    
                # Get the next command from the queue
                command_to_run = run_commands.popleft()
                if "compress-dim8" in command_to_run and "fixed" in command_to_run:
                    continue
                
                print("\n" + "="*50)
                print(f"Found free GPU: {gpu_id}")
                print(f"Launching run {completed_runs + len(active_processes) + 1}/{total_runs}")
                print(f"COMMAND: {command_to_run}")
                print("="*50 + "\n")
                
                # Set the CUDA_VISIBLE_DEVICES environment variable for the subprocess
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
                
                # Launch the process
                process = subprocess.Popen(command_to_run, shell=True, env=env)
                active_processes[gpu_id] = process
        
        # --- 3. Wait before the next cycle ---
        if run_commands:
            # Wait for a bit before checking GPUs again to avoid spamming nvidia-smi
            time.sleep(10)
        else:
            # If all commands have been launched, just wait for the last ones to finish
            time.sleep(10)

    print("\n--- Hyperparameter Sweep Complete ---")
    print(f"Successfully completed {completed_runs} runs.")

if __name__ == "__main__":
    main()