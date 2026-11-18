import subprocess
import time
import itertools
import os
import argparse
from collections import deque

# --- Configuration ---

# 1. Define the base command to run your training script.
#    The script will append the hyperparameter arguments to this.
BASE_COMMAND = "python static_graph.py"


STATIC_GRID = {
    '--model': ['TransformerConv','GCN','GAT'],
    '--num_neighbors': [10],
    '--heads': [2],
    '--seed':[0,1,2],
    '--enable_early_stopping': [True],
    '--lr': [1e-3],
    '--use_sequence_embedding': [False],
    # '--seq_embedding_filename': ["./stored/user_embedding-m0.3-lr1e-3-e200.pkl"]
}


# 3. GPU Configuration
#    Maximum number of GPUs you want the script to use.
MAX_GPUS_TO_USE = 8
#    Memory threshold (in MiB) to consider a GPU "free".
#    Adjust this based on your model's idle memory footprint.
GPU_FREE_MEM_THRESHOLD = 10000


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

def generate_run_commands(base_cmd, grid, args):
    """
    Generates all possible command-line strings from the hyperparameter grid.
    """
    keys = grid.keys()
    values = grid.values()
    
    # breakpoint()
    # Create all combinations of hyperparameter values
    for combination in itertools.product(*values):
        args_dict = dict(zip(keys, combination))
        cmd = base_cmd
        for arg, val in args_dict.items():
            # For boolean flags, only add the flag if it's True
            if isinstance(val, bool):
                if val:
                    cmd += f" {arg}"
            # For other arguments, add the argument and its value
            else:
                cmd += f" {arg} {val}"
                # --- MODIFIED SECTION ---
        # If --use_sequence_embedding is True, add the corresponding filename
        if args_dict.get('--use_sequence_embedding') is True:
            # dataset = args_dict.get('--dataset')
            # if dataset: # Ensure dataset is not None
                # Construct the filename, e.g., "clothing_user_embedding.pkl"
            # embedding_filename = f"{args.dataset}_user_embedding.pkl"
            cmd += f" --seq_embedding_filename ./stored/{args.dataset}_user_embedding-m0.3-lr1e-3-e200.pkl"
            
        yield cmd

# --- Main Execution Logic ---

def main():
    """
    Main function to orchestrate the hyperparameter sweep.
    """
    
    
    # --- MODIFIED: Setup argument parser to fix specific hyperparameters ---
    parser = argparse.ArgumentParser(description="GPU Hyperparameter Sweep Manager")
    parser.add_argument("--dataset", type=str, choices=['brightkite','gowalla'], default='brightkite')

    
    # Parse only the --run_baseline flag first
    args, remaining_argv = parser.parse_known_args()
    
    active_grid = STATIC_GRID

    

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
    RESULTS_DIR = f"./static_results_{args.dataset}/"
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
    # breakpoint()
    run_commands = deque(generate_run_commands(BASE_COMMAND + f" --dataset {args.dataset}", active_grid, args))
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
            time.sleep(15)

    print("\n--- Hyperparameter Sweep Complete ---")
    print(f"Successfully completed {completed_runs} runs.")

if __name__ == "__main__":
    main()
