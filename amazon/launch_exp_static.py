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
    # '--dataset': ['musical','clothing','electronics','movies'],
    # '--dataset': ['clothing','musical','electronics','movies'],
    '--dataset': ['movies', 'electronics', 'clothing'],
    '--model': ['TransformerConv','GCN','GAT'],
    '--num_neighbors': [10],
    '--heads': [2],
    '--seed':[0,1,2],
    '--enable_early_stopping': [True],
    '--lr': [1e-3],
    '--use_sequence_embedding': [True,False],
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

def generate_run_commands(base_cmd, grid, log_dir):
    """
    Generates all possible command-line strings from the hyperparameter grid.
    """
    keys = grid.keys()
    values = grid.values()

    for combination in itertools.product(*values):
        args_dict = dict(zip(keys, combination))
        
        # Add the log directory to the command
        cmd = f"{base_cmd}"

        for arg, val in args_dict.items():
            if isinstance(val, bool):
                if val:
                    cmd += f" {arg}"
            else:
                cmd += f" {arg} {val}"
                
        # --- MODIFIED SECTION ---
        # If --use_sequence_embedding is True, add the corresponding filename
        if args_dict.get('--use_sequence_embedding') is True:
            dataset = args_dict.get('--dataset')
            if dataset: # Ensure dataset is not None
                # Construct the filename, e.g., "clothing_user_embedding.pkl"
                embedding_filename = f"{dataset}_user_embedding.pkl"
                cmd += f" --seq_embedding_filename ./stored/{dataset}-user_embedding-m0.3-lr1e-4.pkl"
            
        yield cmd

# --- Main Execution Logic ---

def main():
    """
    Main function to orchestrate the hyperparameter sweep.
    """
    
    
    # --- MODIFIED: Setup argument parser to fix specific hyperparameters ---
    parser = argparse.ArgumentParser(description="GPU Hyperparameter Sweep Manager")


    
    # Parse only the --run_baseline flag first
    args, remaining_argv = parser.parse_known_args()
    
    active_grid = STATIC_GRID
    all_commands_to_run = deque()

    # --- ✨ MODIFIED: Loop through each dataset to create separate folders and commands ---
    # Store a copy of the original dataset list before iterating
    datasets_to_run = list(active_grid['--dataset'])

    for dataset in datasets_to_run:
        print("\n" + "="*50)
        print(f"--- Preparing runs for dataset: {dataset} ---")

        # 1. Define the results directory for the current dataset
        # This will be passed to the training script
        RESULTS_DIR = f"./logs/static-graph-fraud-{dataset}/"
        print(f"Results will be saved in: {RESULTS_DIR}")
        os.makedirs(RESULTS_DIR, exist_ok=True)

        # 2. Create a temporary grid with the current dataset fixed
        temp_grid = active_grid.copy()
        temp_grid['--dataset'] = [dataset]

        # 3. Generate commands for this dataset and add them to the main queue
        dataset_commands = list(generate_run_commands(BASE_COMMAND, temp_grid, RESULTS_DIR))
        all_commands_to_run.extend(dataset_commands)
        print(f"Generated {len(dataset_commands)} configurations for '{dataset}'.")

    print("\n" + "="*50)
    print("--- Starting Hyperparameter Sweep ---")
    
    run_commands = all_commands_to_run
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
            time.sleep(30)
        else:
            # If all commands have been launched, just wait for the last ones to finish
            time.sleep(60)

    print("\n--- Hyperparameter Sweep Complete ---")
    print(f"Successfully completed {completed_runs} runs.")

if __name__ == "__main__":
    main()
