#!/bin/bash

# This script runs the launch_eval.py script for multiple datasets and seeds.

# --- Configuration ---
# Add all the dataset names you want to evaluate into this array.
# Just separate each name with a space.
DATASETS=("musical" "clothing" "electronics" "movies")

# Define the seeds you want to run.
# This will create a sequence from 0 to 3 (i.e., 0, 1, 2, 3).
SEEDS=$(seq 0 2)

# --- Script Body ---
# This loop goes through each dataset defined in the DATASETS array.
for dataset in "${DATASETS[@]}"; do
  # This loop goes through each seed number.
  for seed in $SEEDS; do
    # Print the command to the console so you can track progress.
    echo "----------------------------------------------------"
    echo "RUNNING: python launch_eval.py --dataset $dataset --seed $seed"
    echo "----------------------------------------------------"

    # Execute the actual command.
    python launch_eval.py --dataset "$dataset" --seed "$seed"

    # Optional: Add a small delay between runs if needed.
    # sleep 1
  done
done

echo "All evaluation runs are complete."
