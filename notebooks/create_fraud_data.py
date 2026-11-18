import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import argparse


def prepare_gnn_data_splits_to_csv(args):
    """
    Processes data to create train, validation, and test splits for GNN training.
    If a legacy 'split_mask' column is absent, a deterministic 80/10/10 column
    is generated for backward compatibility. Canonical model membership is
    created separately by fraud_splits.py.
    """
    # --- 1. Load Processed Data ---
    print("Step 1: Loading processed data...")

    # --- Robust Path Handling ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.join(script_dir, '..', 'data')
    
    personal_events_path = os.path.join(base_data_dir, 'processed', f'amazon-{args.dataset}_all_events.csv')
    labels_path = os.path.join(base_data_dir, 'processed', f'amazon-{args.dataset}_all_events_label.csv')
    
    # --- MODIFIED: Define paths for the split relational files ---
    rel_train_path = os.path.join(base_data_dir, 'tasks', f'amazon-{args.dataset}', 'coreview_prediction', 'relational_train.csv')
    rel_val_path = os.path.join(base_data_dir, 'tasks', f'amazon-{args.dataset}', 'coreview_prediction', 'relational_val.csv')
    rel_test_path = os.path.join(base_data_dir, 'tasks', f'amazon-{args.dataset}', 'coreview_prediction', 'relational_test.csv')


    for path in [personal_events_path, labels_path, rel_train_path, rel_val_path, rel_test_path]:
        if not os.path.exists(path):
            print(f"Error: Required data file not found at '{path}'")
            return

    df_all_events = pd.read_csv(personal_events_path)
    df_labels = pd.read_csv(labels_path)
    
    # --- MODIFIED: Load and concatenate the split edge files ---
    df_rel_train = pd.read_csv(rel_train_path)
    df_rel_val = pd.read_csv(rel_val_path)
    df_rel_test = pd.read_csv(rel_test_path)
    df_all_edges = pd.concat([df_rel_train, df_rel_val, df_rel_test], ignore_index=True)



    # --- NEW: Check for and create split_mask if it doesn't exist ---
    if 'split_mask' not in df_labels.columns:
        print("Warning: 'split_mask' column not found. Generating it now with an 80/10/10 split.")
        all_uids = df_labels['uid'].unique()
        all_uids = all_uids.copy()
        np.random.RandomState(args.seed).shuffle(all_uids)
        
        # Calculate split indices
        train_end = int(len(all_uids) * 0.8)
        val_end = int(len(all_uids) * 0.9)
        
        # Assign users to splits
        train_uids_set = set(all_uids[:train_end])
        val_uids_set = set(all_uids[train_end:val_end])
        test_uids_set = set(all_uids[val_end:])
        
        # Create the mask column
        def get_split(uid):
            if uid in train_uids_set:
                return 'train'
            elif uid in val_uids_set:
                return 'val'
            else:
                return 'test'
        
        df_labels['split_mask'] = df_labels['uid'].apply(get_split)
        print("Generated 'split_mask' column successfully.")

    print("Data loaded successfully.")

    # --- 2. Prepare Event-Level Data ---
    print("\nStep 2: Preparing event-level data...")
    uids_with_labels = df_labels['uid'].unique()
    df_personal = df_all_events[
        (df_all_events['event_set'] == 'personal') &
        (df_all_events['uid'].isin(uids_with_labels))
    ].copy()
    df_personal['reviewText'] = df_personal['reviewText'].astype(str).fillna('')
    
    event_df = pd.merge(df_personal, df_labels, on='uid', how='inner').sort_values('uid').reset_index(drop=True)
    print(f"Prepared {len(event_df)} total events for processing.")
    output_dir = os.path.join(base_data_dir, 'processed', f'amazon-{args.dataset}')
    os.makedirs(output_dir, exist_ok=True)
    # --- 3. Create Unified Nodes DataFrame ---
    print("\nStep 3: Creating the unified 'nodes.csv' file...")
    
    # Filter for personal events of users that have labels
    uids_with_labels = df_labels['uid'].unique()
    df_personal = df_all_events[
        (df_all_events['event_set'] == 'personal') &
        (df_all_events['uid'].isin(uids_with_labels))
    ].copy()
    
    # # Merge event data with labels to create the final node feature set
    nodes_df = pd.merge(df_personal, df_labels, on='uid', how='inner')
    
    # # Ensure reviewText is a string to prevent errors during embedding
    nodes_df['reviewText'] = nodes_df['reviewText'].astype(str).fillna('')
    
    # Save the unified nodes file
    nodes_df.to_csv(os.path.join(output_dir, 'all_nodes.csv'), index=False)
    # print(f"   > Saved {len(nodes_df)} events for {nodes_df['uid'].nunique()} users to '{nodes_output_path}'")

    # --- 4. Create Unified Edges DataFrame ---
    print("\nStep 4: Creating the unified 'edges.csv' file...")
    
    # Concatenate all relational (edge) data
    edges_df = pd.concat([df_rel_train, df_rel_val, df_rel_test], ignore_index=True)
    
    # Ensure we only have the necessary columns and remove duplicates
    edges_df = edges_df[['uid', 'timestamp', 'event_set', 'event','other_uid']].drop_duplicates()
    
    # Save the unified edges file
    edges_df.to_csv(os.path.join(output_dir, 'all_edges.csv'), index=False)
    print("\nProcessing complete.")

    
    

    # # --- 3. Identify Nodes for Each Split using the Mask ---
    # print("\nStep 3: Identifying train, val, and test nodes from split mask...")
    
    # train_uids = set(df_labels[df_labels['split_mask'] == 'train']['uid'].unique())
    # val_uids = set(df_labels[df_labels['split_mask'] == 'val']['uid'].unique())
    # test_uids = set(df_labels[df_labels['split_mask'] == 'test']['uid'].unique())

    # print(f"Training users: {len(train_uids)}")
    # print(f"Validation users: {len(val_uids)}")
    # print(f"Test users: {len(test_uids)}")

    # # --- 4. Split Node DataFrames ---
    # print("\nStep 4: Splitting event data (nodes) into train, val, and test sets...")
    
    # train_nodes_df = event_df[event_df['uid'].isin(train_uids)]
    # val_nodes_df = event_df[event_df['uid'].isin(val_uids)]
    # test_nodes_df = event_df[event_df['uid'].isin(test_uids)]

    # # --- 5. Split Edge DataFrames from Global Edge List ---
    # print("\nStep 5: Splitting relational data (edges) into train, val, and test sets...")

    # train_edges_df = df_all_edges[
    #     (df_all_edges['uid'].isin(train_uids)) &
    #     (df_all_edges['other_uid'].isin(train_uids))
    # ][['uid', 'other_uid']]

    # val_edges_df = df_all_edges[
    #     (df_all_edges['uid'].isin(val_uids)) &
    #     (df_all_edges['other_uid'].isin(val_uids))
    # ][['uid', 'other_uid']]

    # test_edges_df = df_all_edges[
    #     (df_all_edges['uid'].isin(test_uids)) &
    #     (df_all_edges['other_uid'].isin(test_uids))
    # ][['uid', 'other_uid']]
    
    
    


    # # --- 5a. Print Split Statistics ---
    # print("\n--- Data Split Statistics ---")
    
    # # Train Set
    # num_train_users = len(train_uids)
    # fraud_train_users = train_nodes_df[train_nodes_df['fraud_label'] == 1]['uid'].nunique()
    # fraud_rate_train = (fraud_train_users / num_train_users) * 100 if num_train_users > 0 else 0
    # print(f"Train Set:")
    # print(f"  - Users: {num_train_users}")
    # print(f"  - Events (Reviews): {len(train_nodes_df)}")
    # print(f"  - Edges: {len(train_edges_df)}")
    # print(f"  - Fraudulent Users: {fraud_train_users} ({fraud_rate_train:.2f}%)")

    # # Validation Set
    # num_val_users = len(val_uids)
    # fraud_val_users = val_nodes_df[val_nodes_df['fraud_label'] == 1]['uid'].nunique()
    # fraud_rate_val = (fraud_val_users / num_val_users) * 100 if num_val_users > 0 else 0
    # print(f"Validation Set:")
    # print(f"  - Users: {num_val_users}")
    # print(f"  - Events (Reviews): {len(val_nodes_df)}")
    # print(f"  - Edges: {len(val_edges_df)}")
    # print(f"  - Fraudulent Users: {fraud_val_users} ({fraud_rate_val:.2f}%)")

    # # Test Set
    # num_test_users = len(test_uids)
    # fraud_test_users = test_nodes_df[test_nodes_df['fraud_label'] == 1]['uid'].nunique()
    # fraud_rate_test = (fraud_test_users / num_test_users) * 100 if num_test_users > 0 else 0
    # print(f"Test Set:")
    # print(f"  - Users: {num_test_users}")
    # print(f"  - Events (Reviews): {len(test_nodes_df)}")
    # print(f"  - Edges: {len(test_edges_df)}")
    # print(f"  - Fraudulent Users: {fraud_test_users} ({fraud_rate_test:.2f}%)")
    # print("-----------------------------")


    # # --- 6. Save the Splits to CSV Files ---

    

    # print(f"\nStep 6: Saving split data to '{output_dir}'...")

    # train_nodes_df.to_csv(os.path.join(output_dir, 'train_nodes.csv'), index=False)
    # val_nodes_df.to_csv(os.path.join(output_dir, 'val_nodes.csv'), index=False)
    # test_nodes_df.to_csv(os.path.join(output_dir, 'test_nodes.csv'), index=False)
    # print("   > Saved node data for all splits.")

    # train_edges_df.to_csv(os.path.join(output_dir, 'train_edges.csv'), index=False)
    # val_edges_df.to_csv(os.path.join(output_dir, 'val_edges.csv'), index=False)
    # test_edges_df.to_csv(os.path.join(output_dir, 'test_edges.csv'), index=False)
    # print("   > Saved edge data for all splits.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a GNN for Fraud Detection from CSV Splits")
    parser.add_argument("--dataset", type=str, choices=['musical', 'movies','clothing','electronics','home-kitchen','office-products'], default='musical')
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs.")
    parser.add_argument('--seed', type=int, default=0,
                        help="Seed for the legacy split_mask column (canonical splits are separate).")
    args = parser.parse_args()
    prepare_gnn_data_splits_to_csv(args)

