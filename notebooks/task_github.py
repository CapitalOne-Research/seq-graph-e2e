# %%
import numpy as np 
import matplotlib.pyplot as plt
import time
import torch
import pandas as pd
import os
import logging  # added logging

import random
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from functools import partial

from collections import defaultdict
from torch_geometric.utils.convert import from_networkx
from torch_geometric.utils import to_undirected

from sklearn.model_selection import train_test_split

from tqdm import tqdm  # using console-based progress bar

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ----------------------
# Helper functions for parallel negative sample generation

def chunkify(lst, n):
    """
    Yield successive n-sized chunks from lst.
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def process_batch(batch, all_possible_ids, uid_to_positive, negative_count):
    """
    Process a single batch of rows, sampling negative events for each row.
    
    Parameters:
      - batch: List of DataFrame rows.
      - all_possible_ids: Set of all available event ids.
      - uid_to_positive: Dictionary mapping uid to a set of positive event ids.
      - negative_count: Number of negatives to sample per row.
      
    Returns:
      A list of tuples: (uid, event_set, current_event, negatives)
    """
    results = []
    for row in batch:
        uid = row['uid']
        event_set = row['event_set']
        event = row['event']
        
        positive_set = uid_to_positive.get(uid, set())
        candidates = list(all_possible_ids - positive_set)
        if len(candidates) < negative_count:
            negatives = candidates
        else:
            negatives = random.sample(candidates, negative_count)
        results.append((uid, event_set, event, negatives))
    return results

def generate_negative_sample(df, all_possible_ids, uid_to_positive, sample_type='personal', negative_count=10, n_jobs=None, batch_size=100):
    """
    Generate a DataFrame of negative samples in parallel.
    
    Each row computes its candidate pool on the fly: all_possible_ids minus uid_to_positive[uid].
    Rows are processed in batches using parallel processing.
    
    Parameters:
      - df: DataFrame with columns 'uid', 'event' (or similar), and 'event_set'.
      - all_possible_ids: Set of all unique event ids.
      - uid_to_positive: Dictionary mapping uid to a set of positive event ids.
      - negative_count: Number of negatives to sample per row.
      - n_jobs: Number of parallel processes (if None, uses the default).
      - batch_size: Number of rows per batch.
      
    Returns:
      A DataFrame with columns: 'uid', 'timestamp', 'event_set', 'event', 'other_uid'
      (with 'timestamp' and 'other_uid' set to None for negative samples).
    """
    rows = [row for _, row in df.iterrows()]
    batches = list(chunkify(rows, batch_size))
    
    neg_rows = []
    process_func = partial(process_batch, all_possible_ids=all_possible_ids, uid_to_positive=uid_to_positive, negative_count=negative_count)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        batch_results = list(tqdm(executor.map(process_func, batches), total=len(batches), desc="Processing batches"))
    
    # Flatten the batch results.
    for batch in batch_results:
        for uid, event_set, current_event, negatives in batch:
            for neg in negatives:
                neg_rows.append({
                    'uid': uid,
                    'timestamp': None,
                    'event_set': 'negative-' + event_set,
                    'event': neg if sample_type == 'personal' else current_event,
                    'other_uid': neg if sample_type == 'relational' else None
                })
    
    return pd.DataFrame(neg_rows)



# %%
import numpy as np
import torch
import random

def set_seed(seed=42):
    """Set the same seed for reproducibility across NumPy, PyTorch, and Python."""
    np.random.seed(seed)             # NumPy
    torch.manual_seed(seed)          # PyTorch CPU
    torch.cuda.manual_seed(seed)     # PyTorch GPU (if applicable)
    torch.cuda.manual_seed_all(seed) # If using multiple GPUs
    random.seed(seed)                # Python's built-in random module

    # Ensure deterministic behavior in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logging.info(f"Random seed set to {seed}")

# %%
SEED = 42
set_seed(seed=SEED)

# %%
base_path = '../data/processed'
logging.info(f"Base path set to: {base_path}")

# %%
csv_path = f'{base_path}/github_all_events.csv'
df = pd.read_csv(csv_path, dtype={"uid": "Int64", "timestamp": "Int64", "other_uid": "Int64"})
logging.info(f"Loaded CSV data from {csv_path} with shape {df.shape}")

# %%
logging.info("Starting collab prediction processing")
df_count = df.groupby(['uid', 'event_set']).agg(cnt=('event', 'count')).reset_index()
logging.info("Grouped events by uid and event_set")
df_freq = df_count.groupby(['event_set', 'cnt']).agg(freq=('uid', 'count')).reset_index()
logging.info("Calculated frequency of counts per event_set")
dfr = df[df['event_set'] == 'relational'].reset_index(drop=True)
logging.info(f"Filtered relational events: {dfr.shape[0]} records")

# %%
dfr['order'] = dfr.groupby("uid").cumcount()
dfr['max_order'] = dfr.groupby("uid")['order'].transform('max')
dfr['rev_order'] = dfr['max_order'] - dfr['order']
dfr['order_prop'] = dfr['order'] / dfr['max_order']
dfr['rev_order_prop'] = dfr['rev_order'] / dfr['max_order']

# %%
MAX_TEST_EVENT = 20
MAX_VAL_EVENT = 10

TEST_PROP = 0.2
VAL_PROP = 0.1

# %%

dfr['test_flag'] = (((dfr['rev_order'] < np.ceil(TEST_PROP * dfr['max_order'])) & (dfr['rev_order'] < MAX_TEST_EVENT)) | (dfr['rev_order'] == 0))
dfr['val_flag'] = ((dfr['test_flag'] != True) & (dfr['rev_order'] < (np.ceil(VAL_PROP * dfr['max_order']) + np.ceil(TEST_PROP * dfr['max_order']))) & (dfr['rev_order'] < (MAX_TEST_EVENT + MAX_VAL_EVENT)))

logging.info("Flagged test and validation samples for collab prediction")

# %%
dfr_train = dfr[((dfr['test_flag'] == False) & (dfr['val_flag'] == False))][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
dfr_val = dfr[(dfr['val_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
dfr_test = dfr[(dfr['test_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
logging.info(f"Train/Val/Test split for relational: {dfr_train.shape[0]} train, {dfr_val.shape[0]} val, {dfr_test.shape[0]} test")

# %%
dfp = df[df['event_set'] == 'personal'].reset_index(drop=True)
logging.info(f"Filtered personal events: {dfp.shape[0]} records")

# # split dfp into dfp_observed and dfp_unobserved
# # Step 1: Get max timestamp per uid in dfr
# dfr_max_ts = dfr_train.groupby("uid")["timestamp"].max().reset_index()
# dfr_max_ts.rename(columns={"timestamp": "max_timestamp"}, inplace=True)

# # Step 2: Merge dfp with dfr_max_ts on uid
# dfp_merged = dfp.merge(dfr_max_ts, on="uid", how="left")

# # Step 3: Split into observed and unobserved
# dfp_observed = dfp_merged[dfp_merged["timestamp"] <= dfp_merged["max_timestamp"]].copy()
# dfp_unobserved = dfp_merged[dfp_merged["timestamp"] > dfp_merged["max_timestamp"]].copy()

# # Drop the extra 'max_timestamp' column if you want
# dfp_observed.drop(columns="max_timestamp", inplace=True)
# dfp_unobserved.drop(columns="max_timestamp", inplace=True)


# %%
task_path = '../data/tasks/github/collab_prediction'
os.makedirs(task_path, exist_ok=True)
logging.info(f"Task path created: {task_path}")

dfr_train.to_csv(f'{task_path}/relational_train.csv', index=False)
logging.info(f"Saved relational_train.csv with shape {dfr_train.shape}")

dfr_test.to_csv(f'{task_path}/relational_test.csv', index=False)
logging.info(f"Saved relational_test.csv with shape {dfr_test.shape}")

dfr_val.to_csv(f'{task_path}/relational_val.csv', index=False)
logging.info(f"Saved relational_val.csv with shape {dfr_val.shape}")

# dfp_observed.to_csv(f'{task_path}/personal_observed.csv', index=False)
# logging.info(f"Saved personal_observed.csv with shape {dfp_observed.shape}")

# dfp_unobserved.to_csv(f'{task_path}/personal_unobserved.csv', index=False)
# logging.info(f"Saved personal_unobserved.csv with shape {dfp.shape}")

dfp.to_csv(f'{task_path}/personal.csv', index=False)
logging.info(f"Saved personal.csv with shape {dfp.shape}")


# %% [markdown]
# ### Negative sample

# %%
NEG_SAMPLE_CNT = 300
BATCH_SIZE = 10000
N_JOBS = 24

# %%
all_possible_ids = {int(uid) for uid in dfr['uid'].unique()}

# %%
# validation negative sample generation
dfr_val_pos = pd.concat([dfr_val, dfr_train]).groupby('uid')['other_uid'].agg(set).reset_index()
uid_to_positive = dict(zip(dfr_val_pos['uid'], dfr_val_pos['other_uid']))
dfr_val_neg = generate_negative_sample(dfr_val, all_possible_ids, uid_to_positive, 
                                        sample_type='relational',
                                        negative_count=NEG_SAMPLE_CNT,
                                        n_jobs=N_JOBS,
                                        batch_size=BATCH_SIZE)

logging.info(f"Generated negative samples for validation: {dfr_val_neg.shape}")

dfr_val_neg.to_csv(f'{task_path}/relational_val_negative_sample.csv', index=False)
logging.info(f"Saved relational_val_negative_sample.csv with shape {dfr_val_neg.shape}")

# %%
# test negative sample generation
dfr_test_pos = pd.concat([dfr_test, dfr_train]).groupby('uid')['other_uid'].agg(set).reset_index()
uid_to_positive = dict(zip(dfr_test_pos['uid'], dfr_test_pos['other_uid']))
dfr_test_neg = generate_negative_sample(dfr_test, all_possible_ids, uid_to_positive, 
                                        sample_type='relational',
                                        negative_count=NEG_SAMPLE_CNT,
                                        n_jobs=N_JOBS,
                                        batch_size=BATCH_SIZE)
logging.info(f"Generated negative samples for test: {dfr_test_neg.shape}")

# %%
dfr_test_neg.to_csv(f'{task_path}/relational_test_negative_sample.csv', index=False)
logging.info(f"Saved relational_test_negative_sample.csv with shape {dfr_test_neg.shape}")


# # %% [markdown]
# # ## Activity Prediction

# MAX_TEST_EVENT = 20
# MAX_VAL_EVENT = 10

# TEST_PROP = 0.2
# VAL_PROP = 0.1

# # %%
# logging.info("Starting activity prediction processing")
# dfp['order'] = dfp.groupby("uid").cumcount()
# dfp['max_order'] = dfp.groupby("uid")['order'].transform('max')
# dfp['rev_order'] = dfp['max_order'] - dfp['order']
# dfp['order_prop'] = dfp['order'] / dfp['max_order']
# dfp['rev_order_prop'] = dfp['rev_order'] / dfp['max_order']

# # %%
# dfp['test_flag'] = (((dfp['rev_order'] < np.ceil(TEST_PROP * dfp['max_order'])) & (dfp['rev_order'] < MAX_TEST_EVENT)) | (dfp['rev_order'] == 0))
# dfp['val_flag'] = ((dfp['test_flag'] != True) & (dfp['rev_order'] < (np.ceil(VAL_PROP * dfp['max_order']) + np.ceil(TEST_PROP * dfp['max_order']))) & (dfp['rev_order'] < (MAX_TEST_EVENT + MAX_VAL_EVENT)))

# logging.info("Flagged test and validation samples for activity prediction")

# # %%
# dfp_train = dfp[((dfp['test_flag'] == False) & (dfp['val_flag'] == False))][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# dfp_val = dfp[(dfp['val_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# dfp_test = dfp[(dfp['test_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# logging.info(f"Activity prediction split: {dfp_train.shape[0]} train, {dfp_val.shape[0]} val, {dfp_test.shape[0]} test")

# # split dfr into dfr_observed and dfr_unobserved
# # Step 1: Get max timestamp per uid in dfp
# dfp_max_ts = dfp_train.groupby("uid")["timestamp"].max().reset_index()
# dfp_max_ts.rename(columns={"timestamp": "max_timestamp"}, inplace=True)

# # Step 2: Merge dfr with dfp_max_ts on uid
# dfr_merged = dfr.merge(dfp_max_ts, on="uid", how="left")

# # Step 3: Split into observed and unobserved
# dfr_observed = dfr_merged[dfr_merged["timestamp"] <= dfr_merged["max_timestamp"]].copy()
# dfr_unobserved = dfr_merged[dfr_merged["timestamp"] > dfr_merged["max_timestamp"]].copy()

# # (Optional) Drop the extra 'max_timestamp' column if you want
# dfr_observed.drop(columns="max_timestamp", inplace=True)
# dfr_unobserved.drop(columns="max_timestamp", inplace=True)

# # store files
# activity_pred_task_path = f'../data/tasks/github/activity_prediction'
# os.makedirs(activity_pred_task_path, exist_ok=True)
# logging.info(f"Task path created for activity prediction: {activity_pred_task_path}")

# dfr_observed.to_csv(f'{activity_pred_task_path}/relational_observed.csv', index=False)
# logging.info(f"Saved relational_observed.csv with shape {dfr.shape}")

# dfr_unobserved.to_csv(f'{activity_pred_task_path}/relational_unobserved.csv', index=False)
# logging.info(f"Saved relational_unobserved.csv with shape {dfr.shape}")

# dfp_train.to_csv(f'{activity_pred_task_path}/personal_train.csv', index=False)
# logging.info(f"Saved personal_train.csv with shape {dfp_train.shape}")

# dfp_test.to_csv(f'{activity_pred_task_path}/personal_test.csv', index=False)
# logging.info(f"Saved personal_test.csv with shape {dfp_test.shape}")

# dfp_val.to_csv(f'{activity_pred_task_path}/personal_val.csv', index=False)
# logging.info(f"Saved personal_val.csv with shape {dfp_val.shape}")

# # %% Negative sample generation for activity prediction

# NEG_SAMPLE_CNT = 50
# BATCH_SIZE = 5000
# N_JOBS = 16


# all_possible_ids = set(dfp['event'])

# # Generate negative samples for validation in parallel.
# dfp_val_pos = pd.concat([dfp_val, dfp_train]).groupby('uid')['event'].agg(set).reset_index()
# uid_to_positive = dict(zip(dfp_val_pos['uid'], dfp_val_pos['event']))
# dfp_val_neg = generate_negative_sample(dfp_val, all_possible_ids, uid_to_positive, 
#                                         sample_type='personal',
#                                         negative_count=NEG_SAMPLE_CNT,
#                                         n_jobs=N_JOBS,
#                                         batch_size=BATCH_SIZE)
# logging.info(f"Generated activity negative samples for validation with shape {dfp_val_neg.shape}")

# # Generate negative samples for test in parallel.
# dfp_test_pos = pd.concat([dfp_test, dfp_train]).groupby('uid')['event'].agg(set).reset_index()
# uid_to_positive = dict(zip(dfp_test_pos['uid'], dfp_test_pos['event']))
# dfp_test_neg = generate_negative_sample(dfp_test, all_possible_ids, uid_to_positive, 
#                                         sample_type='personal',
#                                         negative_count=NEG_SAMPLE_CNT,
#                                         n_jobs=N_JOBS,
#                                         batch_size=BATCH_SIZE)
# logging.info(f"Generated activity negative samples for test with shape {dfp_test_neg.shape}")

# dfp_test_neg.to_csv(f'{activity_pred_task_path}/personal_test_negative_sample.csv', index=False)
# logging.info(f"Saved personal_test_negative_sample.csv with shape {dfp_test_neg.shape}")

# dfp_val_neg.to_csv(f'{activity_pred_task_path}/personal_val_negative_sample.csv', index=False)
# logging.info(f"Saved personal_val_negative_sample.csv with shape {dfp_val_neg.shape}")

# %%
logging.info("Script finished successfully")
print("DONE")
