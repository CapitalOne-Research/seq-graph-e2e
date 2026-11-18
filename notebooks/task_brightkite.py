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
csv_path = f'{base_path}/brightkite_all_events.csv'
df = pd.read_csv(csv_path, dtype={"uid": "Int64", "timestamp": "Int64", "other_uid": "Int64"})
logging.info(f"Loaded CSV data from {csv_path} with shape {df.shape}")

# %%
logging.info("Starting friend recommendation processing")
df_count = df.groupby(['uid', 'event_set']).agg(cnt=('event', 'count')).reset_index()
logging.info("Grouped events by uid and event_set")
df_freq = df_count.groupby(['event_set', 'cnt']).agg(freq=('uid', 'count')).reset_index()
logging.info("Calculated frequency of counts per event_set")
dfr = df[df['event_set'] == 'relational'].reset_index()
logging.info(f"Filtered relational events: {dfr.shape[0]} records")

# # %%
# # 70-10-20 split, 0.125 * 0.8 = 0.1
# dfr_train, dfr_test = train_test_split(dfr, test_size=0.2, random_state=SEED)
# dfr_train, dfr_val = train_test_split(dfr_train, test_size=0.125, random_state=SEED)
# logging.info(f"Train/Val/Test split for relational: {dfr_train.shape[0]} train, {dfr_val.shape[0]} val, {dfr_test.shape[0]} test")

# # %%
# dfr_train = dfr_train.sort_values(['uid', 'other_uid'])[['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# dfr_test = dfr_test.sort_values(['uid', 'other_uid'])[['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# dfr_val = dfr_val.sort_values(['uid', 'other_uid'])[['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
# logging.info("Sorted relational train, test, and validation datasets")

# %%
dfp = df[df['event_set'] == 'personal'].reset_index(drop=True)
logging.info(f"Filtered personal events: {dfp.shape[0]} records")

# # %%
# task_path = '../data/tasks/brightkite/friend_recommendation'
# os.makedirs(task_path, exist_ok=True)
# logging.info(f"Task path created: {task_path}")

# dfp.to_csv(f'{task_path}/personal.csv', index=False)
# dfr_train.to_csv(f'{task_path}/relational_train.csv', index=False)
# dfr_test.to_csv(f'{task_path}/relational_test.csv', index=False)
# dfr_val.to_csv(f'{task_path}/relational_val.csv', index=False)
# logging.info("Saved friend recommendation CSV files")

# # %%
# # Check saved file (optional)
# pd.read_csv(f'{task_path}/personal.csv')

# # %% [markdown]
# # ### Negative sample

# # %%
# NEG_SAMPLE_CNT = 1000

# # %%
# all_possible_ids = {int(uid) for uid in dfr['uid'].unique()}

# def sample_negative(row, uid_to_positive, negative_count=10):
#     """
#     For a given row from df, sample negative ids.
#     Negative samples are drawn from all_possible_ids excluding the positive ids
#     for the given uid.
#     """
#     uid = row['uid']
#     positive_set = uid_to_positive.get(uid, set())
#     candidates = list(all_possible_ids - positive_set)
#     if len(candidates) < negative_count:
#         return candidates
#     else:
#         return random.sample(candidates, negative_count)
    
# def generate_negative_sample(dfr, uid_to_positive):
#     negative_rows = []
#     for idx, row in tqdm(dfr.iterrows(), total=len(dfr), desc="Generating negative samples"):
#         neg_samples = sample_negative(row, uid_to_positive, negative_count=NEG_SAMPLE_CNT)
#         for neg in neg_samples:
#             negative_rows.append({
#                 'uid': row['uid'],
#                 'timestamp': row['timestamp'],
#                 'event_set': 'negative-' + row['event_set'],
#                 'event': row['event'],
#                 'other_uid': neg
#             })
#     dfr_negative = pd.DataFrame(negative_rows)
#     return dfr_negative

# # %%
# # validation negative sample generation
# dfr_val_pos = pd.concat([dfr_val, dfr_train]).groupby('uid')['other_uid'].agg(set).reset_index()
# uid_to_positive = dict(zip(dfr_val_pos['uid'], dfr_val_pos['other_uid']))
# dfr_val_neg = generate_negative_sample(dfr_val, uid_to_positive)
# logging.info(f"Generated negative samples for validation: {dfr_val_neg.shape}")

# # %%
# # test negative sample generation
# dfr_test_pos = pd.concat([dfr_test, dfr_train]).groupby('uid')['other_uid'].agg(set).reset_index()
# uid_to_positive = dict(zip(dfr_test_pos['uid'], dfr_test_pos['other_uid']))
# dfr_test_neg = generate_negative_sample(dfr_test, uid_to_positive)
# logging.info(f"Generated negative samples for test: {dfr_test_neg.shape}")

# # %%
# dfr_test_neg.to_csv(f'{task_path}/relational_test_negative_sample.csv', index=False)
# dfr_val_neg.to_csv(f'{task_path}/relational_val_negative_sample.csv', index=False)
# logging.info("Saved relational negative sample CSV files for test and validation")

# %% [markdown]
# ## Checkin Recommendation

MAX_TEST_EVENT = 20
MAX_VAL_EVENT = 10

TEST_PROP = 0.2
VAL_PROP = 0.1

# %%
logging.info("Starting checkin recommendation processing")
dfp['order'] = dfp.groupby("uid").cumcount()
dfp['max_order'] = dfp.groupby("uid")['order'].transform('max')
dfp['rev_order'] = dfp['max_order'] - dfp['order']
dfp['order_prop'] = dfp['order'] / dfp['max_order']
dfp['rev_order_prop'] = dfp['rev_order'] / dfp['max_order']

# %%
dfp['test_flag'] = (((dfp['rev_order'] < np.ceil(TEST_PROP * dfp['max_order'])) & (dfp['rev_order'] < MAX_TEST_EVENT)) | (dfp['rev_order'] == 0))
dfp['val_flag'] = ((dfp['test_flag'] != True) & (dfp['rev_order'] < (np.ceil(VAL_PROP * dfp['max_order']) + np.ceil(TEST_PROP * dfp['max_order']))) & (dfp['rev_order'] < (MAX_TEST_EVENT + MAX_VAL_EVENT)))

logging.info("Flagged test and validation samples for checkin recommendation")

# %%
dfp_train = dfp[((dfp['test_flag'] == False) & (dfp['val_flag'] == False))][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
dfp_val = dfp[(dfp['val_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
dfp_test = dfp[(dfp['test_flag'] == True)][['uid', 'timestamp', 'event_set', 'event', 'other_uid']].reset_index(drop=True)
logging.info(f"Checkin recommendation split: {dfp_train.shape[0]} train, {dfp_val.shape[0]} val, {dfp_test.shape[0]} test")

# %%
task_path = '../data/tasks/brightkite/checkin_prediction'
os.makedirs(task_path, exist_ok=True)
logging.info(f"Task path created for checkin recommendation: {task_path}")

dfr.to_csv(f'{task_path}/relational.csv', index=False)
dfp_train.to_csv(f'{task_path}/personal_train.csv', index=False)
dfp_test.to_csv(f'{task_path}/personal_test.csv', index=False)
dfp_val.to_csv(f'{task_path}/personal_val.csv', index=False)
logging.info("Saved checkin recommendation CSV files")

# %% [markdown]
# ### Negative sample for checkin recommendation

# %%
NEG_SAMPLE_CNT = 500
all_possible_ids = set(dfp['event'])

# def sample_negative(dfp_row, uid_to_positive, negative_count=10):
#     """
#     For a given row from df, sample negative ids.
#     Negative samples are drawn from all_possible_ids excluding the positive ids
#     for the given uid.
#     """
#     uid = dfp_row['uid']
#     positive_set = uid_to_positive.get(uid, set())
#     candidates = list(all_possible_ids - positive_set)
#     if len(candidates) < negative_count:
#         return candidates
#     else:
#         return random.sample(candidates, negative_count)
    
# def generate_negative_sample(dfp, uid_to_positive):
#     negative_rows = []
#     for idx, row in tqdm(dfp.iterrows(), total=len(dfp), desc="Generating negative samples for checkin"):
#         neg_samples = sample_negative(row, uid_to_positive, negative_count=NEG_SAMPLE_CNT)
#         for neg in neg_samples:
#             negative_rows.append({
#                 'uid': row['uid'],
#                 'timestamp': None,
#                 'event_set': 'negative-' + row['event_set'],
#                 'event': neg,
#                 'other_uid': None
#             })
#     dfp_negative = pd.DataFrame(negative_rows)
#     return dfp_negative

### more relevant negative sampler

# def sample_negative_for_row(row, all_possible_ids, uid_to_positive, negative_count=10):
#     uid = row['uid']
#     current = row['event']  # e.g., '9q9gn653'
#     # Compute candidate pool on the fly:
#     pos_set = uid_to_positive.get(uid, set())
#     candidates = list(all_possible_ids - pos_set)
#     if not candidates:
#         return []
    
#     # Convert candidates (8-character strings) to a 2D NumPy array of shape (n_candidates, 8)
#     cand_arr = np.array(candidates, dtype='S8').view('S1').reshape(-1, 8)
#     # Convert current geohash into a 1x8 array using its characters.
#     curr_arr = np.array(list(current), dtype='S1').reshape(1, 8)
    
#     # Compute elementwise equality.
#     eq = (cand_arr == curr_arr)  # shape (n_candidates, 8), boolean
#     # For each candidate, find the first index where it does NOT match.
#     p = np.argmax(~eq, axis=1)
    
#     # Define buckets:
#     # Bucket 0: candidates with common prefix length < 1 (i.e., no matching first character)
#     bucket0 = np.where(p < 1)[0]
#     # For buckets 1 to 7: candidates with common prefix length >= k.
#     buckets = {k: np.where(p >= k)[0] for k in range(1, 8)}
    
#     total_buckets = 8
#     base = negative_count // total_buckets
#     rem = negative_count % total_buckets
#     sampled = set()
    
#     # Sample from bucket 0:
#     desired = base + (1 if 0 < rem else 0)
#     if len(bucket0) > 0:
#         chosen = np.random.choice(bucket0, desired, replace=False) if len(bucket0) >= desired else bucket0
#         for i in chosen:
#             sampled.add(candidates[i].decode('ascii') if isinstance(candidates[i], bytes) else candidates[i])
    
#     # Sample from buckets 1 to 7:
#     for k in range(1, 8):
#         desired = base + (1 if k < rem else 0)
#         idxs = buckets.get(k, [])
#         if len(idxs) > 0:
#             chosen = np.random.choice(idxs, desired, replace=False) if len(idxs) >= desired else idxs
#             for i in chosen:
#                 sampled.add(candidates[i].decode('ascii') if isinstance(candidates[i], bytes) else candidates[i])
    
#     sampled = list(sampled)
#     # If the total number of negatives is still less than negative_count, fill in from overall candidates.
#     if len(sampled) < negative_count:
#         used = set(sampled)
#         leftover = [s.decode('ascii') if isinstance(s, bytes) else s for s in candidates if (s.decode('ascii') if isinstance(s, bytes) else s) not in used]
#         extra = negative_count - len(sampled)
#         if len(leftover) <= extra:
#             sampled.extend(leftover)
#         else:
#             sampled.extend(np.random.choice(leftover, extra, replace=False).tolist())
    
#     return sampled

# def sample_negative_for_row_wrapper(row, all_possible_ids, uid_to_positive, negative_count):
#     uid = row['uid']
#     event_set = row['event_set']
#     current = row['event']
#     negatives = sample_negative_for_row(row, all_possible_ids, uid_to_positive, negative_count)
#     return uid, event_set, current, negatives

# def generate_negative_sample(df, all_possible_ids, uid_to_positive, negative_count=10, n_jobs=None):
#     """
#     Generate a DataFrame of negative samples.
    
#     Parameters:
#       - df: DataFrame with columns 'uid', 'event' (8-character geohash), and 'event_set'.
#       - all_possible_ids: Set of all unique geohash strings.
#       - uid_to_positive: Dictionary mapping uid to a set of positive geohash strings.
#       - negative_count: Number of negatives to sample per row.
#       - n_jobs: Number of parallel processes.
      
#     Returns:
#       A DataFrame with columns: 'uid', 'timestamp', 'event_set', 'event', 'other_uid'
#       (with 'timestamp' and 'other_uid' set to None).
#     """
#     neg_rows = []
#     # Convert DataFrame rows to a list for processing.
#     rows = [row for _, row in df.iterrows()]
#     process_func = partial(sample_negative_for_row_wrapper, 
#                            all_possible_ids=all_possible_ids, 
#                            uid_to_positive=uid_to_positive, 
#                            negative_count=negative_count)
    
#     with ProcessPoolExecutor(max_workers=n_jobs) as executor:
#         # Use executor.map to preserve order.
#         results = list(tqdm(executor.map(process_func, rows), total=len(rows), desc="Processing rows"))
    
#     for uid, event_set, current, negatives in results:
#         for neg in negatives:
#             neg_rows.append({
#                 'uid': uid,
#                 'timestamp': None,
#                 'event_set': 'negative-' + event_set,
#                 'event': neg,
#                 'other_uid': None
#             })
    
#     return pd.DataFrame(neg_rows)

def sample_negative_for_row(row, all_possible_ids, uid_to_positive, negative_count=10):
    uid = row['uid']
    current = row['event']  # e.g., '9q9gn653'
    # Compute candidate pool on the fly:
    pos_set = uid_to_positive.get(uid, set())
    candidates = list(all_possible_ids - pos_set)
    if not candidates:
        return []
    
    # Convert candidates (8-character strings) to a 2D NumPy array of shape (n_candidates, 8)
    cand_arr = np.array(candidates, dtype='S8').view('S1').reshape(-1, 8)
    # Convert current geohash into a 1x8 array using its characters.
    curr_arr = np.array(list(current), dtype='S1').reshape(1, 8)
    
    # Compute elementwise equality.
    eq = (cand_arr == curr_arr)  # shape (n_candidates, 8), boolean
    # For each candidate, find the first index where it does NOT match.
    p = np.argmax(~eq, axis=1)
    
    # Define buckets:
    # Bucket 0: candidates with common prefix length < 1 (i.e., no match on first character)
    bucket0 = np.where(p < 1)[0]
    # Buckets 1 to 7: candidates with common prefix length >= k.
    buckets = {k: np.where(p >= k)[0] for k in range(1, 8)}
    
    total_buckets = 8
    base = negative_count // total_buckets
    rem = negative_count % total_buckets
    sampled = set()
    
    # Sample from bucket 0:
    desired = base + (1 if 0 < rem else 0)
    if len(bucket0) > 0:
        chosen = (np.random.choice(bucket0, desired, replace=False)
                  if len(bucket0) >= desired else bucket0)
        for i in chosen:
            sampled.add(candidates[i].decode('ascii') if isinstance(candidates[i], bytes) else candidates[i])
    
    # Sample from buckets 1 to 7:
    # Skip bucket 6 and 7
    # for k in range(1, 8):
    for k in range(1, 6):
        desired = base + (1 if k < rem else 0)
        idxs = buckets.get(k, [])
        if len(idxs) > 0:
            chosen = (np.random.choice(idxs, desired, replace=False)
                      if len(idxs) >= desired else idxs)
            for i in chosen:
                sampled.add(candidates[i].decode('ascii') if isinstance(candidates[i], bytes) else candidates[i])
    
    sampled = list(sampled)
    # If fewer than negative_count negatives have been sampled, fill in from overall candidates.
    if len(sampled) < negative_count:
        used = set(sampled)
        leftover = [s.decode('ascii') if isinstance(s, bytes) else s 
                    for s in candidates if (s.decode('ascii') if isinstance(s, bytes) else s) not in used]
        extra = negative_count - len(sampled)
        if len(leftover) <= extra:
            sampled.extend(leftover)
        else:
            sampled.extend(np.random.choice(leftover, extra, replace=False).tolist())
    
    return sampled

def sample_negative_for_row_wrapper(row, all_possible_ids, uid_to_positive, negative_count):
    uid = row['uid']
    event_set = row['event_set']
    current = row['event']
    negatives = sample_negative_for_row(row, all_possible_ids, uid_to_positive, negative_count)
    return uid, event_set, current, negatives

def process_batch(rows, all_possible_ids, uid_to_positive, negative_count):
    """Process a batch of rows sequentially and return a list of results."""
    results = []
    for row in rows:
        res = sample_negative_for_row_wrapper(row, all_possible_ids, uid_to_positive, negative_count)
        results.append(res)
    return results

def chunkify(lst, n):
    """Split list lst into chunks of size n."""
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def generate_negative_sample(df, all_possible_ids, uid_to_positive, negative_count=10, n_jobs=None, batch_size=100):
    """
    Generate a DataFrame of negative samples.
    
    Each row computes its candidate pool on the fly: all_possible_ids minus uid_to_positive[uid].
    Rows are processed in batches in parallel.
    
    Parameters:
      - df: DataFrame with columns 'uid', 'event' (8-character geohash), and 'event_set'.
      - all_possible_ids: Set of all unique geohash strings.
      - uid_to_positive: Dictionary mapping uid to a set of positive geohash strings.
      - negative_count: Number of negatives to sample per row.
      - n_jobs: Number of parallel processes.
      - batch_size: Number of rows per batch.
      
    Returns:
      A DataFrame with columns: 'uid', 'timestamp', 'event_set', 'event', 'other_uid'
      (with timestamp and other_uid set to None).
    """
    rows = [row for _, row in df.iterrows()]
    batches = list(chunkify(rows, batch_size))
    
    neg_rows = []
    process_func = partial(process_batch, 
                           all_possible_ids=all_possible_ids, 
                           uid_to_positive=uid_to_positive, 
                           negative_count=negative_count)
    
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Use executor.map on batches. The results will be in order.
        batch_results = list(tqdm(executor.map(process_func, batches), total=len(batches), desc="Processing batches"))
    
    # Flatten the batch results.
    for batch in batch_results:
        for uid, event_set, current, negatives in batch:
            for neg in negatives:
                neg_rows.append({
                    'uid': uid,
                    'timestamp': None,
                    'event_set': 'negative-' + event_set,
                    'event': neg,
                    'other_uid': None
                })
    
    return pd.DataFrame(neg_rows)

# %%
# validation negative sample generation for checkin recommendation
dfp_val_pos = pd.concat([dfp_val, dfp_train]).groupby('uid')['event'].agg(set).reset_index()
uid_to_positive = dict(zip(dfp_val_pos['uid'], dfp_val_pos['event']))
dfp_val_neg = generate_negative_sample(dfp_val, all_possible_ids, uid_to_positive, negative_count=NEG_SAMPLE_CNT)
logging.info(f"Generated checkin negative samples for validation: {dfp_val_neg.shape}")

# %%
# test negative sample generation for checkin recommendation
dfp_test_pos = pd.concat([dfp_test, dfp_train]).groupby('uid')['event'].agg(set).reset_index()
uid_to_positive = dict(zip(dfp_test_pos['uid'], dfp_test_pos['event']))
dfp_test_neg = generate_negative_sample(dfp_test, all_possible_ids, uid_to_positive, negative_count=NEG_SAMPLE_CNT)
logging.info(f"Generated checkin negative samples for test: {dfp_test_neg.shape}")

# %%
dfp_test_neg.to_csv(f'{task_path}/personal_test_negative_sample.csv', index=False)
dfp_val_neg.to_csv(f'{task_path}/personal_val_negative_sample.csv', index=False)
logging.info("Saved checkin negative sample CSV files for test and validation")

# %%

logging.info("Script finished successfully")
print("DONE")
