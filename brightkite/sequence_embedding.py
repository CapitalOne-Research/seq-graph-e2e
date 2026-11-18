import argparse
import json
import logging
from pathlib import Path
import pickle
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from transformers import BertConfig, BertModel

# -----------------------------------------------------------------------------
# Configure Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. Generate Vocabulary from Real Data (including [MASK])
# -----------------------------------------------------------------------------
def generate_vocab(csv_file, event_set_filter, 
                   pad_token="[PAD]", unk_token="[UNK]", 
                   cls_token="[CLS]", mask_token="[MASK]"):
    logger.info("Generating vocabulary from CSV file: %s", csv_file)
    df = pd.read_csv(csv_file)
    if event_set_filter:
        df = df[df["event_set"] == event_set_filter]
    
    unique_events = df["event"].dropna().unique()
    tokens = set()
    for event in unique_events:
        event = str(event)
        # Require at least 8 characters; if longer, take the first 8.
        if len(event) < 8:
            continue
        geohash = event[:8]
        # Split the 8-character geohash into 4 parts of 2 characters each.
        parts = [geohash[i:i+2] for i in range(0, 8, 2)]
        prefixes = ["gh12-", "gh34-", "gh56-", "gh78-"]
        for prefix, part in zip(prefixes, parts):
            tokens.add(f"{prefix}{part}")
    # Define special tokens with fixed indices.
    vocab = {
        pad_token: 0,
        unk_token: 1,
        cls_token: 2,
        mask_token: 3,
    }
    for token in sorted(tokens):
        vocab[token] = len(vocab)
    logger.info("Generated vocabulary with %d tokens", len(vocab))
    return vocab

# -----------------------------------------------------------------------------
# 2. Define the GeohashTokenizer Based on the Generated Vocabulary
# -----------------------------------------------------------------------------
class GeohashTokenizer:
    def __init__(self, vocab):
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.cls_token = "[CLS]"
        self.pad_token = "[PAD]"
        self.unk_token = "[UNK]"
        self.mask_token = "[MASK]"

    def tokenize_geohash(self, geohash):
        # Take only the first 8 characters and split into 4 segments of 2 characters.
        geohash = geohash[:8]
        parts = [geohash[i:i+2] for i in range(0, 8, 2)]
        prefixes = ["gh12-", "gh34-", "gh56-", "gh78-"]
        return [f"{prefix}{part}" for prefix, part in zip(prefixes, parts)]
    
    def encode(self, tokens, max_length=None, padding='max_length', truncation=True):
        processed_tokens = []
        for token in tokens:
            if token in [self.cls_token, self.pad_token, self.unk_token, self.mask_token]:
                processed_tokens.append(token)
            else:
                processed_tokens.extend(self.tokenize_geohash(token))
        # Append the [CLS] token at the end.
        tokens_with_special = processed_tokens + [self.cls_token]
        ids = [self.vocab.get(tok, self.vocab[self.unk_token]) for tok in tokens_with_special]
        if truncation and max_length is not None:
            ids = ids[-max_length:]  # keep the rightmost tokens (most recent events and [CLS])
        if padding == 'max_length' and max_length is not None:
            ids = [self.vocab[self.pad_token]] * (max_length - len(ids)) + ids
        return ids

    def decode(self, ids):
        return [self.inv_vocab.get(i, self.unk_token) for i in ids]

# -----------------------------------------------------------------------------
# 3. Masking Function for MLM
# -----------------------------------------------------------------------------
def mask_tokens(input_ids, vocab, mask_prob=0.15):
    masked_input_ids = input_ids.copy()
    labels = [-100] * len(input_ids)
    special_ids = {vocab["[PAD]"], vocab["[CLS]"], vocab["[MASK]"]}
    candidate_ids = [v for k, v in vocab.items() if v not in special_ids]
    for i, token in enumerate(input_ids):
        if token in special_ids:
            continue
        if random.random() < mask_prob:
            labels[i] = token
            rand_val = random.random()
            if rand_val < 0.8:
                masked_input_ids[i] = vocab["[MASK]"]
            elif rand_val < 0.9:
                masked_input_ids[i] = random.choice(candidate_ids)
            else:
                masked_input_ids[i] = token
    return masked_input_ids, labels

# -----------------------------------------------------------------------------
# 4. Data Processing: Read CSV, Group, Truncate/Pad User Sequences
# -----------------------------------------------------------------------------
def load_and_process_sequences(csv_file, event_set_filter, max_events_per_user, pad_token="[PAD]"):
    logger.info("Loading CSV file: %s", csv_file)
    df = pd.read_csv(csv_file)
    if event_set_filter:
        df = df[df["event_set"] == event_set_filter]
    df = df.sort_values(by=["uid", "timestamp"])
    user_groups = df.groupby("uid")["event"].apply(list)
    uids = user_groups.index.tolist()
    def pad_or_truncate(seq, max_len=max_events_per_user, pad_token=pad_token):
        if len(seq) > max_len:
            return seq[-max_len:]
        else:
            num_pad = max_len - len(seq)
            return [pad_token] * num_pad + seq
    sequences = user_groups.apply(lambda seq: pad_or_truncate(seq, max_len=max_events_per_user)).tolist()
    logger.info("Processed %d user sequences", len(sequences))
    return uids, sequences

# -----------------------------------------------------------------------------
# 5. Define the MLMDataset for Masked Language Modeling
# -----------------------------------------------------------------------------
class MLMDataset(Dataset):
    def __init__(self, sequences, tokenizer, max_seq_tokens, mask_prob=0.15):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.max_seq_tokens = max_seq_tokens
        self.mask_prob = mask_prob
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = self.tokenizer.encode(seq, max_length=self.max_seq_tokens, padding="max_length", truncation=True)
        masked_input_ids, labels = mask_tokens(input_ids, self.tokenizer.vocab, self.mask_prob)
        return torch.tensor(masked_input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)

# -----------------------------------------------------------------------------
# 6. Define the MLM Model (BERT + MLM Head)
# -----------------------------------------------------------------------------
class MLMModel(nn.Module):
    def __init__(self, bert_model, hidden_size, vocab_size):
        super(MLMModel, self).__init__()
        self.bert = bert_model
        self.mlm_head = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        logits = self.mlm_head(outputs.last_hidden_state)
        return logits

# -----------------------------------------------------------------------------
# 7. Generate and Save User Embeddings (Using the BERT Encoder)
# -----------------------------------------------------------------------------
def generate_and_save_embeddings(model, uids, sequences, tokenizer, max_seq_tokens, output_file, embedding_method="cls", embedding_layer=-1):
    model.eval()
    embeddings = {}
    pad_id = tokenizer.vocab[tokenizer.pad_token]
    device = next(model.parameters()).device
    with torch.no_grad():
        for uid, seq in zip(uids, sequences):
            input_ids = torch.tensor(
                [tokenizer.encode(seq, max_length=max_seq_tokens, padding="max_length", truncation=True)],
                dtype=torch.long
            ).to(device)
            attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long().to(device)
            outputs = model.bert(input_ids, attention_mask=attention_mask)
            if embedding_layer == -1:
                layer_output = outputs.last_hidden_state
            else:
                hidden_states = outputs.hidden_states
                layer_output = hidden_states[embedding_layer]
            if embedding_method == "cls":
                emb = layer_output[:, -1, :]
            elif embedding_method == "mean":
                mask = (input_ids != pad_id).float().unsqueeze(-1)
                sum_embeds = (layer_output * mask).sum(dim=1)
                mask_sum = mask.sum(dim=1).clamp(min=1e-9)
                emb = sum_embeds / mask_sum
            elif embedding_method == "max":
                mask = (input_ids != pad_id).float().unsqueeze(-1)
                embeds = layer_output * mask + (1 - mask) * (-1e9)
                emb, _ = embeds.max(dim=1)
            else:
                raise ValueError(f"Unknown embedding method: {embedding_method}")
            embeddings[uid] = emb.squeeze(0).cpu().numpy()
    with open(output_file, "wb") as f:
        pickle.dump(embeddings, f)
    logger.info("Saved embeddings for %d users to %s", len(embeddings), output_file)

# -----------------------------------------------------------------------------
# 8. Main Function with Configurable Parameters, GPU Support, and Mode
# -----------------------------------------------------------------------------
def main(args):
    mlm_stage_start = time.perf_counter()
    Path("stored").mkdir(parents=True, exist_ok=True)

    # Set device to a specific GPU if available.
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)
    
    # Generate vocabulary and tokenizer.
    vocab = generate_vocab(args.csv_file, args.event_set)
    tokenizer = GeohashTokenizer(vocab)
    
    # Calculate maximum token sequence length.
    # Each event now contributes 4 tokens, so:
    # max_seq_tokens = (max_events * 4) + 1 ([CLS] token at the end)
    max_seq_tokens = args.max_events * 4 + 1
    logger.info("Max sequence tokens: %d", max_seq_tokens)
    
    # Create BERT configuration.
    config = BertConfig(
        vocab_size=len(vocab),
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.hidden_size * 2,
        max_position_embeddings=max_seq_tokens,
        output_hidden_states=True,
    )
    bert_model = BertModel(config)
    mlm_model = MLMModel(bert_model, hidden_size=config.hidden_size, vocab_size=len(vocab))
    mlm_model.to(device)
    
    if args.mode == "embed":
        logger.info("Loading model from stored/%s", args.model_file)
        mlm_model.load_state_dict(torch.load(f"stored/{args.model_file}", map_location=device))
    else:
        uids, sequences = load_and_process_sequences(args.csv_file, args.event_set, args.max_events)
        dataset = MLMDataset(sequences, tokenizer, max_seq_tokens, mask_prob=args.mask_prob)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        optimizer = optim.Adam(mlm_model.parameters(), lr=args.learning_rate)
        logger.info("Starting MLM training for %d epochs", args.num_epochs)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_start = time.perf_counter()
        
        for epoch in range(args.num_epochs):
            mlm_model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            batch_count = 0
            for batch in dataloader:
                input_ids, labels = batch
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                
                attention_mask = (input_ids != tokenizer.vocab[tokenizer.pad_token]).long().to(device)
                optimizer.zero_grad()
                
                logits = mlm_model(input_ids, attention_mask=attention_mask)
                loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                
                # Compute accuracy on masked tokens.
                preds = torch.argmax(logits, dim=-1)
                mask = labels != -100
                correct = (preds == labels).masked_select(mask).sum().item()
                total = mask.sum().item()
                
                epoch_correct += correct
                epoch_total += total
                batch_count += 1
                
                if batch_count % 100 == 0:
                    batch_acc = 100 * correct / total if total > 0 else 0
                    logger.info("Epoch %d Batch %d: Loss: %.4f, Batch Accuracy: %.2f%%", epoch+1, batch_count, loss.item(), batch_acc)
            avg_loss = epoch_loss / len(dataloader)
            avg_acc = 100 * epoch_correct / epoch_total if epoch_total > 0 else 0
            logger.info("Epoch %d/%d: Avg Loss: %.4f, Avg Accuracy: %.2f%%", epoch+1, args.num_epochs, avg_loss, avg_acc)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_wall_sec = time.perf_counter() - optimization_start

        torch.save(mlm_model.state_dict(), f"stored/{args.model_file}")
        logger.info("Saved trained model to stored/%s", args.model_file)

        mlm_stage_wall_sec = time.perf_counter() - mlm_stage_start
        timing = {
            "mlm_optimization_wall_sec": optimization_wall_sec,
            "mlm_stage_wall_sec": mlm_stage_wall_sec,
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "device": str(device),
            "model_file": args.model_file,
        }
        timing_file = f"stored/{args.model_file}.timing.json"
        with open(timing_file, "w") as f:
            json.dump(timing, f, indent=2)
        logger.info(
            "MLM optimization wall-clock: %.2f s (%.2f min, %.2f h)",
            optimization_wall_sec,
            optimization_wall_sec / 60,
            optimization_wall_sec / 3600,
        )
        logger.info(
            "MLM stage wall-clock (setup + training + checkpoint): %.2f s (%.2f min, %.2f h)",
            mlm_stage_wall_sec,
            mlm_stage_wall_sec / 60,
            mlm_stage_wall_sec / 3600,
        )
        logger.info("Saved MLM timing metadata to %s", timing_file)
        uids, sequences = load_and_process_sequences(args.csv_file, args.event_set, args.max_events)
    
    generate_and_save_embeddings(
        mlm_model, uids, sequences, tokenizer, max_seq_tokens,
        f"stored/{args.embedding_file}",
        embedding_method=args.embedding_method,
        embedding_layer=args.embedding_layer
    )

# -----------------------------------------------------------------------------
# 9. Argument Parser for Configurable Parameters and Mode
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLM Training and Embedding Generation with BERT (Geohash 8 split into 4 parts with prefixes)")
    parser.add_argument("--csv_file", type=str, default="sequence_data.csv",
                        help="Path to the CSV file containing event data")
    parser.add_argument("--event_set", type=str, default="personal",
                        help="Filter for event_set column (e.g., 'personal')")
    parser.add_argument("--max_events", type=int, default=100,
                        help="Maximum number of events per user (will truncate/pad accordingly)")
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate for optimizer")
    parser.add_argument("--hidden_size", type=int, default=128,
                        help="Hidden size for the BERT model (and embedding size)")
    parser.add_argument("--num_hidden_layers", type=int, default=4,
                        help="Number of hidden layers in the BERT model")
    parser.add_argument("--num_attention_heads", type=int, default=4,
                        help="Number of attention heads in the BERT model")
    parser.add_argument("--mask_prob", type=float, default=0.15,
                        help="Probability of masking a token for MLM")
    parser.add_argument("--embedding_file", type=str, default="user_embeddings.pkl",
                        help="Output file to store user embeddings")
    parser.add_argument("--embedding_method", type=str, default="cls", choices=["cls", "mean", "max"],
                        help="Pooling method to generate embeddings: 'cls', 'mean', or 'max'")
    parser.add_argument("--embedding_layer", type=int, default=-1,
                        help="Layer index to use for generating embeddings. Use -1 for final layer (default).")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "embed"],
                        help="Mode: 'train' to train and save model, 'embed' to load a trained model and generate embeddings.")
    parser.add_argument("--model_file", type=str, default="trained_model.pth",
                        help="File to save/load the trained model")
    parser.add_argument("--gpu_id", type=int, default=0,
                        help="ID of the GPU to use (if available)")
    args = parser.parse_args()
    
    main(args)
