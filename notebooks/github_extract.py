import argparse
import glob
import pandas as pd
import json
import gzip
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Convert downloaded GH Archive JSON files to events CSV.")
parser.add_argument("--input-glob", required=True, help="Glob for local .json.gz files")
parser.add_argument("--output", default="github-events-1m.csv")
args = parser.parse_args()
file_paths = sorted(glob.glob(args.input_glob))
if not file_paths:
    parser.error("No files matched --input-glob")
output_file = args.output

# Flag to indicate whether the header has been written.
header_written = False

for file_path in tqdm(file_paths, desc="Processing files"):
    filtered_records = []

    # Open the gzipped file in text mode.
    with gzip.open(file_path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue  # Skip any empty lines.
            record = json.loads(line)
            
            # Build the filtered dictionary.
            filtered_record = {
                "actor.id": record.get("actor", {}).get("id"),
                "actor.login": record.get("actor", {}).get("login"),
                "repo.id": record.get("repo", {}).get("id"),
                "type": record.get("type"),
                "created_at": record.get("created_at"),
                # Create a single column "payload_fields" that lists all immediate field names in the payload.
                "payload_fields": list(record.get("payload", {}).keys()),
                "payload.action": record.get("payload", {}).get("action"),
                "payload.ref_type": record.get("payload", {}).get("ref_type"),
            }

            if '[bot]' not in filtered_record['actor.login']:
                filtered_records.append(filtered_record)
            
    
    # Convert the filtered records list to a pandas DataFrame.
    df = pd.DataFrame(filtered_records)

    df["payload_fields"] = df["payload_fields"].apply(lambda x: tuple(x) if isinstance(x, list) else x)
    df["action"] = df["type"].str.replace('Event', '').str.cat(df['payload.action'].str.capitalize(), na_rep='').str.cat(df['payload.ref_type'].str.capitalize(), na_rep='') 
    df["timestamp"] = pd.to_datetime(df["created_at"]).astype('int64') // 10**9

    dfs = df[['actor.id', 'actor.login', 'repo.id', 'timestamp', 'action']]
    # print(f"This round df: {dfs.shape}")


    dfs.to_csv(output_file, mode='a', index=False, header=not header_written)
    header_written = True  # After the first write, disable writing the header for subsequent fi

print("Processing complete. Combined CSV written to", output_file)
