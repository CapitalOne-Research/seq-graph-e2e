#!/usr/bin/env python3
import re
import os
import shutil
from pathlib import Path
from datetime import datetime
import pandas as pd
from collections import defaultdict
import argparse

# START / FINISH regex
# This only works for Check In Prediction. For Friend Recommendation & Co-review, the elapsed time is extracted from the results line
RE_START  = re.compile(r'^\[(?P<ts>[\d\- :]+)\] START')
RE_FINISH = re.compile(r'^\[(?P<ts>[\d\- :]+)\] FINISH.*? - SUCCESS')

# 3) Metric patterns keyed by filename
METRIC_PATTERNS = {
    'seqmodel_n2v_input.log':  re.compile(
        r"Final Test Evaluation Result: \{'global': \{"
        r"'MRR': (?P<MRR>[0-9.]+), 'Hits@3': (?P<Hits3>[0-9.]+), 'Hits@5': (?P<Hits5>[0-9.]+), "
        r"'Hits@10': (?P<Hits10>[0-9.]+), 'Hits@50': (?P<Hits50>[0-9.]+)"
    ),
    'seqmodel_n2v_post.log':  re.compile(
        r"Final Test Evaluation Result: \{'global': \{"
        r"'MRR': (?P<MRR>[0-9.]+), 'Hits@3': (?P<Hits3>[0-9.]+), 'Hits@5': (?P<Hits5>[0-9.]+), "
        r"'Hits@10': (?P<Hits10>[0-9.]+), 'Hits@50': (?P<Hits50>[0-9.]+)"
    ),
    'plain_sequence_model.log': re.compile(
        r"Final Test Evaluation Result: \{'global': \{"
        r"'MRR': (?P<MRR>[0-9.]+), 'Hits@3': (?P<Hits3>[0-9.]+), 'Hits@5': (?P<Hits5>[0-9.]+), "
        r"'Hits@10': (?P<Hits10>[0-9.]+), 'Hits@50': (?P<Hits50>[0-9.]+)"
    ),
    'static_PR_GAT.log': re.compile(
        r"GAT: MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'static_P_GAT.log': re.compile(
        r"GAT: MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'static_P_GCN.log': re.compile(
        r"GCN: MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'static_PR_GCN.log': re.compile(
        r"GCN: MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'temporal_TGN_P.log': re.compile(
        r"TGN \(Personal Prediction\): MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), "
        r"Hits@5: (?P<Hits5>[0-9.]+), Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'temporal_TGN_PR.log': re.compile(
        r"TGN \(Personal Prediction\): MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), "
        r"Hits@5: (?P<Hits5>[0-9.]+), Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'temporal_DyRep_P.log': re.compile(
        r"DyRep \(Personal Prediction\): MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), "
        r"Hits@5: (?P<Hits5>[0-9.]+), Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    'temporal_DyRep_PR.log': re.compile(
        r"DyRep \(Personal Prediction\): MRR: (?P<MRR>[0-9.]+), Hits@3: (?P<Hits3>[0-9.]+), "
        r"Hits@5: (?P<Hits5>[0-9.]+), Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+)"
    ),
    ### Graph logs
    'DyRep_temporal.log': re.compile(
        r"DyRep: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'TGN_temporal.log': re.compile(
        r"TGN: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GAT_no_seq.log': re.compile(
        r"GAT: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GCN_no_seq.log': re.compile(
        r"GCN: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GAT_seq_as_feats.log': re.compile(
        r"GAT: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GCN_seq_as_feats.log': re.compile(
        r"GCN: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GAT_seq_as_nodes.log': re.compile(
        r"GAT: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
    'GCN_seq_as_nodes.log': re.compile(
        r"GCN: MRR: (?P<MRR>[0-9.]+), Hits@5: (?P<Hits5>[0-9.]+), "
        r"Hits@10: (?P<Hits10>[0-9.]+), Hits@50: (?P<Hits50>[0-9.]+), Hits@100: (?P<Hits100>[0-9.]+), Time: (?P<runtime_s>[0-9.]+)s"
    ),
}

def format_for_log_parse(stored_dir: str):
    """ Helper function to move from directories per configuration (with multiple runs in each) to 1 run of every configuration per directory. Built for Friend Recommendation & Co-Review tasks."""
    LOG_ROOTS = [stored_dir + f"/formatted/logs{i}" for i in range(5)]
    for new_dir in LOG_ROOTS:
        os.makedirs(new_dir, exist_ok = True)
        
    for file in os.listdir(stored_dir):
        if not file.endswith('.log'):
            continue
        if file.lower() == 'make_seq.log':
            continue

        model = file.split('_',1)[0]
        run_number = int(file.split('_',1)[1][0])
        if run_number not in [0,1,2,3,4]:
            raise ValueError(f"{file} does not match expected format")
        if 'personal_nodes' in file:
            new_name = model + '_seq_as_nodes.log'
        elif 'w_seq' in file:
            new_name = model + '_seq_as_feats.log'
        elif 'DyRep' in file or 'TGN' in file:
            new_name = model + '_temporal.log'
        else:
            new_name = model + '_no_seq.log'
        shutil.copyfile(os.path.join(stored_dir, file), os.path.join(LOG_ROOTS[run_number], new_name))


def main(log_parent: str):
    data = defaultdict(list)
    LOG_ROOTS = [log_parent / f"logs{i}" for i in range(5)]
    for run_dir in LOG_ROOTS:
        run_name = run_dir.name  # logs0, logs1, ...
        for logfile in sorted(run_dir.glob("*.log")):
            fname = logfile.name
            if fname == "node2vec_embedding.log":
                continue
    
            lines = logfile.read_text().splitlines()
    
            # find start/end
            t_start = None
            t_end = None
            for L in lines:
                m = RE_START.match(L)
                if m:
                    t_start = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                    break
            for L in reversed(lines):
                m = RE_FINISH.match(L)
                if m:
                    t_end = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S")
                    break
    
            runtime = (t_end - t_start).total_seconds() if t_start and t_end else None
    
            # find final metrics
            pat = METRIC_PATTERNS.get(fname)
            extracted = {}
    
            if pat:
                for L in reversed(lines):
                    mm = pat.search(L)
                    if mm:
                        for metric in pat.groupindex.keys():
                            # Add @ if the metric is "Hits*"
                            value = float(mm.group(metric))
                            if metric.startswith("Hits"):
                                extracted["Hits@" + metric[4:]] = value
                            else:
                                extracted[metric] = value
                        break
            else:
                print(f"Could not find regexp pattern for file: {fname}")

            if runtime:
                extracted['runtime_s'] = runtime
                
            # append under this algorithm
            data[fname].append( (run_name, extracted) )
    
    # For each algorithm, build a DataFrame, append mean/std, and save CSV
    out_dir = os.path.join(log_parent, "all_results")
    os.makedirs(out_dir, exist_ok=True)
    
    for algo, runs in data.items():
        # build DataFrame indexed by run_name
        df = pd.DataFrame(
            { run: vals for run, vals in runs }
        ).T  # transpose so rows are runs
    
        # compute summary rows
        mean_row = df.mean(axis=0)
        std_row  = df.std(axis=0)
    
        df.loc['mean'] = mean_row
        df.loc['std']  = std_row
    
        # save
        csv_path = os.path.join(out_dir,f"{algo.replace('.log','')}.csv")
        df.to_csv(csv_path, index_label="run")
    
        print(f"→ {csv_path} ({len(runs)} runs + 2 summary rows)")
    
    print("Done.")

if __name__ == "__main__":
    # Parse command-line argument
    parser = argparse.ArgumentParser(description="Parse logs from multiple runs")
    parser.add_argument("log_parent", type=Path, help="Parent directory containing logs0, logs1, etc.")
    args = parser.parse_args()
    main(args.log_parent)
    