"""Collect held-out link metrics with validation-only model selection.

E2E runs must log best_test_* (or identify an evaluated *_best checkpoint).
Static/temporal summary test_* rows follow those trainers' contract of testing
the validation-selected state; run them with early stopping enabled.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import re
from statistics import mean, stdev

METRICS = ('mrr', 'hits_at_1', 'hits_at_3', 'hits_at_5', 'hits_at_10', 'hits_at_50')


class IncompleteResultsError(ValueError):
    """The selected experiment cannot yet produce a held-out result table."""


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(values, prefix=''):
    found = {}
    for key, value in values.items():
        key = str(key).lower().replace('@', '_at_')
        if prefix and not key.startswith(prefix):
            continue
        metric = key[len(prefix):]
        number = _finite(value)
        if metric in METRICS and number is not None:
            found[metric] = number
    return found


def extract_metrics(rows, run_type):
    """Select the validation epoch; never substitute its scores for test scores."""
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError('Expected a list of epoch/summary dictionaries')
    validations = []
    for row in rows:
        nested = row.get('evaluation_metrics') or {}
        values = _metrics(nested) if isinstance(nested, dict) else {}
        values.update(_metrics(row, 'val_'))
        if 'mrr' in values:
            validations.append((values['mrr'], row.get('epoch')))
    if not validations:
        raise IncompleteResultsError('No finite validation MRR; cannot select a configuration')
    # max is stable on ties, matching the trainers' strict-greater improvement rule.
    val_mrr, epoch = max(validations, key=lambda item: item[0])
    selected = None
    provenance = None
    for row in rows:
        best = _metrics(row, 'best_test_')
        if 'mrr' in best:
            selected, provenance = best, 'best_test'
    if selected is None:
        for row in rows:
            test = _metrics(row, 'test_')
            checkpoint = str(row.get('checkpoint_used', ''))
            best_checkpoint = bool(re.search(r'_best\.(?:pth|pt)$', checkpoint))
            summary_contract = (run_type in ('static', 'temporal')
                                and row.get('summary') is True
                                and row.get('selection') == 'validation_mrr')
            explicit_epoch = row.get('checkpoint_epoch') is not None and row['checkpoint_epoch'] == epoch
            if 'mrr' in test and (best_checkpoint or summary_contract or explicit_epoch):
                selected, provenance = test, 'best_checkpoint' if best_checkpoint else 'val_selected_summary'
    return {'val_mrr': val_mrr, 'selected_epoch': epoch,
            'test_metrics': selected or {}, 'test_provenance': provenance}


def parse_filename(path, run_type):
    stem = Path(path).stem
    seed_match = re.search(r'_seed(\d+)(.*)$', stem)
    if not seed_match:
        raise ValueError('Missing _seed<number> in run filename')
    config = stem[:seed_match.start()] + seed_match.group(2)
    spec = stem.removeprefix('e2e_').split('_', 1)[0]
    spec = re.sub(r'\d+x\d+$', '', spec).lower()
    names = {'gcn': 'GCN', 'gat': 'GAT', 'transformerconv': 'TFConv',
             'dyrep': 'DyRep', 'tgn': 'TGN', 'tncn': 'TNCN', 'nat': 'NAT',
             'tcl': 'TCL', 'graphmixer': 'GraphMixer', 'tpnet': 'TPNet',
             'dygformer': 'DyGFormer', 'cawn': 'CAWN', 'tgat': 'TGAT', 'ctan': 'CTAN'}
    if run_type == 'e2e':
        backbone = spec.split('-')
        method = 'BRIDGE-' + names[backbone[0]] if len(set(backbone)) == 1 and backbone[0] in names else spec
    else:
        method = names.get(spec, spec)
        if '_use_seq' in stem:
            method += ' + S'
    tag = seed_match.group(2).lstrip('_')
    if tag:
        method += f' [{tag}]'
    return {'seed': int(seed_match.group(1)), 'configuration': config, 'method': method}


def collect_runs(task_dir, datasets):
    """Read recognized training-result locations and retain missing-test runs."""
    task_dir = Path(task_dir)
    runs = []
    for dataset in datasets:
        sources = [('static', task_dir / 'logs' / f'static-results-{dataset}'),
                   ('temporal', task_dir / 'logs' / f'temporal-results-{dataset}'),
                   ('e2e', task_dir / f'{dataset}_results_seeded')]
        if dataset == 'brightkite':
            sources.append(('e2e', task_dir / 'results_seeded'))
        for run_type, directory in sources:
            for path in sorted(directory.glob('*.json')):
                try:
                    metadata = parse_filename(path, run_type)
                    metrics = extract_metrics(json.loads(path.read_text()), run_type)
                except (ValueError, OSError) as error:
                    raise IncompleteResultsError(f'{path}: {error}') from error
                runs.append({'dataset': dataset, 'type': run_type, 'source': str(path), **metadata, **metrics})
    if not runs:
        raise IncompleteResultsError('No result logs found in the selected task directory')
    return runs


def summarize_runs(runs, expected_seeds=(0, 1, 2), metrics=METRICS):
    """Choose configurations by mean validation MRR across exactly the same seeds."""
    expected = set(expected_seeds)
    if not expected or len(expected) != len(expected_seeds):
        raise ValueError('Expected seeds must be distinct and nonempty')
    configurations = defaultdict(dict)
    for run in runs:
        if run['seed'] not in expected:
            continue
        key = (run['dataset'], run['type'], run['method'], run['configuration'])
        if run['seed'] in configurations[key]:
            raise IncompleteResultsError(f'Duplicate seed {run["seed"]} for {key}: do not count reruns as extra seeds')
        configurations[key][run['seed']] = run
    choices = defaultdict(list)
    for key, seeds in sorted(configurations.items()):
        if set(seeds) != expected:
            raise IncompleteResultsError(f'{key}: missing seeds {sorted(expected - set(seeds))}')
        validation = [_finite(seeds[seed]['val_mrr']) for seed in sorted(expected)]
        if any(value is None for value in validation):
            raise IncompleteResultsError(f'{key}: invalid validation MRR')
        choices[key[:3]].append((mean(validation), key[3], seeds))
    if not choices:
        raise IncompleteResultsError('No runs match the expected seed set')
    summaries = []
    for (dataset, run_type, method), candidates in sorted(choices.items()):
        # Test availability and magnitude never influence configuration selection.
        validation, config, seeds = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        missing = [(seed, metric) for seed in sorted(expected) for metric in metrics
                   if _finite(seeds[seed]['test_metrics'].get(metric)) is None]
        if missing:
            raise IncompleteResultsError(f'{dataset}/{method}, selected config {config}: missing held-out metrics {missing}; evaluate the validation-selected checkpoints')
        summary = {'dataset': dataset, 'type': run_type, 'method': method, 'configuration': config,
                   'seeds': sorted(expected), 'num_seeds': len(expected), 'validation_mrr_mean': validation,
                   'sources': [seeds[seed]['source'] for seed in sorted(expected)]}
        for metric in metrics:
            values = [float(seeds[seed]['test_metrics'][metric]) for seed in sorted(expected)]
            summary[f'test_{metric}_mean'] = mean(values)
            summary[f'test_{metric}_std'] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def latex_table(summaries, dataset):
    rows = [r'\begin{tabular}{l' + 'c' * len(METRICS) + '}', r'\toprule',
            r'Method & MRR & Hits@1 & Hits@3 & Hits@5 & Hits@10 & Hits@50 \\', r'\midrule']
    for row in summaries:
        if row['dataset'] != dataset:
            continue
        label = row['method'].replace('_', r'\_')
        values = [f"${row[f'test_{metric}_mean']:.4f} \\pm {row[f'test_{metric}_std']:.4f}$" for metric in METRICS]
        rows.append(' & '.join([label, *values]) + r' \\')
    rows.extend([r'\bottomrule', r'\end{tabular}'])
    return '\n'.join(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-dir', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--output', type=Path, required=True, help='JSON summary; adjacent per-dataset .tex files are also written')
    args = parser.parse_args()
    try:
        result = summarize_runs(collect_runs(args.task_dir, args.datasets), args.seeds)
    except ValueError as error:
        parser.exit(1, f'{error}\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    for dataset in args.datasets:
        args.output.with_name(f'{args.output.stem}_{dataset}.tex').write_text(latex_table(result, dataset) + '\n')
    print(f'Wrote {len(result)} held-out result rows to {args.output}')


if __name__ == '__main__':
    main()
