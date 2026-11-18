#!/usr/bin/env python3
"""Plan fixed paper rows without importing training code; execution requires --run.

Examples:
  python scripts/reproduce_paper.py --suite link --datasets brightkite --methods bridge-gcn --seeds 0 1 2
  python scripts/reproduce_paper.py --suite fraud --datasets movies --methods bridge-tfconv --dry-run

Paper settings and inferred source defaults are separated in the JSON registry.
An executable plan is a reconstruction using those defaults, not recovered run metadata.
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / 'configs' / 'paper_experiments.json'


def load_registry(path=DEFAULT_REGISTRY):
    with Path(path).open() as handle:
        return json.load(handle)


def argument_specs(path):
    """Read CLI declarations as syntax; never import GPU training modules."""
    specs = {}
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'add_argument'):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        flag = node.args[0].value
        if not isinstance(flag, str) or not flag.startswith('--'):
            continue
        kw = {item.arg: item.value for item in node.keywords}
        specs[flag[2:]] = {
            key: ast.literal_eval(kw[key]) for key in ('action', 'choices') if key in kw
        }
    return specs


def command_arguments(values, specs):
    result = []
    for name, value in sorted(values.items()):
        if name not in specs:
            raise ValueError(f'Unknown training flag --{name}; registry/source mismatch')
        spec = specs[name]
        if value is None:
            continue
        action = spec.get('action')
        if action in ('store_true', 'store_false'):
            if not isinstance(value, bool):
                raise ValueError(f'--{name} requires a boolean in the registry')
            if value == (action == 'store_true'):
                result.append('--' + name)
            continue
        if 'choices' in spec and value not in spec['choices']:
            raise ValueError(f'Unsupported --{name}={value!r}')
        result.extend(['--' + name, str(value)])
    return result


def build_plan(registry, suite, datasets=None, methods=None, seeds=None, root=ROOT):
    root = Path(root).resolve()
    suite_config = registry['suites'][suite]
    datasets = suite_config['datasets'] if datasets is None else datasets
    methods = suite_config['default_methods'] if methods is None else methods
    seeds = registry['seed_policy']['default'] if seeds is None else seeds
    if not datasets or not methods or not seeds:
        raise ValueError('Select at least one dataset, method and seed')
    if len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers')
    if len(set(datasets)) != len(datasets) or len(set(methods)) != len(methods):
        raise ValueError('Dataset and method selections must not contain duplicates')
    rows = {(row['dataset'], row['method']): row for row in registry['rows'] if row['suite'] == suite}
    plan = []
    for dataset in datasets:
        if dataset not in suite_config['datasets']:
            raise ValueError(f'Dataset {dataset!r} is not in suite {suite!r}')
        for method in methods:
            if (dataset, method) not in rows:
                available = sorted({key[1] for key in rows})
                raise ValueError(f'Unknown method {method!r}; available: {", ".join(available)}')
            row = rows[dataset, method]
            entry = registry['entrypoints'][row['entrypoint']]
            specs = argument_specs(root / row['entrypoint'])
            for seed in seeds:
                # All retained scalar defaults are frozen in the registry; operational paths
                # and caller-selected seeds are set here, not inferred as paper settings.
                values = dict(entry['inferred_code_defaults'])
                values.update(entry.get('inferred_launcher_overrides', {}))
                values['dataset'] = dataset
                values.update(row['arguments'])
                values['seed'] = seed
                if 'experiment_name' in specs:
                    values['experiment_name'] = f'paper-{suite}-{dataset}-{method}-seed{seed}'
                if suite == 'fraud' and 'data_root' in specs:
                    values['data_root'] = str(root / 'data' / 'processed')
                    if 'log_dir' in specs:
                        values['log_dir'] = str(root / 'notebooks' / 'logs_paper')
                required = list(row['required_files'])
                if suite == 'fraud':
                    required.append(f'data/processed/amazon-{dataset}/splits/{dataset}_canonical_user_split_seed{seed}.csv')
                plan.append({
                    'suite': suite, 'dataset': dataset, 'method': method, 'seed': seed,
                    'cwd': str(root / entry['cwd']),
                    'command': [sys.executable, Path(row['entrypoint']).name] + command_arguments(values, specs),
                    'paper_settings': row['paper_settings'], 'provenance': row['provenance'],
                    'unresolved': row['unresolved'],
                    'required_files': [str(root / filename) for filename in required],
                    'configuration_basis': 'PDF-stated settings plus frozen, explicitly inferred code defaults',
                })
    return plan


def run_plan(plan):
    """Validate the whole selection before launching any one of its training jobs."""
    errors = []
    for job in plan:
        name = f"{job['suite']}/{job['dataset']}/{job['method']}/seed{job['seed']}"
        errors.extend(f'{name}: unresolved: {item}' for item in job['unresolved'])
        errors.extend(f'{name}: missing required artifact: {path}'
                      for path in job['required_files'] if not Path(path).is_file())
    if errors:
        raise ValueError('Refusing --run before training:\n' + '\n'.join(errors))
    for job in plan:
        subprocess.run(job['command'], cwd=job['cwd'], check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', required=True, choices=['link', 'fraud'])
    parser.add_argument('--datasets', nargs='+')
    parser.add_argument('--methods', nargs='+', help='Registry method IDs, e.g. static-gcn two-stage-gat bridge-tfconv tgn sbert')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--registry', type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument('--json', action='store_true', help='Print full plans including provenance and input requirements')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true', help='Print commands only (the default)')
    mode.add_argument('--run', action='store_true', help='Execute only after checking unresolved settings and required inputs')
    args = parser.parse_args(argv)
    try:
        registry = load_registry(args.registry)
        plan = build_plan(registry, args.suite, args.datasets, args.methods, args.seeds)
        if args.json:
            print(json.dumps(plan, indent=2))
        else:
            print('# Configuration basis: PDF-stated settings + inferred retained-code defaults.')
            for job in plan:
                print(f"# {job['suite']}/{job['dataset']}/{job['method']} seed={job['seed']} cwd={job['cwd']}")
                print(f"# PDF table {job['provenance']['table']}, page {job['provenance']['page']}")
                for item in job['unresolved']:
                    print('# UNRESOLVED: ' + item)
                print(shlex.join(job['command']))
        if args.run:
            run_plan(plan)
        return 0
    except (ValueError, KeyError, OSError, SyntaxError) as exc:
        parser.exit(2, f'{exc}\n')


if __name__ == '__main__':
    main()
