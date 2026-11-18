#!/usr/bin/env python3
"""Audit current source files without displaying matched sensitive values."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import warnings


SKIP_DIRS = {'.git', '.venv', 'venv', '__pycache__', 'dist', '.pytest_cache'}
# Public dataset provenance explicitly selected for the preparation workflow.
PUBLIC_DATASET_ID = 'capital' + 'one/PRES'
PUBLIC_DATASET_REFERENCES = re.compile(
    re.escape('https://huggingface.co/datasets/' + PUBLIC_DATASET_ID)
    + r'(?![\w.-])(?:[/?#][^\s"\x27<>]*)?|["\x27]'
    + re.escape(PUBLIC_DATASET_ID) + r'["\x27]')
# Fragments keep the auditor's own source free of the references it detects.
RULES = {
    'organization-reference': re.compile(
        r'capital' + r'[\s_-]*one|\bcap' + r'one\b|cof' + r'-sandbox', re.I),
    'private-infrastructure': re.compile(
        r'\b(?:cube' + r'-(?:storage|interns|training)|bam' + r'odel\w*)\b|'
        r'fm_model' + r'\.models\.|\b(?:from|import)\s+c' + r'1(?:\.|\b)|'
        r'https?://[^\s/"\x27]+\.(?:corp|internal)(?:[/:\s"\x27]|$)|'
        r'\b(?:VAULT_AUTH_ROLE|VAULT_SECRET_PATH|lockbox_id)\s*[:=]', re.I),
    'personal-path-or-identifier': re.compile(
        r'/(?:home|Users|projects)/(?!you(?:/|\b)|user(?:/|\b))[^\s"\x27<>]+|'
        r'\b(?:jri' + r'105|hdn' + r'974|qga' + r'328|biy' + r'd)\b', re.I),
    'credential-format': re.compile(
        r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|'
        r'\b(?:gh[pousr]_|github_pat_)[A-Za-z0-9_]{20,}\b|'
        r'\bhf_[A-Za-z0-9]{20,}\b|\bxox[baprs]-[A-Za-z0-9-]{15,}\b|'
        r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|'
        r'https?://[^\s/:]+:[^\s/@]+@[^\s/]+'),
    'literal-credential': re.compile(
        r'''(?i)["']?(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|passwd|secret[_-]?key|wandb[_-]?api[_-]?key)["']?\s*[:=]\s*["'](?!<|YOUR_|REPLACE_|example|placeholder)[^"'\n]{8,}["']'''),
}


def source_paths(root):
    """Select tracked files and non-ignored additions, including edited content."""
    root = Path(root)
    if (root / '.git').exists():
        result = subprocess.run(
            ['git', '-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z'],
            capture_output=True, check=True)
        return sorted({Path(p.decode()) for p in result.stdout.split(b'\0')
                       if p and (root / p.decode()).exists()})
    return sorted(p.relative_to(root) for p in root.rglob('*')
                  if p.is_file() and not SKIP_DIRS.intersection(p.relative_to(root).parts))


def audit(root, paths=None):
    root = Path(root)
    paths = source_paths(root) if paths is None else paths
    findings = []

    def add(path, line, rule):
        findings.append({'file': str(path), 'line': line, 'rule': rule})

    for rel in paths:
        rel = Path(rel)
        path = root / rel
        if rel.is_absolute() or '..' in rel.parts or SKIP_DIRS.intersection(rel.parts):
            add(rel, 1, 'excluded-path')
            continue
        if path.is_symlink():
            add(rel, 1, 'source-symlink')
            continue
        if rel.name.startswith('.env') or rel.suffix in {'.pem', '.key', '.p12', '.pfx'} or '.aws' in rel.parts or '.kube' in rel.parts or rel.name.endswith(('.local.yaml', '.local.yml')):
            add(rel, 1, 'local-configuration')
        raw = path.read_bytes()
        # Decode binary strings as well; image pixels require separate visual review.
        text = raw.decode('utf-8', errors='replace')
        public_spans = [match.span() for match in PUBLIC_DATASET_REFERENCES.finditer(text)]
        for name, pattern in RULES.items():
            for match in pattern.finditer(text):
                if name == 'organization-reference' and any(
                        start <= match.start() and match.end() <= end for start, end in public_spans):
                    continue
                add(rel, text.count('\n', 0, match.start()) + 1, name)
        if rel.suffix == '.py':
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', SyntaxWarning)
                    ast.parse(text, filename=str(rel))
            except SyntaxError as error:
                add(rel, error.lineno or 1, 'python-syntax')
        if rel.suffix == '.ipynb':
            try:
                notebook = json.loads(text)
                if notebook.get('nbformat') != 4 or not isinstance(notebook.get('cells'), list):
                    raise ValueError('Unsupported notebook format')
                if set(notebook.get('metadata', {})) - {'kernelspec', 'language_info'}:
                    add(rel, 1, 'notebook-session-metadata')
                for cell in notebook['cells']:
                    if cell.get('outputs') or cell.get('execution_count') is not None:
                        add(rel, 1, 'saved-notebook-output')
                    if set(cell.get('metadata', {})) - {'tags'}:
                        add(rel, 1, 'notebook-session-metadata')
            except (ValueError, TypeError, KeyError):
                add(rel, 1, 'notebook-format')
    return findings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    paths = source_paths(args.root)
    findings = audit(args.root, paths)
    if args.json:
        print(json.dumps({'file_count': len(paths), 'findings': findings}, indent=2))
    else:
        for finding in findings:
            print('{file}:{line}: {rule}'.format(**finding))
        print(f'Checked {len(paths)} source files; {len(findings)} findings. Git history and ignored files are excluded.')
    return bool(findings)


if __name__ == '__main__':
    sys.exit(main())
