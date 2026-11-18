#!/usr/bin/env python3
"""Create an audited source snapshot, excluding Git history and ignored files."""
import argparse
from pathlib import Path
import sys
from zipfile import ZIP_DEFLATED, ZipFile

from audit_public_tree import audit, source_paths


def export_source(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    paths = [p for p in source_paths(root) if (root / p).resolve() != output]
    findings = audit(root, paths)
    if findings:
        for item in findings:
            print('{file}:{line}: {rule}'.format(**item), file=sys.stderr)
        raise ValueError('Source audit failed; no archive was written.')
    if output.exists():
        raise FileExistsError('Output exists; choose a new archive path.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, 'x', compression=ZIP_DEFLATED) as archive:
        for rel in paths:
            archive.write(root / rel, Path('event-seq-graph') / rel)
    # Verify the archive itself; no working-directory or Git metadata is added.
    with ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive integrity check failed.')
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    try:
        count = export_source(args.root, args.output)
    except (ValueError, FileExistsError) as error:
        parser.exit(1, f'{error}\n')
    print(f'Exported {count} audited source files to {args.output}')


if __name__ == '__main__':
    main()
