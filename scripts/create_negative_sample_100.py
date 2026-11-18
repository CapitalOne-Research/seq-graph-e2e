#!/usr/bin/env python3
"""Keep the first K rows of each original negative-sample block, in order.

Extracted from the Amazon and Brightkite create_subset notebooks. Defaults match
their conversion: first 100 of every 1,000 rows, without grouping or resampling.
Uses streaming CSV I/O so the complete source file need not fit in memory.
"""
import argparse
from contextlib import ExitStack
import csv
import os
from pathlib import Path
import tempfile


def convert_file(source, output, *, block_size=1000, keep=100, positives=None, overwrite=False):
    source, output = Path(source), Path(output)
    if not 0 < keep <= block_size:
        raise ValueError('Require 0 < keep <= block_size')
    if source.resolve() == output.resolve():
        raise ValueError('Source and output must be different files')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; pass --overwrite to replace it')

    temporary = None
    try:
        with ExitStack() as stack:
            reader = csv.reader(stack.enter_context(source.open(newline='', encoding='utf-8-sig')))
            header = next(reader, None)
            if not header or 'uid' not in header or 'other_uid' not in header or len(set(header)) != len(header):
                raise ValueError(f'{source}: expected unique columns including uid and other_uid')
            uid_column = header.index('uid')
            positive_reader = None
            if positives is not None:
                positive_reader = csv.DictReader(stack.enter_context(Path(positives).open(newline='', encoding='utf-8-sig')))
                if not positive_reader.fieldnames or 'uid' not in positive_reader.fieldnames:
                    raise ValueError(f'{positives}: missing uid column')

            handle = stack.enter_context(tempfile.NamedTemporaryFile(
                mode='w', newline='', encoding='utf-8', dir=output.parent,
                prefix=f'.{output.name}.', suffix='.tmp', delete=False))
            temporary = Path(handle.name)
            writer = csv.writer(handle, lineterminator='\n')
            writer.writerow(header)
            read_count = written_count = 0
            block_uid = None
            for row in reader:
                if not row:  # Match the notebook's handling of blank CSV lines.
                    continue
                if len(row) != len(header):
                    raise ValueError(f'{source}: incorrect column count at data row {read_count + 1}')
                offset = read_count % block_size
                if offset == 0:
                    block_uid = row[uid_column]
                    if positive_reader is not None:
                        positive = next(positive_reader, None)
                        if positive is None or positive['uid'] != block_uid:
                            raise ValueError(f'{source}: positive UID/block mismatch at block {read_count // block_size + 1}')
                if row[uid_column] != block_uid:
                    raise ValueError(f'{source}: UID changes inside block {read_count // block_size + 1}')
                if offset < keep:
                    writer.writerow(row)
                    written_count += 1
                read_count += 1

            if not read_count or read_count % block_size:
                raise ValueError(f'{source}: expected complete {block_size}-row blocks; found {read_count} data rows')
            if positive_reader is not None and next(positive_reader, None) is not None:
                raise ValueError(f'{source}: fewer negative blocks than positive rows')
        os.replace(temporary, output)
        temporary = None
        return read_count, written_count
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True, type=Path, nargs='+', help='one or more prepared task directories')
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--block-size', type=int, default=1000, help='original negatives per positive (default: 1000)')
    parser.add_argument('--keep', type=int, default=100, help='first rows retained per block (default: 100)')
    parser.add_argument('--overwrite', action='store_true', help='replace existing output files after successful validation')
    args = parser.parse_args()
    if not 0 < args.keep <= args.block_size:
        parser.error('Require 0 < --keep <= --block-size')

    jobs = []
    for directory in dict.fromkeys(args.data_dir):
        for split in dict.fromkeys(args.splits):
            source = directory / f'relational_{split}_negative_sample.csv'
            output = directory / f'relational_{split}_negative_sample_{args.keep}.csv'
            positive = directory / f'relational_{split}.csv'
            if not source.is_file():
                parser.error(f'Missing input: {source}')
            if output.exists() and not args.overwrite:
                parser.error(f'{output} exists; pass --overwrite to replace it')
            jobs.append((source, output, positive if positive.is_file() else None))
    for source, output, positive in jobs:
        try:
            read_count, written_count = convert_file(
                source, output, block_size=args.block_size, keep=args.keep,
                positives=positive, overwrite=args.overwrite)
        except (OSError, ValueError, csv.Error) as error:
            parser.exit(1, f'{error}\n')
        alignment = 'checked against positives' if positive is not None else 'no positive CSV; block UID/count checks only'
        print(f'{source.name}: {read_count:,} -> {written_count:,} rows; {alignment}; wrote {output}')


if __name__ == '__main__':
    main()
