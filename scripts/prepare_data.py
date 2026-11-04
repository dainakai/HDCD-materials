#!/usr/bin/env python3
"""Verify and unpack the bundled numerical inputs and particle-measurement weights."""
import argparse
import hashlib
import json
from pathlib import Path
import os
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def contained(root, name):
    root = Path(root).resolve()
    path = Path(name)
    destination = (root / path).resolve()
    if path.is_absolute() or '..' in path.parts or destination == root or root not in destination.parents:
        raise ValueError(f'Unsafe relative path: {name}')
    return destination


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(profile='all', root=ROOT):
    import zstandard as zstd
    root = Path(root).resolve()
    manifest = json.loads((root / 'data_sources/manifest.json').read_text())
    records = []
    for item in manifest['objects']:
        if profile != 'all' and item['profile'] != profile:
            continue
        destination = contained(root, item['destination'])
        parts = [(part, contained(root, part['path'])) for part in item['parts']]
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            if digest(destination) != item['sha256']:
                raise ValueError(f'Existing file differs; move it aside before retrying: {destination}')
        else:
            for part, source in parts:
                if digest(source) != part['sha256']:
                    raise ValueError(f'Input checksum mismatch: {part["path"]}')
            fd, temporary = tempfile.mkstemp(prefix=destination.name + '.', dir=destination.parent)
            try:
                with os.fdopen(fd, 'wb') as output:
                    with zstd.ZstdDecompressor().stream_writer(output, closefd=False) as decoder:
                        for part, source_path in parts:
                            with source_path.open('rb') as source:
                                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                                    decoder.write(block)
                if digest(temporary) != item['sha256']:
                    raise ValueError(f'Decoded checksum mismatch: {destination}')
                os.replace(temporary, destination)
            finally:
                if Path(temporary).exists():
                    Path(temporary).unlink()
        records.append({'file': item['destination'], 'sha256': item['sha256']})
    return {'status': 'PASS', 'profile': profile, 'files': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['all', 'test', 'measurement'], default='all')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = prepare(args.profile)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(f"PASS: {len(report['files'])} verified files ({args.profile}).")
