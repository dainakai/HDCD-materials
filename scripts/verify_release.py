#!/usr/bin/env python3
"""Verify every distributed file and the decoded inputs that are present."""
import argparse
import json
from pathlib import Path

from prepare_data import contained, digest

ROOT = Path(__file__).resolve().parents[1]


def verify(root=ROOT):
    root = Path(root).resolve()
    manifest_path = root / 'manifest.json'
    expected = (root / 'manifest.sha256').read_text().split()[0]
    if digest(manifest_path) != expected:
        raise ValueError('The file manifest checksum differs')
    manifest = json.loads(manifest_path.read_text())
    seen = set()
    for item in manifest['files']:
        name = item['path']
        if name in seen:
            raise ValueError(f'Duplicate manifest path: {name}')
        seen.add(name)
        path = contained(root, name)
        if path.stat().st_size != item['bytes'] or digest(path) != item['sha256']:
            raise ValueError(f'Distributed file differs: {name}')
    decoded = []
    for item in json.loads((root / 'data_sources/manifest.json').read_text())['objects']:
        path = contained(root, item['destination'])
        if path.exists():
            if digest(path) != item['sha256']:
                raise ValueError(f'Decoded file differs: {item["destination"]}')
            decoded.append(item['destination'])
    return {'status': 'PASS', 'distributed_files': len(seen), 'decoded_files_checked': decoded}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
