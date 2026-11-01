"""Package the report, implementation, and evidence without caches or environments."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_DIRS = {'__pycache__', '.venv', '.git', 'node_modules'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'milestone2.zip')
    args = parser.parse_args()
    files = sorted(p for p in ROOT.rglob('*') if p.is_file()
                   and not set(p.relative_to(ROOT).parts) & EXCLUDED_DIRS
                   and p.suffix not in {'.pyc', '.pyo', '.zip'}
                   and p.name != 'MANIFEST.json')
    required = {'README.md', 'report/milestone2_report.pdf', 'read_the_boss/env.py',
                'tests/test_environment.py', 'evidence/tests.txt'}
    names = {str(p.relative_to(ROOT)) for p in files}
    if not required <= names:
        raise SystemExit(f'Missing submission files: {sorted(required - names)}')
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in files}
    (ROOT / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n')
    files.append(ROOT / 'MANIFEST.json')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(args.output, 'w', compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, Path('milestone2') / path.relative_to(ROOT))
    with ZipFile(args.output) as archive:
        if archive.testzip() is not None:
            raise SystemExit('ZIP integrity check failed.')
    print(f'Created {args.output} with {len(files)} files ({args.output.stat().st_size:,} bytes).')


if __name__ == '__main__':
    main()
