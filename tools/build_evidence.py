"""Regenerate report figures and test evidence from the shipped implementation.

Run from the project root after the demo and smoke commands in README.md.
The code image is a verbatim excerpt; the execution image uses saved outputs.
"""
from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import time
import tokenize
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import gymnasium
import numpy as np
from PIL import Image, ImageDraw
from read_the_boss import ReadTheBossEnv
from read_the_boss.render import font

EVIDENCE = ROOT / 'evidence'


def draw_text_panel(lines, title, destination, *, size=24, numbered=False):
    """Render supplied text exactly, with a neutral source/output viewer header."""
    mono = font(size, mono=True)
    header_font = font(size, bold=True)
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    widest = max(probe.textlength(line, font=mono) for line in lines)
    width = max(1100, int(widest) + 80)
    line_height = size + 12
    height = 94 + len(lines) * line_height + 30
    image = Image.new('RGB', (width, height), '#101827')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 70), fill='#1c293b')
    draw.text((30, 23), title, font=header_font, fill='#dbeaff')
    for i, line in enumerate(lines):
        draw.text((30, 92 + i * line_height), line, font=mono,
                  fill='#89dfba' if line.startswith(('PASS', 'END')) else '#e9eff8')
    image.save(destination)
    return image


def main():
    EVIDENCE.mkdir(exist_ok=True)
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'))
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - start
    log = stream.getvalue()
    (EVIDENCE / 'tests.txt').write_text(log, encoding='utf-8')
    print(log, end='')
    if not result.wasSuccessful():
        raise SystemExit('Tests failed; refusing to generate successful evidence.')

    env = ReadTheBossEnv(opponent='balanced', render_mode='rgb_array')
    initial_obs, _ = env.reset(seed=7)
    Image.fromarray(env.render()).save(EVIDENCE / 'figure1_environment.png')
    env.close()

    source = (ROOT / 'read_the_boss' / 'env.py').read_text(encoding='utf-8')
    source_lines = source.splitlines()
    first = next(i for i, s in enumerate(source_lines) if s.strip() == 'action = Action(int(action))')
    last = next(i for i in range(first, len(source_lines))
                if source_lines[i].strip().startswith('return self._observation(), reward,'))
    excerpt = [f'{i + 1:3d}  {source_lines[i][8:]}' for i in range(first, last + 1)]
    draw_text_panel(excerpt, 'read_the_boss/env.py | step() after input validation',
                    EVIDENCE / 'figure2_code.png', size=23)

    trace = json.loads((EVIDENCE / 'demo' / 'trace.json').read_text())
    smoke = json.loads((EVIDENCE / 'smoke_results.json').read_text())
    if trace['initial_observation'] != initial_obs.tolist():
        raise ValueError('Saved trace does not match the current default reset.')
    if not np.isclose(sum(s['reward'] for s in trace['steps']), trace['summary']['return']):
        raise ValueError('Trace reward total does not match recorded return.')

    rows = [
        '$ python -m unittest discover -s tests -v',
        f'PASS {result.testsRun}/{result.testsRun} tests; failures=0; errors=0',
        'Gymnasium checker + seeded combat/range/resource tests',
        '',
        '$ python -m read_the_boss demo --seed 7 --policy reactive',
        'ORDER: [player HP, boss HP, stamina, distance, intent, turns]',
        f'RESET {trace["initial_observation"]}',
    ]
    for step in trace['steps'][:3]:
        rows.append(f't={step["turn"]:02d} {step["action_name"]:<8} '
                    f'{step["observation_after"]}  r={step["reward"]:+.3f}')
    rows += [
        f'... full {len(trace["steps"])}-turn trace: evidence/demo/trace.txt',
        f'END {trace["summary"]["outcome"].upper()} | '
        f'HP={trace["summary"]["remaining_player_health"]} | '
        f'return={trace["summary"]["return"]:+.3f}',
        '',
        f'SMOKE {smoke["summary"]["episodes"]:,} heuristic episodes | '
        f'{smoke["summary"]["environment_steps"]:,} steps | '
        f'{smoke["summary"]["invalid_actions"]} masked actions submitted',
        'No trained RL policy; no held-out policy benchmark.',
    ]
    draw_text_panel(rows, 'Measured execution output | complete logs in evidence/',
                    EVIDENCE / 'figure3_execution.png', size=23)

    frame_paths = [EVIDENCE / 'demo' / name for name in trace['frames']]
    if frame_paths:
        frames = []
        for path in frame_paths:
            with Image.open(path) as image:
                frames.append(image.resize((840, 504)).convert('RGB'))
        frames[0].save(EVIDENCE / 'demo' / 'demo.gif', save_all=True,
                       append_images=frames[1:], duration=[1100] * (len(frames) - 1) + [2500],
                       loop=0, optimize=False)

    summary = {
        'kind': 'measured_milestone2_evidence',
        'tests': {'run': result.testsRun, 'failures': len(result.failures),
                  'errors': len(result.errors), 'skipped': len(result.skipped),
                  'seconds': round(elapsed, 3)},
        'versions': {'python': platform.python_version(), 'gymnasium': gymnasium.__version__,
                     'numpy': np.__version__},
        'code_figure': {'path': 'read_the_boss/env.py', 'first_line': first + 1,
                        'last_line': last + 1,
                        'sha256': hashlib.sha256(source.encode()).hexdigest()},
        'demo': trace['summary'],
        'smoke': smoke['summary'],
        'note': 'All policies are untrained heuristics. Held-out mechanics receive unit tests only.',
    }
    (EVIDENCE / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print('Generated three figures, full test log, evidence summary, and animated replay.')


if __name__ == '__main__':
    main()
