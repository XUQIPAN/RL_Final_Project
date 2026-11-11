"""Build measured Milestone 3 figures and an editable LaTeX/PDF report.

No training or evaluation is performed here. The input must be a completed,
measured final_summary.json from the experiment runner. Figure code excerpts
are taken verbatim from the implementation and linked to full-file SHA256s.
"""
from __future__ import annotations

import argparse
import ast
from collections import defaultdict
import hashlib
import io
import json
import keyword
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import tokenize

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'report' / 'milestone3'
FIGURES = REPORT / 'figures'
COLORS = {'q_learning': '#147d92', 'sarsa': '#c65c35'}


def canonical_algorithm(value):
    value = value.lower().replace('-', '_').replace(' ', '_')
    return 'q_learning' if value in ('qlearning', 'q_learning', 'q') else value


def algorithm_label(value):
    return {'q_learning': 'Q-learning', 'sarsa': 'SARSA'}[canonical_algorithm(value)]


def tex_escape(value):
    replacements = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$',
                    '#': r'\#', '_': r'\_', '{': r'\{', '}': r'\}',
                    '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}
    return ''.join(replacements.get(c, c) for c in str(value))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_summary(path):
    data = json.loads(path.read_text(encoding='utf-8'))
    if data.get('status') != 'complete':
        raise ValueError('Only a completed experiment summary may generate the report.')
    for field in ('training_runs', 'baseline_results', 'condition_aggregates',
                  'held_out_aggregates', 'learning_curves', 'config'):
        if not data.get(field):
            raise ValueError(f'Missing or empty measured field: {field}')
    if any(r['episodes'] <= 0 for r in data['baseline_results']):
        raise ValueError('Baseline results must contain measured episodes.')
    return data


def _font(size, mono=False, bold=False):
    family = 'DejaVuSansMono' if mono else 'DejaVuSans'
    if bold:
        family += '-Bold'
    for path in (Path('/usr/share/fonts/truetype/dejavu') / (family + '.ttf'),
                 Path(family + '.ttf')):
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def source_figure(path, first, last, filename, title):
    """Render an unchanged contiguous source excerpt, retaining line numbers."""
    source = path.read_text(encoding='utf-8')
    lines = source.splitlines()
    selected = lines[first - 1:last]
    indent = min((len(line) - len(line.lstrip()) for line in selected if line.strip()), default=0)
    shown = [line[indent:] for line in selected]
    mono = _font(24, mono=True)
    probe = ImageDraw.Draw(Image.new('RGB', (1, 1)))
    width = max(1180, int(max(probe.textlength(line, font=mono) for line in shown)) + 135)
    height = 96 + len(shown) * 33 + 22
    image = Image.new('RGB', (width, height), '#101827')
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 68), fill='#1c293b')
    draw.text((26, 21), title, font=_font(23, bold=True), fill='#dbeaff')
    for index, line in enumerate(shown):
        y = 91 + 33 * index
        draw.text((24, y), f'{first + index:3}', font=mono, fill='#8296af')
        draw.text((101, y), line, font=mono, fill='#e7eef7')
    # Syntax coloring overlays the same exact characters; it does not rewrite code.
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        row, column = token.start
        if not first <= row <= last or token.end[0] != row or column < indent:
            continue
        color = None
        if token.type == tokenize.NAME and keyword.iskeyword(token.string):
            color = '#c39bff'
        elif token.type == tokenize.STRING:
            color = '#a0d7a2'
        elif token.type == tokenize.NUMBER:
            color = '#ffbc79'
        elif token.type == tokenize.COMMENT:
            color = '#8c9eb5'
        if color:
            prefix = lines[row - 1][indent:column]
            x = 101 + probe.textlength(prefix, font=mono)
            draw.text((x, 91 + 33 * (row - first)), token.string, font=mono, fill=color)
    image.save(FIGURES / filename)
    return {'path': str(path.relative_to(ROOT)), 'first_line': first, 'last_line': last,
            'sha256': hashlib.sha256(source.encode()).hexdigest(), 'figure': filename}


def find_update_excerpt(path):
    """Find the value-update core after method argument validation."""
    source = path.read_text()
    tree = ast.parse(source)
    candidates = [node for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                  and node.name == 'update']
    if not candidates:
        raise ValueError(f'No update method found in {path}')
    node = candidates[0]
    target = next(child for child in node.body if isinstance(child, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == 'target' for t in child.targets))
    return target.lineno, node.end_lineno


def find_training_excerpt(path):
    source = path.read_text()
    tree = ast.parse(source)
    candidates = [node for node in ast.walk(tree) if isinstance(node, ast.While)
                  and any(isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                          and call.func.attr == 'step' for call in ast.walk(node))]
    if not candidates:
        raise ValueError(f'No interaction while-loop found in {path}')
    node = min(candidates, key=lambda n: n.end_lineno - n.lineno)
    update = next(child for child in node.body if isinstance(child, ast.Expr)
                  and isinstance(child.value, ast.Call)
                  and isinstance(child.value.func, ast.Attribute)
                  and child.value.func.attr == 'update')
    return node.lineno, update.end_lineno


def configure_plotting():
    # Select a writable cache before Matplotlib initializes its font manager.
    configured = os.environ.get('MPLCONFIGDIR')
    candidates = ([Path(configured)] if configured else []) + [ROOT / 'tmp' / 'matplotlib']
    for directory in candidates:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryFile(dir=directory):
                pass
        except OSError:
            continue
        os.environ['MPLCONFIGDIR'] = str(directory.resolve())
        break
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#233546', 'text.color': '#233546',
                         'axes.edgecolor': '#9caebb', 'grid.color': '#dce5eb',
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
    return plt


def learning_curve_figure(data):
    plt = configure_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.2), sharey=True)
    rows = data['learning_curves']
    seed_counts = set()
    curve_top = 0.0
    for ax, opponent in zip(axes, ('balanced', 'mixed')):
        for algorithm in ('q_learning', 'sarsa'):
            grouped = defaultdict(list)
            for row in rows:
                if row['training_opponent'] == opponent and canonical_algorithm(row['algorithm']) == algorithm:
                    grouped[int(row['checkpoint_episode'])].append(float(row['win_rate']))
            if not grouped:
                raise ValueError(f'Missing learning curve: {algorithm}/{opponent}')
            x = np.array(sorted(grouped))
            y = np.array([np.mean(grouped[k]) for k in x]) * 100
            sd = np.array([np.std(grouped[k], ddof=1) if len(grouped[k]) > 1 else 0 for k in x]) * 100
            curve_top = max(curve_top, float(np.max(y + sd)))
            seed_counts.update(len(grouped[k]) for k in x)
            ax.plot(x / 1000, y, label=algorithm_label(algorithm), color=COLORS[algorithm],
                    lw=2.0, marker='o', markersize=3.5)
            ax.fill_between(x / 1000, np.maximum(0, y - sd), np.minimum(100, y + sd),
                            color=COLORS[algorithm], alpha=0.13, linewidth=0)
        ax.set_title(f'Train/evaluate: {opponent}', fontsize=11, weight='bold')
        ax.set_xlabel('Training episodes (thousands)')
        ax.grid(axis='y', alpha=0.8)
        ax.legend(frameon=False, fontsize=9, loc='upper left')
    upper = max(10, min(105, 10 * math.ceil((curve_top + 5) / 10)))
    for ax in axes:
        ax.set_ylim(-0.7, upper)
        ax.set_yticks(np.arange(0, upper + 1, 10))
    axes[0].set_ylabel('Greedy checkpoint win rate (%)')
    fig.tight_layout(pad=1.1, w_pad=2)
    fig.savefig(FIGURES / 'figure3_learning_curve.png', dpi=220)
    plt.close(fig)
    return {'seed_counts': sorted(seed_counts), 'vertical_axis_percent': [-0.7, upper],
            'definition': 'mean plus/minus sample SD across training seeds; shaded bounds clipped to 0--100 percent'}


def baseline_figure(data):
    plt = configure_plotting()
    labels, means, errors, colors = [], [], [], []
    for policy in ('random', 'aggressive', 'reactive'):
        row = next(r for r in data['baseline_results'] if r['opponent'] == 'mixed' and r['policy'] == policy)
        labels.append(policy.capitalize())
        means.append(100 * float(row['win_rate']))
        errors.append(0)
        colors.append('#97a8b5')
    for algorithm in ('q_learning', 'sarsa'):
        for opponent in ('balanced', 'mixed'):
            row = next(r for r in data['condition_aggregates']
                       if r['opponent'] == 'mixed' and r['training_opponent'] == opponent
                       and canonical_algorithm(r['algorithm']) == algorithm)
            labels.append(('Q-learning' if algorithm == 'q_learning' else 'SARSA') + '\n' + opponent + ' train')
            means.append(100 * float(row['win_rate_mean']))
            errors.append(100 * float(row['win_rate_sd']))
            colors.append(COLORS[algorithm] if opponent == 'mixed' else
                          ('#8ab9c1' if algorithm == 'q_learning' else '#e0a58e'))
    fig, ax = plt.subplots(figsize=(10.4, 3.3))
    bars = ax.bar(range(len(labels)), means, color=colors, width=0.68, zorder=3)
    # Fixed heuristics have one measured evaluation, not three trained replicas.
    ax.errorbar(range(3, len(labels)), means[3:], yerr=errors[3:], fmt='none',
                capsize=3, elinewidth=1.1, ecolor='#233546', zorder=4)
    for bar, mean, error in zip(bars, means, errors):
        ax.text(bar.get_x() + bar.get_width() / 2, min(101.5, mean + error + 2),
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 109)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_ylabel('Win rate against mixed boss (%)')
    ax.grid(axis='y', alpha=0.8, zorder=0)
    fig.tight_layout(pad=1.1)
    fig.savefig(FIGURES / 'figure4_baseline_comparison.png', dpi=220)
    plt.close(fig)


def compile_latex(tex_path):
    """Compile portably; recover a source-only Debian TeX setup when necessary."""
    if not shutil.which('pdflatex'):
        raise RuntimeError('pdfLaTeX is unavailable; install TeX Live or use Overleaf.')
    with tempfile.TemporaryDirectory(prefix='.latex-build-', dir=REPORT) as temporary:
        build = Path(temporary)
        env = dict(os.environ)
        extra = []
        def run(command):
            result = subprocess.run(command, cwd=REPORT, env=env, text=True,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError('LaTeX command failed:\n' + result.stdout[-6000:])
            return result.stdout
        configured = all(subprocess.run(['kpsewhich', name], text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.DEVNULL).returncode == 0 for name in ('article.cls', 'pdflatex.fmt'))
        if not configured and Path('/usr/share/texlive/texmf-dist').exists():
            env['TEXMF'] = '{/usr/share/texlive/texmf-dist,/usr/share/texmf}'
            env['TEXMFVAR'] = str(build / 'texcache')
            run(['pdftex', '-ini', '-etex', '-jobname=pdflatex', '-progname=pdflatex',
                 '-interaction=nonstopmode', '-halt-on-error', '-output-directory=' + str(build), 'pdflatex.ini'])
            extra.append('-fmt=' + str(build / 'pdflatex.fmt'))
            map_result = subprocess.run(['kpsewhich', 'lm.map'], env=env, text=True,
                                        stdout=subprocess.PIPE, check=True)
            shutil.copyfile(map_result.stdout.strip(), build / 'pdftex.map')
            env['TEXFONTMAPS'] = str(build) + ':'
        command = ['pdflatex', *extra, '-interaction=nonstopmode', '-halt-on-error',
                   '-output-directory=' + str(build), tex_path.name]
        run(command)
        log = run(command)
        pdf_path = build / tex_path.with_suffix('.pdf').name
        pdfinfo = '/usr/bin/pdfinfo' if Path('/usr/bin/pdfinfo').exists() else shutil.which('pdfinfo')
        if pdfinfo:
            info = subprocess.run([pdfinfo, str(pdf_path)], capture_output=True, text=True, check=True).stdout
            match = re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)
        else:
            match = re.search(r'Output written on .*?\((\d+) pages?[,)]', log, re.DOTALL)
        if match is None:
            raise RuntimeError('Cannot verify the page limit. Install Poppler pdfinfo, then rebuild.')
        pages = int(match.group(1))
        if pages > 5:
            raise ValueError(f'Report exceeds the five-page limit: {pages} pages.')
        destination = tex_path.with_suffix('.pdf')
        shutil.copyfile(pdf_path, destination)
        warnings = [line for line in log.splitlines() if 'Overfull' in line or 'Underfull' in line]
        return {'pages': pages, 'warnings': warnings, 'pdf': str(destination.relative_to(ROOT))}


def generated_values(data, author, source_metadata):
    config = data['config']
    train = config['training']
    checkpoint = config['checkpoint_evaluation']
    final = config['final_evaluation']
    runs = data['training_runs']
    if len(runs) != len(train['algorithms']) * len(train['opponents']) * len(train['seeds']):
        raise ValueError('The complete report requires every configured training run.')
    run_summaries = [r['training_summary']['summary'] for r in runs]
    steps = [s['environment_steps'] for s in run_summaries]
    mixed_rows = [r for r in data['condition_aggregates'] if r['opponent'] == 'mixed']
    best = max(mixed_rows, key=lambda r: r['win_rate_mean'])
    reactive = next(r for r in data['baseline_results']
                    if r['opponent'] == 'mixed' and r['policy'] == 'reactive')
    best_name = algorithm_label(best['algorithm']) + ', ' + best['training_opponent'] + ' training'
    pool = best['pooled']
    reactive_gap = 100 * (best['win_rate_mean'] - reactive['win_rate'])
    if reactive_gap < 0:
        baseline_conclusion = (f'The strongest learned condition trailed the reactive heuristic by '
                               f'{abs(reactive_gap):.1f} percentage points; this training budget '
                               'has not matched that hand-coded policy.')
    elif reactive_gap > 0:
        baseline_conclusion = (f'The strongest learned condition exceeded the reactive heuristic by '
                               f'{reactive_gap:.1f} percentage points in this sample; the small '
                               'seed count limits the strength of this comparison.')
    else:
        baseline_conclusion = 'The strongest learned condition tied the reactive heuristic in this sample.'
    start_end = []
    for algorithm in train['algorithms']:
        for opponent in train['opponents']:
            rows = [r for r in data['learning_curves']
                    if canonical_algorithm(r['algorithm']) == canonical_algorithm(algorithm)
                    and r['training_opponent'] == opponent]
            start = [r['win_rate'] for r in rows if r['checkpoint_episode'] == 0]
            end = [r['win_rate'] for r in rows if r['checkpoint_episode'] == train['episodes']]
            if len(start) != len(train['seeds']) or len(end) != len(train['seeds']):
                raise ValueError('Initial/final checkpoint coverage is incomplete.')
            start_end.append((algorithm_label(algorithm), opponent,
                              100 * float(np.mean(start)), 100 * float(np.mean(end))))
    endpoints = '; '.join(f'{a}/{o}: {x:.1f} to {y:.1f}\\%' for a, o, x, y in start_end)
    if all(end > start for _, _, start, end in start_end):
        trend = 'Each condition improved from its untrained checkpoint.'
    else:
        trend = 'Improvement was not uniform across conditions.'
    rows = []
    for algorithm in train['algorithms']:
        for opponent in train['opponents']:
            row = next(r for r in data['held_out_aggregates']
                       if canonical_algorithm(r['algorithm']) == canonical_algorithm(algorithm)
                       and r['training_opponent'] == opponent)
            rows.append(f'{algorithm_label(algorithm)} / {opponent} & '
                        f'{100 * row["mean_win_rate_mean"]:.1f} $\\pm$ {100 * row["mean_win_rate_sd"]:.1f} & '
                        f'{100 * row["worst_win_rate_mean"]:.1f} $\\pm$ {100 * row["worst_win_rate_sd"]:.1f} \\\\')
    deltas = []
    for algorithm in train['algorithms']:
        values = {r['training_opponent']: r['mean_win_rate_mean'] for r in data['held_out_aggregates']
                  if canonical_algorithm(r['algorithm']) == canonical_algorithm(algorithm)}
        delta = 100 * (values['mixed'] - values['balanced'])
        deltas.append(f'{algorithm_label(algorithm)} {delta:+.1f} percentage points')
    initial_versions = data.get('versions', {})
    values = {
        'StudentName': tex_escape(author),
        'TrainingEpisodes': f'{train["episodes"]:,}',
        'RunCount': str(len(runs)),
        'SeedCount': str(len(train['seeds'])),
        'SeedList': ', '.join(map(str, train['seeds'])),
        'DemoSeed': str(config['demo']['seed']),
        'AlphaValue': str(train['alpha']),
        'GammaValue': str(train['gamma']),
        'EpsilonStart': str(train['epsilon_start']),
        'EpsilonEnd': str(train['epsilon_end']),
        'DecayPercent': f'{100 * train["epsilon_decay_fraction"]:.0f}',
        'DecayEpisodes': f'{math.ceil(train["episodes"] * train["epsilon_decay_fraction"]):,}',
        'CheckpointEvery': f'{train["checkpoint_every"]:,}',
        'CheckpointEpisodes': f'{checkpoint["episodes"]:,}',
        'FinalEpisodes': f'{final["episodes"]:,}',
        'TrainingStepsTotal': f'{sum(steps):,}',
        'TrainingStepsRange': f'{min(steps):,}--{max(steps):,}',
        'InvalidTrainingActions': str(sum(s['invalid_actions'] for s in run_summaries)),
        'CurveTrend': trend,
        'CurveEndpoints': endpoints,
        'BestMixedName': tex_escape(best_name),
        'BestMixedWin': f'{100 * best["win_rate_mean"]:.1f}',
        'BestMixedSD': f'{100 * best["win_rate_sd"]:.1f}',
        'ReactiveMixedWin': f'{100 * reactive["win_rate"]:.1f}',
        'BestMixedReturn': f'{best["mean_return_mean"]:.2f}',
        'BestMixedTurns': f'{pool["mean_turns_among_wins"]:.1f}' if pool['mean_turns_among_wins'] is not None else '--',
        'BaselineConclusion': baseline_conclusion,
        'HeldOutTableRows': '\n'.join(rows),
        'HeldOutDeltas': '; '.join(deltas),
        'AgentCodeLines': f'{source_metadata[0]["first_line"]}--{source_metadata[0]["last_line"]}',
        'AgentCodeHash': source_metadata[0]['sha256'][:12],
        'TrainCodeLines': f'{source_metadata[1]["first_line"]}--{source_metadata[1]["last_line"]}',
        'TrainCodeHash': source_metadata[1]['sha256'][:12],
    }
    text = '% Generated only from measured experiments/milestone3/final_summary.json.\n'
    text += '\n'.join('\\newcommand{\\' + name + '}{' + value + '}' for name, value in values.items())
    (REPORT / 'results_generated.tex').write_text(text + '\n', encoding='utf-8')
    return {'learning_endpoints': start_end, 'best_mixed_condition': best_name,
            'training_steps': steps, 'initial_versions': initial_versions}


TEX_TEMPLATE = r'''% Editable report compiled from measured experiment data.
% Rebuild figures/data: python tools/build_milestone3_report.py --compile
% Compile manually from this folder: pdflatex milestone3_report.tex (twice).
% Author and measured numbers live in results_generated.tex.
\documentclass[10pt,letterpaper]{article}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\renewcommand{\familydefault}{\sfdefault}
\usepackage[left=40pt,right=40pt,top=32pt,bottom=36pt,footskip=20pt]{geometry}
\usepackage{graphicx,booktabs,tabularx,xcolor,enumitem,titlesec,fancyhdr,microtype,amsmath}
\usepackage[hidelinks]{hyperref}
\input{results_generated.tex}
\definecolor{ink}{HTML}{15384D}
\definecolor{muted}{HTML}{536A7C}
\setlength{\parindent}{0pt}
\setlength{\parskip}{4pt}
\setlist[itemize]{leftmargin=11pt,label=\textendash,itemsep=3pt,topsep=1pt,parsep=0pt}
\titleformat{\section}{\fontsize{12}{15}\selectfont\bfseries\color{ink}}{}{0pt}{}
\titlespacing*{\section}{0pt}{8pt}{5pt}
\pagestyle{fancy}\fancyhf{}
\fancyfoot[L]{\fontsize{7}{9}\selectfont\color{muted}READ THE BOSS \quad | \quad MILESTONE 3 \quad | \quad IMPLEMENTATION AND INITIAL RESULTS}
\fancyfoot[R]{\fontsize{7}{9}\selectfont\color{muted}\thepage}
\renewcommand{\headrulewidth}{0pt}\renewcommand{\footrulewidth}{0.3pt}
\newcommand{\reportfigure}[3]{\par\begingroup\centering
  \includegraphics[width=\linewidth,height=#2,keepaspectratio]{figures/#1}\par
  \endgroup\vspace{2pt}{\fontsize{8}{10.2}\selectfont\color{muted}#3\par}}
\hypersetup{pdftitle={Read the Boss - Milestone 3},pdfauthor={\StudentName}}
\begin{document}
\fontsize{9.5}{12.4}\selectfont
{\fontsize{8}{10}\selectfont\bfseries\color{muted}MILESTONE 3 / REINFORCEMENT LEARNING IMPLEMENTATION\par}
\vspace{2pt}{\fontsize{22}{26}\selectfont\bfseries\color{ink}Read the Boss\par}
{\color{muted}\StudentName\enspace | \enspace Stamina-constrained tactical combat\par}

\section*{1. Project Progress}
\begin{itemize}
\item \textbf{Completed since Milestone 2:} Implemented masked tabular Q-learning/SARSA, seeded training, greedy checkpoint evaluation, portable saved Q-tables, episode logs, and baseline/held-out evaluation. This report presents \RunCount{} measured runs with \TrainingEpisodes{} episodes each.
\item \textbf{Environment changes:} Combat, observations, rewards, action masks, boss controllers, and the observed 40-turn deadline are unchanged. Learning and experiment infrastructure were added around the existing environment.
\item \textbf{Remaining:} The final equal-step-budget, five-seed study, reward ablation, and deeper failure analysis remain. These are initial results, not the full experiment budget proposed earlier.
\end{itemize}

\section*{2. RL Algorithm Implementation}
\textbf{Algorithm and reason.} Q-learning and SARSA fit the small discrete problem and permit inspection of learned values. A zero-initialized table indexes $s=(h_p,h_b,e,d,i,\tau)$: player/boss health, stamina, distance, visible boss intent, and turns left. Time preserves the finite-horizon state; actions are approach, retreat, strike, guard, and rest.

\textbf{Action selection.} An $\epsilon$-greedy policy explores uniformly among legal actions and otherwise chooses a maximum-valued legal action; ties are randomized. Both action selection and Q-learning's target exclude masked actions. Greedy evaluation uses $\epsilon=0$ with a separate RNG and never updates the table.

\textbf{Learning.} With legal next actions $\mathcal A(s')$, both methods apply
\[
 Q(s,a)\leftarrow Q(s,a)+\alpha\,[y-Q(s,a)],\qquad
 y=\begin{cases}
 r, & \text{terminal},\\
 r+\gamma\max_{b\in\mathcal A(s')}Q(s',b), & \text{Q-learning},\\
 r+\gamma Q(s',a'), & \text{SARSA},
 \end{cases}
\]
where SARSA's $a'$ is the behavior action actually carried into the next step. Victory, death, and intrinsic timeout all terminate the task and bootstrap zero; $\gamma=\GammaValue$ is used for this finite horizon.

\reportfigure{figure1_agent_update.png}{239pt}{\textbf{Figure 1.} Actual update core in \texttt{read\_the\_boss/agents.py}, lines \AgentCodeLines{} (SHA256 prefix \texttt{\AgentCodeHash}). The terminal branch leaves the target equal to reward; nonterminal branches use the legal maximum or SARSA's selected action. The final lines apply the TD error and count visits. Full hashes and line ranges are in \texttt{figures\_manifest.json}.}

\newpage
\section*{3. Training Process}
\textbf{Procedure.} Each episode resets with a reproducible seed, computes $\epsilon$, then repeats transitions and TD updates until termination. SARSA carries its selected next action forward. Independent environment/action/evaluation RNGs and a mutation check ensure evaluation preserves training.

\begin{tabularx}{\linewidth}{@{}lXlX@{}}
\toprule
\textbf{Setting} & \textbf{Value} & \textbf{Setting} & \textbf{Value} \\
\midrule
Learning rate & $\alpha=\AlphaValue$ & Discount & $\gamma=\GammaValue$ \\
Training & \TrainingEpisodes{} episodes/run & Seeds & \SeedList{} (\SeedCount{} runs/condition) \\
Exploration & \EpsilonStart{} to \EpsilonEnd{}, linear & Decay span & first \DecayPercent\% (\DecayEpisodes{} episodes) \\
Conditions & Q-learning/SARSA $\times$ balanced/mixed & Checkpoints & 0, then every \CheckpointEvery{} episodes \\
Checkpoint tests & \CheckpointEpisodes{} greedy episodes & Final tests & \FinalEpisodes{} episodes/boss/model \\
\bottomrule
\end{tabularx}

\textbf{Interaction and outputs.} Mixed training resamples balanced/aggressive/distance-aware rules each turn. Reward is $-0.1-0.02c$ per step ($c$: stamina spent), plus $100+10h_p/6$ for victory or $-100$ for death/timeout. Runs save \texttt{episodes.csv}, metadata, and final \texttt{agent.npz}; \InvalidTrainingActions{} masked actions were submitted.

\reportfigure{figure2_training_loop.png}{177pt}{\textbf{Figure 2.} Actual \texttt{read\_the\_boss/training.py}, lines \TrainCodeLines{} (SHA256 prefix \texttt{\TrainCodeHash}): transition, next SARSA action, and TD update. Reset, action continuation, counters, and checkpoint logging appear in the full source.}

\reportfigure{figure3_learning_curve.png}{168pt}{\textbf{Figure 3.} Greedy checkpoint win rate, mean $\pm$ shaded sample SD over \SeedCount{} training seeds (bands clipped to 0--100\%). Each checkpoint uses \CheckpointEpisodes{} fixed, separate evaluation episodes against its training controller. Opponents differ between panels; Figure 4 gives a matched-opponent comparison.}

\textbf{Budget and selection.} Episodes are matched; steps vary: \TrainingStepsRange{} per run (\TrainingStepsTotal{} total). Only final checkpoints enter final tests. Neither curve peaks nor held-out outcomes select a model; held-out rules and final seeds never enter training or parameter selection.

\newpage
\section*{4. Initial Results \& Baseline}
\textbf{Improvement.} \CurveTrend{} Mean greedy win rates from episode 0 to the final checkpoint were: \CurveEndpoints{}. These checkpoint rates use the training controller; final comparison below tests every policy on the same mixed boss.

\reportfigure{figure4_baseline_comparison.png}{187pt}{\textbf{Figure 4.} Final greedy policies and the random/aggressive/reactive heuristics against the mixed boss, with \FinalEpisodes{} episodes per evaluation. Trained bars are means with sample-SD error bars across \SeedCount{} training seeds; fixed heuristics have one evaluation and no training-seed error bar. Shared evaluation seed lists support comparison; they are separate from checkpoint seeds.}

\textbf{Baseline comparison.} Among learned conditions, the highest mixed-boss win rate was \BestMixedWin{} $\pm$ \BestMixedSD\% (\BestMixedName), versus \ReactiveMixedWin\% for the reactive heuristic. \BaselineConclusion{} For that learned condition, mean return was \BestMixedReturn{} and mean duration among wins was \BestMixedTurns{} turns. Win rate is primary; return, death/timeout, health, duration, stamina use, and zero-stamina frequency are also logged.

\textbf{Unseen behavior tests.} Final checkpoints were tested on the sweep specialist, relentless pursuer, and health-triggered enraged boss. Values below are percentages, mean $\pm$ sample SD over training seeds. ``Worst'' is computed across the three held-out bosses separately for each seed, then averaged.

\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
\textbf{Training condition} & \textbf{Held-out mean win rate} & \textbf{Worst boss per seed} \\
\midrule
\HeldOutTableRows
\bottomrule
\end{tabularx}

Mixed minus balanced training changed the held-out average by \HeldOutDeltas. These are descriptive initial differences; three training seeds and finite evaluation samples do not establish statistical superiority.

\section*{5. Next Steps}
\begin{itemize}
\item \textbf{Limitations:} Equal episode counts yield different interaction budgets; small seed counts and fixed checkpoint samples limit conclusions. Sparse wins and time-indexed state coverage may impede learning; longer training alone is not guaranteed to resolve this. Scripted opponents and discrete distances simplify combat.
\item \textbf{Plan:} Run the proposed equal-step, five-seed comparison; add the health/stamina reward ablation and inspect failures and state coverage. If tuning responds to these held-out results, reserve fresh boss rules for final testing; compare resource metrics and document subsequent changes.
\end{itemize}

{\fontsize{8}{10.2}\selectfont\color{muted}\textbf{Reproduce.} Run the commands in \texttt{README.md} using \texttt{configs/milestone3.json}. Raw results and saved agents are under \texttt{experiments/milestone3/}. Rebuild with \texttt{python tools/build\_milestone3\_report.py --compile}; the editable \texttt{.tex} uses included figures and \texttt{results\_generated.tex}. \textbf{Optional animated replay:} \texttt{experiments/milestone3/demo/demo.gif} and its matching trace show one fixed-seed (\DemoSeed) episode, not an estimate of success rate.\par}
\end{document}
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, default=ROOT / 'experiments/milestone3/final_summary.json')
    parser.add_argument('--author', default='Qipan Xu')
    parser.add_argument('--compile', action='store_true', help='Compile the emitted LaTeX to PDF (requires pdfLaTeX).')
    args = parser.parse_args()
    data = load_summary(args.summary)
    FIGURES.mkdir(parents=True, exist_ok=True)
    agents = ROOT / 'read_the_boss/agents.py'
    training = ROOT / 'read_the_boss/training.py'
    first, last = find_update_excerpt(agents)
    source_metadata = [source_figure(agents, first, last, 'figure1_agent_update.png',
                                    'read_the_boss/agents.py | TD target and update')]
    first, last = find_training_excerpt(training)
    source_metadata.append(source_figure(training, first, last, 'figure2_training_loop.png',
                                        'read_the_boss/training.py | environment interaction'))
    curve_metadata = learning_curve_figure(data)
    baseline_figure(data)
    analysis = generated_values(data, args.author, source_metadata)
    tex_path = REPORT / 'milestone3_report.tex'
    tex_path.write_text(TEX_TEMPLATE, encoding='utf-8')
    manifest = {'summary_path': str(args.summary.relative_to(ROOT)) if args.summary.is_relative_to(ROOT) else str(args.summary),
                'summary_sha256': digest(args.summary), 'source_excerpts': source_metadata,
                'learning_curve': curve_metadata, 'analysis': analysis,
                'figures': {p.name: digest(p) for p in sorted(FIGURES.glob('*.png'))}}
    if args.compile:
        manifest['compilation'] = compile_latex(tex_path)
    (REPORT / 'figures_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'latex': str(tex_path), 'figures': 4,
                      'compilation': manifest.get('compilation')}, indent=2))


if __name__ == '__main__':
    main()
