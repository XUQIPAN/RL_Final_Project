# Read the Boss — Milestone 3

This repository contains the complete environment, Q-learning and SARSA
implementations, heuristic baselines, tests, configuration, and utility scripts.
Train your own agents and generate your own results using the commands below.
Milestone 3 models, experiment logs, figures, and PDFs are generated locally when
you run the corresponding commands. Earlier milestone evidence and reports are
preserved in `evidence/` and `report/`.

## 1. Setup

Use Python 3.12 and run all commands from the `RL_Final_Project/` repository
root, which contains this README.

```bash
python -m venv .venv
```

Activate the environment on macOS/Linux:

```bash
source .venv/bin/activate
```

Or on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install all runtime, training, evaluation, and report-generation dependencies:

```bash
python -m pip install -r requirements.txt
```

`requirements.txt` specifies supported package versions for both milestones.
To reproduce a run, match the package versions recorded in its experiment
manifest and reuse the same seeds. No GPU or API key is needed.

Install Git and ensure `git --version` works before using the experiment runner:
its provenance code invokes Git. A standalone extracted folder does not need to
be a Git repository. Single-agent training and evaluation do not require Git.

## 2. Check the installation

Run the environment and learning tests:

```bash
python -m unittest discover -s tests -v
```

There are 48 tests covering environment behavior, Q-learning/SARSA update targets,
action masks, terminal handling, saved-model round trips, and reproducibility.

Optional small pipeline check:

```bash
python -m read_the_boss.experiments --config configs/milestone3.json --quick --output experiments/quick_check
```

This uses four 40-episode training runs and five evaluation episodes per condition.
It excludes held-out bosses and is not a performance benchmark or a report dataset.
All experiment commands protect existing outputs: use a new output directory
for another run instead of reusing a populated one.

## 3. Train an individual agent

Q-learning against the Mixed boss:

```bash
python -m read_the_boss.training --algorithm q_learning --opponent mixed --episodes 20000 --seed 0 --alpha 0.15 --gamma 1.0 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-fraction 0.8 --checkpoint-every 2000 --eval-episodes 100 --output experiments/local_qlearning
```

SARSA against the same boss:

```bash
python -m read_the_boss.training --algorithm sarsa --opponent mixed --episodes 20000 --seed 0 --alpha 0.15 --gamma 1.0 --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-decay-fraction 0.8 --checkpoint-every 2000 --eval-episodes 100 --output experiments/local_sarsa
```

Use `--opponent balanced` for single-controller training. Change `--episodes` and
`--seed` to choose your budget and independent run. Exploration decreases linearly
over the first 80% of episodes and then stays at 0.05. Evaluation uses epsilon=0
and independent RNGs. Both methods use only legal actions, and terminal states
(including the 40-turn deadline) bootstrap zero.

Each training directory contains:

| File | Contents |
|---|---|
| `agent.npz` | Final learned Q-table, visit counts, and model metadata |
| `episodes.csv` | Per-episode returns, wins/losses, duration, health, stamina, seeds, and epsilon |
| `metadata.json` | Configuration, total environment steps, update count, and summary |
| `learning_curve.json` | Greedy checkpoint measurements when `--eval-episodes` is positive |
| `learning_curve_episodes.csv` | Individual checkpoint evaluation episodes |

Training starts from a zero-initialized table. Checkpoint loading supports
inference/evaluation; resuming an interrupted training run is not implemented.

## 4. Evaluate a model you have trained

After the Q-learning command above has created its model:

```bash
python -m read_the_boss.evaluate --model experiments/local_qlearning/agent.npz --opponent mixed --episodes 500 --output experiments/local_qlearning_eval --demo
```

For the SARSA model, use `experiments/local_sarsa/agent.npz` and a different output
directory. The command writes per-episode `episodes.csv` and `summary.json`.
`--demo` additionally creates a `demo/` folder with an animated GIF and action trace.
The replay uses seed 7 by default; it is one episode, not a success-rate estimate.

Compare a heuristic using the same default environment/policy seed lists:

```bash
python -m read_the_boss.evaluate --policy random --opponent mixed --episodes 500 --output experiments/local_random_eval
python -m read_the_boss.evaluate --policy aggressive --opponent mixed --episodes 500 --output experiments/local_aggressive_eval
python -m read_the_boss.evaluate --policy reactive --opponent mixed --episodes 500 --output experiments/local_reactive_eval
```

Other evaluation opponents: `balanced`, `aggressive`, `distance_aware`,
`sweep_specialist`, `relentless_pursuer`, and `enraged`. The final three are the
held-out behaviors in the supplied design. Use training/validation results for
tuning; if you tune using held-out outcomes, reserve new conditions for a final test.

## 5. Run the complete experiment matrix

This is an alternative to the individual-run commands: the runner trains every
model itself, then evaluates all final models and all three heuristics.

```bash
python -m read_the_boss.experiments --config configs/milestone3.json --output experiments/milestone3
```

The default configuration is:

| Parameter | Value |
|---|---|
| Algorithms | Q-learning and SARSA |
| Training opponents | Balanced and Mixed |
| Training seeds | 0, 1, 2 |
| Budget | 20,000 episodes per run; 12 runs total |
| Learning rate / discount | alpha=0.15, gamma=1.0 |
| Exploration | epsilon 1.0 to 0.05 over the first 80% of episodes |
| Checkpoint evaluation | 100 episodes at episode 0 and every 2,000 episodes |
| Final evaluation | 500 episodes per trained model per boss |
| Baselines | Random, Aggressive, Reactive; 500 episodes each per boss |

The complete run creates:

- `experiments/milestone3/final_summary.json`: aggregate results, including
  variation across training seeds and held-out performance.
- `runs/`: final models, training logs, and raw checkpoint/final evaluations.
- `baseline_results.json` and `baseline_episodes.csv`: heuristic measurements.
- `learning_curves.json`: checkpoint results across all runs.
- `manifest.json` and `resolved_config.json`: provenance, versions, seeds, hashes,
  and the exact resolved experiment settings.
- `demo/`: the predetermined model/seed replay and trace.

Edit `configs/milestone3.json` to change budgets or parameters. For the supplied
report builder, retain both algorithms, both training regimes, at least two
training seeds, and all seven final opponents. If you remove seed 0, update
`demo.training_seed` to a retained seed or set `demo.enabled` to false.
Use a new output directory for each configuration so results remain separate.

Budgets here match episodes, not environment steps; actual step totals are logged.
Only the final model is evaluated in the final comparison. Training randomness,
checkpoint seeds, and final evaluation seeds are separate. Heuristics are evaluated
once per boss; uncertainty for learned methods is sample SD across training seeds.

## 6. Optional: generate figures and a report from your results

After the full experiment matrix finishes:

```bash
python tools/build_milestone3_report.py --summary experiments/milestone3/final_summary.json --author "Your Name"
```

This generates the four figures, result tables, and LaTeX source under
`report/milestone3/`. Plotting uses `tmp/matplotlib/` for its cache unless
`MPLCONFIGDIR` points to a writable directory.

Add `--compile` if a TeX distribution providing `pdflatex` is installed:

```bash
python tools/build_milestone3_report.py --summary experiments/milestone3/final_summary.json --author "Your Name" --compile
```

Review the generated report's wording and interpretation against your own results
before submitting. To use Overleaf, upload the entire generated
`report/milestone3/` folder, including `results_generated.tex` and `figures/`.
The report builder replaces its generated LaTeX when rerun; edit `TEX_TEMPLATE`
in the builder if you want wording changes to persist across regeneration.

Once the full results and PDF exist, package your submission:

```bash
python tools/package_milestone3.py --last-name Your Name --first-name Your Name
```

The archive is written to `dist/Xu_Qipan_milestone3.zip`.

## 7. Source layout

| Path | Purpose |
|---|---|
| `read_the_boss/env.py` | Gymnasium arena environment |
| `read_the_boss/controllers.py` | Scripted boss strategies |
| `read_the_boss/policies.py` | Random, Aggressive, and Reactive player baselines |
| `read_the_boss/agents.py` | Tabular Q-learning/SARSA, action masks, save/load |
| `read_the_boss/training.py` | Training loop and single-run CLI |
| `read_the_boss/evaluation.py` | Evaluation metrics and replay recording |
| `read_the_boss/evaluate.py` | Saved-agent and heuristic evaluation CLI |
| `read_the_boss/experiments.py` | Reproducible multi-run experiment driver |
| `read_the_boss/render.py` | Arena visualization |
| `tests/` | Environment and RL tests |
| `configs/milestone3.json` | Complete experiment design |
| `tools/build_milestone3_report.py` | Generate figures and LaTeX/PDF from measurements |


## 7. Quick demo

```bash
python run_demo.py
python -m read_the_boss play --opponent mixed --seed 7
```

## 8. Command-line help
Use `python -m read_the_boss.training --help`,
`python -m read_the_boss.evaluate --help`, or
`python -m read_the_boss.experiments --help` for command options.
