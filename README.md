# Read the Boss - Milestone 2

A custom, turn-based 1v1 reinforcement-learning environment. The fighter balances
attacks, defense, spacing, and limited stamina against a scripted stochastic boss.
This submission demonstrates a functioning environment, **not a trained RL agent**.

## Submission contents

- `report/milestone2_report.pdf`: three-page report with the three required figures.
- `read_the_boss/`: environment, controllers, renderer, heuristic policies, and CLI.
- `tests/test_environment.py`: 33 behavioral, integration, and stochastic tests.
- `evidence/`: original execution logs, JSON results, figures, and a seeded demo.
- `tools/`: reproducible evidence/report generation and ZIP packaging scripts.
- `requirements.txt`: runtime dependencies; `requirements-report.txt`: report tools.
- `requirements-tested.txt`: exact package versions used to generate the evidence.

The ZIP contains a single `milestone2/` directory. Extract it and run all commands
below **inside that directory**. No GPU, external assets, account, or API key is needed.

## Setup

Use Python 3.10 or newer; the recorded run used Python 3.12.14.

```bash
python -m venv .venv
```

Activate on macOS/Linux:

```bash
source .venv/bin/activate
```

Activate on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Install runtime dependencies:

```bash
python -m pip install -r requirements.txt
```

For the exact recorded package versions, use Python 3.12 and install
`requirements-tested.txt` instead. Reproducibility is checked within a fixed
package version; changing NumPy/Gymnasium versions may change sampled trajectories.

## Run the environment

Run a seeded reactive-heuristic demonstration and save its complete trace and PNGs:

```bash
python -m read_the_boss demo --opponent balanced --policy reactive --seed 7 --output evidence/demo
```

The recorded seed-7 run wins in 10 turns with 1 HP remaining and total return
`100.4666666667`. This is one demonstration episode, not a success-rate estimate.
Use `--no-images` for a text/JSON-only run. Frame images are read-only arena
visualizations; the actual human controls are in the terminal:

```bash
python -m read_the_boss play --opponent mixed --seed 7
```

Enter an action number/name from the displayed legal choices. Enter `q` to quit.
Quitting does not score the unfinished episode as a timeout.

## Run tests and the environment smoke check

```bash
python -m unittest discover -s tests -v
python -m read_the_boss smoke --episodes 100 --seed 1000 --output evidence/smoke_results.json
```

The recorded run passed **33/33 tests**, with no failures, errors, or skips.
Tests include Gymnasium's checker (with render modes), reset/reseeding, observation
bounds, action legality, stamina, attack ranges, lethal-hit priority, reward terms,
death/timeout, and immutable observations. They also run 280 complete random legal
episodes across all seven controllers and 60,000 seeded intent draws for distribution checks.

The smoke run uses three untrained policies and four training modes, with 100
episodes per combination (seeds 1000-1099). Its 1,200 episodes completed in 10,834
environment steps, with no masked actions submitted by the policies.

| Boss controller | Random win rate | Aggressive heuristic | Reactive heuristic |
|---|---:|---:|---:|
| Balanced | 0% | 22% | 22% |
| Aggressive | 0% | 8% | 13% |
| Distance-aware | 0% | 20% | 56% |
| Mixed | 0% | 20% | 36% |

These are **small environment smoke checks, not final RL benchmarks**. All
non-winning smoke episodes ended in death; timeout behavior is separately tested
with controlled states. Low heuristic success on some settings motivates further
baseline/difficulty calibration. Held-out controller mechanics receive unit tests,
but no held-out policy benchmark or tuning was performed.

## Environment API

```python
import gymnasium as gym
import read_the_boss  # registers ReadTheBoss-v0

env = gym.make("ReadTheBoss-v0", opponent="balanced", render_mode="ansi")
obs, info = env.reset(seed=7)
action = env.action_space.sample(mask=info["action_mask"])
obs, reward, terminated, truncated, info = env.step(action)
print(env.render())
env.close()
```

- Observation order: `[player_hp, boss_hp, stamina, distance, intent, turns_remaining]`.
- Default reset: `[6, 8, 4, 1, 3, 40]`.
- Space: `MultiDiscrete([7, 9, 5, 3, 4, 41])`, integer NumPy array. Zero health and
  zero turns are reserved for terminal observations.
- Distances: `0=close`, `1=medium`, `2=far`.
- Intents: `0=slash`, `1=sweep`, `2=lunge`, `3=recovery`.
- Action space: `Discrete(5)`; legal mask is `info["action_mask"]` (`int8`).
- `reset()` returns `(observation, info)`; `step()` returns
  `(observation, reward, terminated, truncated, info)`.
- `render_mode="ansi"` returns text; `render_mode="rgb_array"` returns a
  deterministic `720 x 1200 x 3` uint8 image. Rendering never advances the RNG.
- `info` includes outcome, events, mask, invalid-action flag, damage, stamina spent,
  and separately auditable reward terms. It does not reveal the mixed-controller draw.

The 40-turn deadline is an **intrinsic task failure**, and remaining time is part
of the observation. Victory, death, and timeout therefore all set `terminated=True`
and `truncated=False`. Registration deliberately does not add an external
`TimeLimit` wrapper. Learning code must bootstrap zero at these terminal states.

## Combat rules

| ID | Action | Effect | Stamina |
|---:|---|---|---:|
| 0 | Approach | Move one band closer | 0 |
| 1 | Retreat | Move one band farther away | -1 |
| 2 | Strike | Deal 2 damage when close | -2 |
| 3 | Guard | Block up to 2 damage this turn | -1 |
| 4 | Rest | Recover stamina, capped at 4; no protection | +2 |

The fighter acts first. A lethal strike prevents retaliation. Otherwise, the
announced boss intent resolves at the updated distance: slash deals 2 if close;
sweep deals 3 if close/medium; lunge advances one band and deals 1 if then close;
recovery does nothing. The clock decreases, terminal outcomes are checked, and
only a continuing episode samples the next visible intent. Victory/death precede
the deadline on the final turn.

Unaffordable actions, strikes out of range, and boundary-crossing movement are
masked. The supplied policies and human interface enforce masks. A direct API
call with a masked but in-range action wastes a turn: no player effect or stamina
charge, normal boss retaliation, and the normal time cost. This explicitly defined
fallback keeps generic Gymnasium callers valid. Non-integer/out-of-range IDs raise
`ValueError` without advancing state or RNG; stepping before reset/after termination
raises `RuntimeError`.

### Enemy controllers

After an offensive intent, combo probability `p_c` decides whether another attack
or recovery follows. Recovery is always followed by an offensive intent. Attack
weights below are conditional probabilities in slash/sweep/lunge order.

| Controller | p_c | Attack weights/rule |
|---|---:|---|
| `balanced` | 0.50 | (1/3, 1/3, 1/3) |
| `aggressive` | 0.75 | (0.50, 0.20, 0.30) |
| `distance_aware` | 0.25 | 0.70 for slash when close, sweep when medium, lunge when far; 0.15 each otherwise |
| `mixed` | mean 0.50 | Independently select one of the three training controllers at each next-intent selection |
| `sweep_specialist` | 0.65 | (0.15, 0.65, 0.20); held-out policy evaluation reserved |
| `relentless_pursuer` | 0.85 | (0.10, 0.20, 0.70); held-out policy evaluation reserved |
| `enraged` | 0.35 / 0.85 | (0.20, 0.60, 0.20); high combo rate when boss HP <= 4; held-out evaluation reserved |

Mixed selection occurs before the combo draw, and no sampled identity persists.
Thus conditional offensive weights after an attack are weighted by component
combo probabilities. The observable health trigger also keeps the enraged rule
Markov. The RNG is local to each environment and seeded through Gymnasium.

### Rewards

Every step gives `-0.1 - 0.02 * stamina_spent`. Victory adds
`100 + 10 * remaining_player_hp / 6`; death/timeout adds `-100` without a health
bonus. Rest has zero stamina expenditure but still incurs the time cost.
`reward_mode="terminal_time"` removes only the health bonus and stamina penalty
for the planned reward ablation. The detailed reward terms in `info` sum to the
returned reward. There is no reward for waiting or merely staying alive.

For deterministic edge-case testing only,
`env.reset(seed=7, options={"state": [1, 2, 2, 0, 1, 1]})` initializes a validated
nonterminal fixture. Normal demo/smoke episodes always use the default reset.

## Rebuild figures and report

First run the demo and smoke commands above. Then:

```bash
python -m pip install -r requirements-report.txt
python tools/build_evidence.py
python tools/build_report.py --author "Your Name"
```

The evidence builder runs the tests, records their actual log, renders a verbatim
source excerpt with line numbers, summarizes measured output, and creates
`evidence/demo/demo.gif` from the saved PNG frames. The GIF is an optional animated
replay; no MP4 video is supplied. The report uses the saved measurements and images.
Replace the default `[Your Name]` via `--author` before submission.

To regenerate the required archive after any change:

```bash
python tools/package_submission.py
```

It writes `milestone2.zip` next to the extracted `milestone2/` directory, excluding
virtual environments, caches, and previous ZIPs. The final archive includes the
PDF, source, README, tests, evidence, and `MANIFEST.json` file hashes.

## Status and next steps

Implemented: complete environment; seven boss modes; legal-action masks; two reward
modes; seeded reset/step/render; human terminal play; three heuristic policies;
tests; reproducible evidence. No Q-learning, SARSA, DQN, or trained checkpoint is
included yet. The renderer is a schematic of discrete spacing, not a physics engine.

Next: calibrate heuristics and difficulty using development/validation episodes,
then freeze the design; implement masked Q-learning/SARSA; compare single/mixed
training with equal step budgets and multiple seeds; perform the reward ablation;
finally evaluate frozen policies on reserved behaviors. Any change to combat values
or reward weights must be logged before formal experiments. The small smoke sample
does not establish final performance or prove that mixed training is better.

## Interface references

- Gymnasium custom environment guide: https://gymnasium.farama.org/introduction/create_custom_env/
- Task termination versus external truncation: https://farama.org/Gymnasium-Terminated-Truncated-Step-API

Combat rules, controllers, policies, rendering, tests, and report content are specific
to this project; no prebuilt game environment is used.
