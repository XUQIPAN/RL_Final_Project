"""Scripted, stochastic boss controllers; no learning or hidden persistent type.

State order is (player_hp, boss_hp, stamina, distance, intent, turns_remaining).
Intent codes 0, 1, 2, 3 mean slash, sweep, lunge, recovery respectively.
"""

from collections.abc import Sequence

import numpy as np

TRAIN_OPPONENTS = ("balanced", "aggressive", "distance_aware")
HELD_OUT_OPPONENTS = ("sweep_specialist", "relentless_pursuer", "enraged")
OPPONENTS = TRAIN_OPPONENTS + ("mixed",) + HELD_OUT_OPPONENTS


def controller_parameters(
    name: str, state: Sequence[int]
) -> tuple[float, np.ndarray]:
    """Return combo probability and conditional slash/sweep/lunge weights.

For ``mixed``, these are the equivalent marginal parameters. Conditional
attack weights are combo-probability weighted after an offensive intent,
because high-combo components are more likely to generate another attack.
"""
    if name == "balanced":
        return 0.50, np.full(3, 1.0 / 3.0)
    if name == "aggressive":
        return 0.75, np.array([0.50, 0.20, 0.30])
    if name == "distance_aware":
        weights = np.full(3, 0.15)
        weights[int(state[3])] = 0.70
        return 0.25, weights
    if name == "sweep_specialist":
        return 0.65, np.array([0.15, 0.65, 0.20])
    if name == "relentless_pursuer":
        return 0.85, np.array([0.10, 0.20, 0.70])
    if name == "enraged":
        return (0.35 if state[1] > 4 else 0.85), np.array([0.20, 0.60, 0.20])
    if name == "mixed":
        parameters = [controller_parameters(n, state) for n in TRAIN_OPPONENTS]
        probabilities = np.array([p for p, _ in parameters])
        distributions = np.array([w for _, w in parameters])
        weights = (
            distributions.mean(axis=0)
            if state[4] == 3
            else np.average(distributions, axis=0, weights=probabilities)
        )
        return float(probabilities.mean()), weights
    raise ValueError(f"Unknown opponent {name!r}; choose one of {OPPONENTS}.")


def choose_next_intent(
    opponent: str, state: Sequence[int], rng: np.random.Generator
) -> int:
    """Sample the next intent using post-combat state and its resolved intent.

An independent controller draw is made for every selection in mixed mode.
After recovery the next intent is always offensive. No controller choice is
retained between decisions or exposed to the agent.
"""
    if opponent not in OPPONENTS:
        raise ValueError(f"Unknown opponent {opponent!r}.")
    controller = opponent
    if opponent == "mixed":
        controller = TRAIN_OPPONENTS[int(rng.integers(len(TRAIN_OPPONENTS)))]
    combo_probability, weights = controller_parameters(controller, state)
    if state[4] != 3 and rng.random() >= combo_probability:
        return 3
    return int(rng.choice(3, p=weights))
