"""Observation-only heuristic baselines; no policy in this module is trained.

Each policy receives the same six-component observation and legal-action mask.
The reactive policy knows combat rules but not the boss controller or RNG state.
"""

from collections.abc import Sequence

import numpy as np

from .env import Action, Distance, Intent


def _legal(mask: Sequence[int]) -> np.ndarray:
    allowed = np.asarray(mask, dtype=bool)
    if allowed.shape != (5,) or not allowed.any():
        raise ValueError("A policy needs a five-element mask with a legal action.")
    return allowed


def random_policy(observation, mask, rng: np.random.Generator) -> int:
    """Choose uniformly among currently legal actions."""
    del observation
    return int(rng.choice(np.flatnonzero(_legal(mask))))


def aggressive_policy(observation, mask, rng=None) -> int:
    """Approach, strike whenever possible, and rest when attacking is unavailable."""
    del observation, rng
    legal = _legal(mask)
    if legal[Action.STRIKE]:
        return int(Action.STRIKE)
    if legal[Action.APPROACH]:
        return int(Action.APPROACH)
    return int(Action.REST)


def _incoming_damage(intent: int, distance: int, guarding: bool = False) -> int:
    """Apply only the visible current cue's range and damage rules."""
    if intent == Intent.SLASH:
        damage = 2 if distance == Distance.CLOSE else 0
    elif intent == Intent.SWEEP:
        damage = 3 if distance <= Distance.MEDIUM else 0
    elif intent == Intent.LUNGE:
        damage = 1 if max(0, distance - 1) == Distance.CLOSE else 0
    else:
        damage = 0
    return max(0, damage - 2) if guarding else damage


def reactive_policy(observation, mask, rng=None) -> int:
    """Exploit safe openings and conserve enough stamina for visible threats.

    This fixed rule set is intentionally a reproducible baseline rather than
    a planner. It never reads the selected opponent name or future attacks.
    """
    del rng
    hp, boss_hp, stamina, distance, intent, _ = map(int, observation)
    legal = _legal(mask)
    incoming = _incoming_damage(intent, distance)

    # Player-first turn order makes a lethal strike safe against any cue.
    if legal[Action.STRIKE] and boss_hp <= 2:
        return int(Action.STRIKE)

    # A small lunge hit can be worth taking; waiting for perfect openings
    # otherwise risks exhausting the deadline or spending all stamina guarding.
    if legal[Action.STRIKE] and (incoming == 0 or (incoming == 1 and hp >= 3)):
        return int(Action.STRIKE)

    if incoming:
        retreat_damage = _incoming_damage(intent, min(2, distance + 1))
        guard_damage = _incoming_damage(intent, distance, guarding=True)
        if legal[Action.RETREAT] and retreat_damage == 0 and guard_damage > 0:
            return int(Action.RETREAT)
        if legal[Action.GUARD] and (stamina >= 2 or incoming >= hp):
            if guard_damage < hp:
                return int(Action.GUARD)
        if legal[Action.RETREAT] and retreat_damage == 0:
            return int(Action.RETREAT)
        # Rest under a nonlethal threat when the next useful action needs energy.
        # This can lose health, which is part of the resource-management tradeoff.
        return int(Action.REST)

    if stamina <= 2:
        return int(Action.REST)
    if legal[Action.APPROACH] and _incoming_damage(intent, distance - 1) == 0:
        return int(Action.APPROACH)
    return int(Action.REST)


POLICIES = {
    "random": random_policy,
    "aggressive": aggressive_policy,
    "reactive": reactive_policy,
}
