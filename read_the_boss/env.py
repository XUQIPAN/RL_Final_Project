"""Custom finite-horizon combat environment implementing the Gymnasium API."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from enum import IntEnum
from numbers import Integral
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .controllers import OPPONENTS, choose_next_intent


class Action(IntEnum):
    APPROACH = 0
    RETREAT = 1
    STRIKE = 2
    GUARD = 3
    REST = 4


class Intent(IntEnum):
    SLASH = 0
    SWEEP = 1
    LUNGE = 2
    RECOVERY = 3


class Distance(IntEnum):
    CLOSE = 0
    MEDIUM = 1
    FAR = 2


class ReadTheBossEnv(gym.Env):
    """A player-first 1v1 duel with a visible next boss intent.

Observation: [player_hp, boss_hp, stamina, distance, intent, turns_remaining].
Legal policies use ``info['action_mask']``. If a caller submits a masked but
in-range action, it wastes a turn: no player effect, no stamina charge, and
normal boss retaliation. Invalid action IDs/types raise without mutation.

The 40-turn deadline is part of the task and observation, so timeout is a
terminal loss (terminated=True), not an external rollout truncation.
"""

    metadata = {"render_modes": ["ansi", "rgb_array"], "render_fps": 2}
    MAX_PLAYER_HP = 6
    MAX_BOSS_HP = 8
    MAX_STAMINA = 4
    MAX_TURNS = 40
    INITIAL_STATE = (6, 8, 4, 1, 3, 40)

    def __init__(
        self,
        opponent: str = "balanced",
        render_mode: str | None = None,
        reward_mode: str = "full",
    ) -> None:
        super().__init__()
        if opponent not in OPPONENTS:
            raise ValueError(f"Unknown opponent {opponent!r}; choose {OPPONENTS}.")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError("render_mode must be None, 'ansi', or 'rgb_array'.")
        if reward_mode not in ("full", "terminal_time"):
            raise ValueError("reward_mode must be 'full' or 'terminal_time'.")
        self.opponent = opponent
        self.render_mode = render_mode
        self.reward_mode = reward_mode
        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.MultiDiscrete(
            [7, 9, 5, 3, 4, 41], dtype=np.int64
        )
        self._state: tuple[int, ...] | None = None
        self._terminated = False
        self._last_info: dict[str, Any] = {}

    @property
    def state(self) -> tuple[int, ...]:
        """An immutable snapshot; modifying observations cannot change it."""
        if self._state is None:
            raise RuntimeError("Call reset() before accessing the state.")
        return self._state

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("options must be a mapping or None.")
        options = {} if options is None else options
        if set(options) - {"state"}:
            raise ValueError("The only supported reset option is 'state'.")
        initial = options.get("state", self.INITIAL_STATE)
        self._state = self._validate_initial_state(initial)
        self._terminated = False
        info = self._info(events=["Reset: ready for the first decision."])
        self._last_info = deepcopy(info)
        return self._observation(), info

    @staticmethod
    def _validate_initial_state(values: Sequence[int]) -> tuple[int, ...]:
        try:
            values = tuple(values)
        except TypeError as exc:
            raise ValueError("Initial state must contain six integers.") from exc
        if len(values) != 6 or any(
            not isinstance(v, Integral) or isinstance(v, (bool, np.bool_))
            for v in values
        ):
            raise ValueError("Initial state must contain six integers.")
        state = tuple(int(v) for v in values)
        bounds = ((1, 6), (1, 8), (0, 4), (0, 2), (0, 3), (1, 40))
        if any(not lower <= v <= upper for v, (lower, upper) in zip(state, bounds)):
            raise ValueError("Initial state is outside the nonterminal bounds.")
        return state

    def action_masks(self) -> np.ndarray:
        """Return state-dependent legality; terminal states have no legal action."""
        _, _, stamina, distance, _, _ = self.state
        if self._terminated:
            return np.zeros(len(Action), dtype=bool)
        return np.array(
            [distance > 0, distance < 2 and stamina >= 1,
             distance == 0 and stamina >= 2, stamina >= 1, True],
            dtype=bool,
        )

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        if self._state is None or self._terminated:
            raise RuntimeError("Call reset() before stepping a new episode.")
        if not isinstance(action, Integral) or isinstance(action, (bool, np.bool_)):
            raise ValueError("Action must be an integer ID from 0 to 4.")
        if not self.action_space.contains(action):
            raise ValueError("Action must be an integer ID from 0 to 4.")

        action = Action(int(action))
        state = list(self.state)
        events: list[str] = []
        invalid = not bool(self.action_masks()[action])
        spent, dealt, guard = self._apply_player_action(state, action, invalid, events)
        taken = self._resolve_boss(state, guard, events) if state[1] > 0 else 0
        if state[1] == 0:
            events.append("Lethal strike: boss retaliation cancelled.")
        state[5] -= 1
        outcome = self._outcome(state)
        self._terminated = outcome != "ongoing"
        if not self._terminated:
            state[4] = choose_next_intent(self.opponent, state, self.np_random)
            events.append(f"Next intent: {Intent(state[4]).name.lower()}.")
        else:
            events.append(f"Episode ended: {outcome}.")
        self._state = tuple(state)
        terms = self._reward_terms(outcome, spent)
        reward = float(sum(terms.values()))
        info = self._info(outcome, invalid, events, spent, dealt, taken, terms)
        self._last_info = deepcopy(info)
        return self._observation(), reward, self._terminated, False, info

    @staticmethod
    def _apply_player_action(
        state: list[int], action: Action, invalid: bool, events: list[str]
    ) -> tuple[int, int, int]:
        if invalid:
            events.append(f"{action.name.lower()} is masked: the turn is wasted.")
            return 0, 0, 0
        spent = {Action.RETREAT: 1, Action.STRIKE: 2, Action.GUARD: 1}.get(action, 0)
        state[2] -= spent
        dealt, guard = 0, 0
        if action == Action.APPROACH:
            state[3] -= 1
            events.append("Approach: moved one band closer.")
        elif action == Action.RETREAT:
            state[3] += 1
            events.append("Retreat: moved one band farther away.")
        elif action == Action.STRIKE:
            dealt = min(2, state[1])
            state[1] -= dealt
            events.append(f"Strike: dealt {dealt} damage.")
        elif action == Action.GUARD:
            guard = 2
            events.append("Guard: block up to 2 damage this turn.")
        else:
            regained = min(2, 4 - state[2])
            state[2] += regained
            events.append(f"Rest: recovered {regained} stamina.")
        return spent, dealt, guard

    @staticmethod
    def _resolve_boss(state: list[int], guard: int, events: list[str]) -> int:
        intent = Intent(state[4])
        damage = 0
        if intent == Intent.SLASH and state[3] == Distance.CLOSE:
            damage = 2
        elif intent == Intent.SWEEP and state[3] <= Distance.MEDIUM:
            damage = 3
        elif intent == Intent.LUNGE:
            state[3] = max(0, state[3] - 1)
            damage = 1 if state[3] == Distance.CLOSE else 0
        taken = min(state[0], max(0, damage - guard))
        state[0] -= taken
        events.append(f"Boss {intent.name.lower()}: player takes {taken} damage.")
        return taken

    @staticmethod
    def _outcome(state: Sequence[int]) -> str:
        if state[1] == 0:
            return "victory"
        if state[0] == 0:
            return "death"
        if state[5] == 0:
            return "timeout"
        return "ongoing"

    def _reward_terms(self, outcome: str, spent: int) -> dict[str, float]:
        full = self.reward_mode == "full"
        terminal = 100.0 if outcome == "victory" else (
            -100.0 if outcome in ("death", "timeout") else 0.0
        )
        return {
            "time": -0.1,
            "stamina": -0.02 * spent if full else 0.0,
            "terminal": terminal,
            "health": 10.0 * self.state[0] / 6.0 if full and outcome == "victory" else 0.0,
        }

    def _info(
        self, outcome: str = "ongoing", invalid: bool = False,
        events: list[str] | None = None, spent: int = 0, dealt: int = 0,
        taken: int = 0, terms: dict[str, float] | None = None,
    ) -> dict[str, Any]:
        return {
            "action_mask": self.action_masks().astype(np.int8),
            "outcome": outcome,
            "invalid_action": invalid,
            "events": [] if events is None else list(events),
            "stamina_spent": spent,
            "damage_dealt": dealt,
            "damage_taken": taken,
            "reward_terms": dict.fromkeys(("time", "stamina", "terminal", "health"), 0.0)
            if terms is None else dict(terms),
        }

    def _observation(self) -> np.ndarray:
        return np.array(self.state, dtype=np.int64)

    def render(self) -> str | np.ndarray | None:
        if self.render_mode is None:
            return None
        if self.render_mode == "rgb_array":
            from .render import render_frame
            return render_frame(
                self.state, self._last_info,
                opponent=self.opponent, reward_mode=self.reward_mode,
            )
        hp, boss_hp, stamina, distance, intent, remaining = self.state
        available = ", ".join(a.name.lower() for a in Action if self.action_masks()[a])
        return (
            f"READ THE BOSS | opponent={self.opponent} | turns left={remaining}\n"
            f"Player HP {hp}/6 | Boss HP {boss_hp}/8 | Stamina {stamina}/4\n"
            f"Distance: {Distance(distance).name.lower()} | "
            f"Announced intent: {Intent(intent).name.lower()}\n"
            f"Legal actions: {available or '(episode ended)'}\n"
            f"Outcome: {self._last_info.get('outcome', 'ongoing')}"
        )

    def close(self) -> None:
        """No windows or external resources are retained by this environment."""

