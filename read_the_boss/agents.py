"""Masked tabular Q-learning and SARSA with portable, deterministic snapshots."""

from __future__ import annotations

import hashlib
import io
import json
import os
from numbers import Integral
from pathlib import Path
from typing import Any
import zipfile

import numpy as np

STATE_DIMS = (7, 9, 5, 3, 4, 41)
STATE_FIELDS = ("player_hp", "boss_hp", "stamina", "distance", "intent", "turns_remaining")
ACTION_NAMES = ("approach", "retreat", "strike", "guard", "rest")
ALGORITHMS = ("q_learning", "sarsa")
FORMAT_VERSION = 1


def environment_descriptor() -> dict[str, Any]:
    """Fingerprint mechanics, not opponent choice: held-out evaluation is allowed."""
    package = Path(__file__).resolve().parent
    return {
        "id": "ReadTheBoss-v0",
        "state_dims": list(STATE_DIMS),
        "state_fields": list(STATE_FIELDS),
        "actions": list(ACTION_NAMES),
        "source_sha256": {
            name: hashlib.sha256((package / name).read_bytes()).hexdigest()
            for name in ("env.py", "controllers.py")
        },
    }


class TabularAgent:
    """One value for each complete state/action, including the remaining clock.

The dense table also reserves unreachable and terminal combinations. All values
start at zero; unseen-state ties are broken uniformly among legal actions.
The caller owns the policy RNG, so evaluation cannot consume training randomness.
"""

    def __init__(
        self, algorithm: str = "q_learning", alpha: float = 0.15, gamma: float = 1.0
    ) -> None:
        if algorithm not in ALGORITHMS:
            raise ValueError(f"algorithm must be one of {ALGORITHMS}.")
        if not np.isfinite(alpha) or not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not np.isfinite(gamma) or not 0 <= gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        self.algorithm = algorithm
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.q_values = np.zeros(STATE_DIMS + (len(ACTION_NAMES),), dtype=np.float64)
        self.visit_counts = np.zeros_like(self.q_values, dtype=np.uint32)
        self.updates = 0
        self.metadata: dict[str, Any] = {}

    @staticmethod
    def state_key(observation) -> tuple[int, ...]:
        values = np.asarray(observation)
        if values.shape != (6,) or values.dtype.kind not in "iu":
            raise ValueError("Observation must contain exactly six integers.")
        key = tuple(int(v) for v in values)
        if any(v < 0 or v >= bound for v, bound in zip(key, STATE_DIMS)):
            raise ValueError("Observation is outside the environment state bounds.")
        return key

    @staticmethod
    def _legal_actions(mask) -> np.ndarray:
        values = np.asarray(mask)
        if values.shape != (5,) or not np.all((values == 0) | (values == 1)):
            raise ValueError("Action mask must contain five Boolean/0-or-1 values.")
        legal = np.flatnonzero(values)
        if legal.size == 0:
            raise ValueError("A nonterminal decision requires at least one legal action.")
        return legal

    def act(
        self, observation, mask, rng: np.random.Generator, epsilon: float = 0.0
    ) -> int:
        """Sample an epsilon-greedy action, masking exploration and exploitation."""
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        key = self.state_key(observation)
        legal = self._legal_actions(mask)
        if epsilon > 0 and rng.random() < epsilon:
            return int(rng.choice(legal))
        values = self.q_values[key][legal]
        tied_best = legal[values == values.max()]
        return int(rng.choice(tied_best))

    def update(
        self, observation, action: int, reward: float, next_observation,
        next_mask, terminated: bool, next_action: int | None = None,
    ) -> float:
        """Apply one TD update and return its pre-update TD error.

SARSA's next_action must be the action actually selected by the behavior
policy and carried into the next environment step, not a new greedy sample.
True terminals (including this task's timeout) never bootstrap.
"""
        key = self.state_key(observation)
        if not isinstance(action, Integral) or isinstance(action, (bool, np.bool_)):
            raise ValueError("action must be an integer ID.")
        if not 0 <= int(action) < 5 or not np.isfinite(reward):
            raise ValueError("Action must be 0--4 and reward must be finite.")
        target = float(reward)
        if not terminated:
            next_key = self.state_key(next_observation)
            legal = self._legal_actions(next_mask)
            if self.algorithm == "q_learning":
                bootstrap = float(self.q_values[next_key][legal].max())
            else:
                if (not isinstance(next_action, Integral)
                        or isinstance(next_action, (bool, np.bool_))
                        or int(next_action) not in legal):
                    raise ValueError("Nonterminal SARSA requires its selected legal next_action.")
                bootstrap = float(self.q_values[next_key + (int(next_action),)])
            target += self.gamma * bootstrap
        index = key + (int(action),)
        td_error = target - float(self.q_values[index])
        self.q_values[index] += self.alpha * td_error
        self.visit_counts[index] += 1
        self.updates += 1
        return float(td_error)

    def learning_digest(self) -> str:
        """Hash learned values and counters, useful for read-only evaluation checks."""
        digest = hashlib.sha256(self.q_values.tobytes(order="C"))
        digest.update(self.visit_counts.tobytes(order="C"))
        digest.update(str(self.updates).encode("ascii"))
        return digest.hexdigest()

    def save(self, path: str | Path, metadata: dict | None = None) -> Path:
        """Write a compressed NPZ with fixed ZIP timestamps and JSON metadata.

No pickle/object arrays are used. Identical values and metadata produce
identical archive bytes; paths and wall-clock times are not embedded.
"""
        path = Path(path)
        if path.suffix != ".npz":
            raise ValueError("Agent snapshot must have a .npz extension.")
        combined = dict(self.metadata)
        if metadata is not None:
            combined.update(metadata)
        document = {
            "format_version": FORMAT_VERSION,
            "algorithm": self.algorithm, "alpha": self.alpha, "gamma": self.gamma,
            "updates": self.updates, "environment": environment_descriptor(),
            "metadata": combined,
        }
        payloads = {
            "q_values": self.q_values,
            "visit_counts": self.visit_counts,
            "metadata": np.array(json.dumps(document, sort_keys=True, separators=(",", ":"))),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".tmp")
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, values in payloads.items():
                stream = io.BytesIO()
                np.lib.format.write_array(stream, np.asarray(values), allow_pickle=False)
                member = zipfile.ZipInfo(name + ".npy", date_time=(1980, 1, 1, 0, 0, 0))
                member.compress_type = zipfile.ZIP_DEFLATED
                member.external_attr = 0o600 << 16
                archive.writestr(member, stream.getvalue(), compresslevel=6)
        os.replace(temporary, path)
        self.metadata = combined
        return path

    @classmethod
    def load(cls, path: str | Path, env=None) -> "TabularAgent":
        """Load a snapshot and reject incompatible mechanics or malformed tables."""
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"q_values", "visit_counts", "metadata"}:
                raise ValueError("Snapshot has unexpected or missing arrays.")
            document = json.loads(str(archive["metadata"].item()))
            if document.get("format_version") != FORMAT_VERSION:
                raise ValueError("Unsupported agent snapshot version.")
            if document.get("environment") != environment_descriptor():
                raise ValueError("Snapshot is incompatible with current environment mechanics.")
            agent = cls(document["algorithm"], document["alpha"], document["gamma"])
            q_values, visits = archive["q_values"], archive["visit_counts"]
            expected = STATE_DIMS + (5,)
            if (q_values.shape != expected or q_values.dtype != np.dtype("float64")
                    or not np.isfinite(q_values).all()):
                raise ValueError("Invalid Q-table shape, dtype, or non-finite value.")
            if visits.shape != expected or visits.dtype != np.dtype("uint32"):
                raise ValueError("Invalid visit-count table.")
            updates = document.get("updates")
            if not isinstance(updates, int) or updates < 0 or int(visits.sum()) != updates:
                raise ValueError("Snapshot update counter does not match its visit counts.")
            if not isinstance(document.get("metadata"), dict):
                raise ValueError("Snapshot metadata must be a JSON object.")
            agent.q_values[:] = q_values
            agent.visit_counts[:] = visits
            agent.updates = updates
            agent.metadata = document["metadata"]
        if env is not None:
            base = env.unwrapped
            if (tuple(base.observation_space.nvec) != STATE_DIMS
                    or base.action_space.n != 5):
                raise ValueError("Provided environment has incompatible spaces.")
        return agent
