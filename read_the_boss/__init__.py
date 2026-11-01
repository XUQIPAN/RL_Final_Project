"""Read the Boss: a custom, fully observed tactical combat environment."""

from gymnasium.envs.registration import register, registry

from .controllers import HELD_OUT_OPPONENTS, OPPONENTS, TRAIN_OPPONENTS
from .env import Action, Distance, Intent, ReadTheBossEnv

# The observed 40-turn deadline belongs to the task; do not add TimeLimit.
if "ReadTheBoss-v0" not in registry:
    register(id="ReadTheBoss-v0", entry_point="read_the_boss.env:ReadTheBossEnv")

__all__ = [
    "Action", "Distance", "Intent", "ReadTheBossEnv",
    "TRAIN_OPPONENTS", "HELD_OUT_OPPONENTS", "OPPONENTS",
]
