"""Reproducible episode-budget training for the unchanged Read the Boss task."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import csv
import json
import math
from numbers import Integral
from pathlib import Path
from typing import Callable

import numpy as np

from .agents import ALGORITHMS, TabularAgent, environment_descriptor
from .env import ReadTheBossEnv


@dataclass(frozen=True)
class TrainingConfig:
    algorithm: str = "q_learning"
    opponent: str = "balanced"
    episodes: int = 20_000
    seed: int = 0
    alpha: float = 0.15
    gamma: float = 1.0
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_fraction: float = 0.8
    reward_mode: str = "full"
    checkpoint_every: int = 2_000

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {self.algorithm}.")
        if self.opponent not in ("balanced", "mixed"):
            raise ValueError("Training is restricted to balanced or mixed opponents.")
        for name, minimum in (("episodes", 1), ("seed", 0), ("checkpoint_every", 0)):
            value = getattr(self, name)
            if not isinstance(value, Integral) or isinstance(value, bool) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not np.isfinite(self.alpha) or not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1].")
        if not np.isfinite(self.gamma) or not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be in [0, 1].")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1.")
        if not 0 < self.epsilon_decay_fraction <= 1:
            raise ValueError("epsilon_decay_fraction must be in (0, 1].")
        if self.reward_mode not in ("full", "terminal_time"):
            raise ValueError("Unknown reward mode.")


def epsilon_for_episode(config: TrainingConfig, episode_index: int) -> float:
    """Linear decay over the first fraction of episodes; index is zero-based."""
    if episode_index < 0:
        raise ValueError("episode_index must be nonnegative.")
    decay_episodes = max(1, math.ceil(config.episodes * config.epsilon_decay_fraction))
    fraction = min(1.0, episode_index / max(1, decay_episodes - 1))
    return float(config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start))


def train(
    config: TrainingConfig,
    output_dir: str | Path,
    checkpoint_callback: Callable[[TabularAgent, int, int, float], object] | None = None,
) -> dict:
    """Train once, log every episode, and save only the final policy.

The callback evaluates a read-only agent at episode 0, periodic intervals,
and the final episode. ``checkpoint_every=0`` disables periodic callbacks,
but not initial/final callbacks. No training RNG is passed to evaluation.
Output paths must be unused to avoid silently overwriting another run.
"""
    if not isinstance(config, TrainingConfig):
        raise TypeError("config must be a TrainingConfig instance.")
    output_dir = Path(output_dir)
    agent_path = output_dir / "agent.npz"
    episodes_path = output_dir / "episodes.csv"
    metadata_path = output_dir / "metadata.json"
    if any(p.exists() for p in (agent_path, episodes_path, metadata_path)):
        raise FileExistsError("Training output already exists; choose an unused output directory.")
    output_dir.mkdir(parents=True, exist_ok=True)
    agent = TabularAgent(config.algorithm, config.alpha, config.gamma)
    seed_children = np.random.SeedSequence(config.seed).spawn(2)
    environment_rng = np.random.default_rng(seed_children[0])
    action_rng = np.random.default_rng(seed_children[1])
    env = ReadTheBossEnv(opponent=config.opponent, reward_mode=config.reward_mode)
    total_steps = 0
    outcomes: Counter = Counter()
    return_sum = 0.0
    invalid_total = 0
    callback_episodes: list[int] = []

    def evaluate_checkpoint(episode: int, epsilon: float) -> None:
        if checkpoint_callback is None:
            return
        before = agent.learning_digest()
        checkpoint_callback(agent, episode, total_steps, epsilon)
        if before != agent.learning_digest():
            raise RuntimeError("Checkpoint evaluation modified the learned agent.")
        callback_episodes.append(episode)

    evaluate_checkpoint(0, config.epsilon_start)
    fields = [
        "episode", "environment_seed", "environment_steps", "episode_return",
        "outcome", "win", "length", "player_hp", "boss_hp", "stamina_remaining",
        "stamina_spent", "zero_stamina_turns", "invalid_actions", "epsilon",
    ]
    try:
        with episodes_path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for episode_index in range(config.episodes):
                epsilon = epsilon_for_episode(config, episode_index)
                # Reserve the high half of uint32 for training environment seeds.
                environment_seed = int(environment_rng.integers(2**31, 2**32, dtype=np.uint64))
                observation, info = env.reset(seed=environment_seed)
                action = agent.act(observation, info["action_mask"], action_rng, epsilon)
                episode_return = 0.0
                length = stamina_spent = zero_stamina_turns = invalid_actions = 0
                while True:
                    zero_stamina_turns += int(observation[2] == 0)
                    next_observation, reward, terminated, truncated, next_info = env.step(action)
                    next_action = None
                    if not terminated and config.algorithm == "sarsa":
                        next_action = agent.act(
                            next_observation, next_info["action_mask"], action_rng, epsilon
                        )
                    agent.update(
                        observation, action, reward, next_observation,
                        next_info["action_mask"], terminated, next_action,
                    )
                    total_steps += 1
                    length += 1
                    episode_return += reward
                    stamina_spent += int(next_info["stamina_spent"])
                    invalid_actions += int(next_info["invalid_action"])
                    observation, info = next_observation, next_info
                    if terminated or truncated:
                        break
                    # SARSA carries its previously selected behavior action forward.
                    action = next_action if config.algorithm == "sarsa" else agent.act(
                        observation, info["action_mask"], action_rng, epsilon
                    )
                outcomes[info["outcome"]] += 1
                return_sum += episode_return
                invalid_total += invalid_actions
                writer.writerow({
                    "episode": episode_index + 1, "environment_seed": environment_seed,
                    "environment_steps": total_steps, "episode_return": episode_return,
                    "outcome": info["outcome"], "win": int(info["outcome"] == "victory"),
                    "length": length, "player_hp": int(observation[0]),
                    "boss_hp": int(observation[1]), "stamina_remaining": int(observation[2]),
                    "stamina_spent": stamina_spent, "zero_stamina_turns": zero_stamina_turns,
                    "invalid_actions": invalid_actions, "epsilon": epsilon,
                })
                episode = episode_index + 1
                scheduled = config.checkpoint_every > 0 and episode % config.checkpoint_every == 0
                if scheduled or episode == config.episodes:
                    stream.flush()
                    evaluate_checkpoint(episode, epsilon)
    finally:
        env.close()

    # Evaluation frequency is telemetry, not part of the learned model identity.
    learning_config = asdict(config)
    learning_config.pop("checkpoint_every")
    agent.save(agent_path, metadata={
        "training": learning_config, "training_seed": config.seed,
        "episode": config.episodes, "environment_steps": total_steps,
        "epsilon": epsilon_for_episode(config, config.episodes - 1),
    })
    summary = {
        "episodes": config.episodes, "environment_steps": total_steps,
        "updates": agent.updates, "training_win_rate": outcomes["victory"] / config.episodes,
        "mean_training_return": return_sum / config.episodes,
        "outcomes": dict(sorted(outcomes.items())), "invalid_actions": invalid_total,
        "visited_state_actions": int(np.count_nonzero(agent.visit_counts)),
        "learning_digest": agent.learning_digest(),
    }
    metadata = {
        "config": asdict(config), "environment": environment_descriptor(), "summary": summary,
        "callback_episodes": callback_episodes,
        "seed_protocol": "SeedSequence(seed).spawn(2): environment seeds in [2**31,2**32), independent action RNG",
        "selection": "Final episode only; no best-checkpoint or held-out model selection.",
        "budget": "Fixed episode count; total environment steps are measured, not matched.",
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "agent_path": str(agent_path), "episodes_csv": str(episodes_path),
        "metadata_path": str(metadata_path), "checkpoint_paths": [], "summary": summary,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="q_learning")
    parser.add_argument("--opponent", choices=("balanced", "mixed"), default="balanced")
    parser.add_argument("--episodes", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.15)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon-start", "--eps-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", "--eps-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-fraction", type=float, default=0.8)
    parser.add_argument("--reward-mode", choices=("full", "terminal_time"), default="full")
    parser.add_argument("--checkpoint-every", type=int, default=2_000)
    parser.add_argument("--eval-episodes", type=int, default=0,
                        help="Optional greedy checkpoint episodes; zero disables evaluation.")
    parser.add_argument("--eval-seed", type=int, default=100_000)
    args = parser.parse_args(argv)
    if args.eval_episodes < 0:
        parser.error("--eval-episodes must be nonnegative.")
    config = TrainingConfig(**{
        name: getattr(args, name) for name in TrainingConfig.__dataclass_fields__
    })
    callback = None
    if args.eval_episodes:
        from .evaluation import make_checkpoint_callback
        callback = make_checkpoint_callback(
            args.output, args.opponent, args.eval_episodes, args.eval_seed,
            algorithm=args.algorithm, training_seed=args.seed,
            reward_mode=args.reward_mode,
        )
    print(json.dumps(train(config, args.output, callback), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
