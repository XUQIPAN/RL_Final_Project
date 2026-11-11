"""Evaluate a saved model or heuristic and optionally render a seeded model replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .agents import TabularAgent
from .controllers import OPPONENTS
from .evaluation import evaluate_policy, record_trained_demo, write_episode_csv, write_json
from .policies import POLICIES


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--model", type=Path, help="A final agent.npz snapshot.")
    choice.add_argument("--policy", choices=tuple(POLICIES), help="An untrained heuristic.")
    parser.add_argument("--opponent", choices=OPPONENTS, default="mixed")
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=300000, help="First environment evaluation seed.")
    parser.add_argument("--policy-seed", type=int, default=400000)
    parser.add_argument("--reward-mode", choices=("full", "terminal_time"), default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--demo", action="store_true", help="Also save one model trace and GIF.")
    parser.add_argument("--demo-seed", type=int, default=7)
    args = parser.parse_args()
    if args.demo and args.model is None:
        parser.error("--demo requires --model; use the existing environment demo command for heuristics.")
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Output directory is not empty; use a fresh output directory.")
    policy = TabularAgent.load(args.model) if args.model else args.policy
    training = policy.metadata.get("training", {}) if args.model else {}
    reward_mode = args.reward_mode or training.get("reward_mode", "full")
    result = evaluate_policy(
        policy, args.opponent, episodes=args.episodes, seed_start=args.seed,
        policy_seed_start=args.policy_seed, reward_mode=reward_mode,
        training_opponent=training.get("opponent"), training_seed=training.get("seed"),
        stage="manual_evaluation", checkpoint_episode=training.get("episodes"),
    )
    write_episode_csv(args.output / "episodes.csv", result["episodes"])
    write_json(args.output / "summary.json", {
        "model": str(args.model) if args.model else None,
        "reward_mode": reward_mode, "epsilon": 0.0,
        "env_seed_start": args.seed, "policy_seed_start": args.policy_seed,
        "summary": result["summary"],
    })
    print(json.dumps(result["summary"], indent=2))
    if args.demo:
        summary = record_trained_demo(
            policy, args.output / "demo", opponent=args.opponent,
            seed=args.demo_seed, policy_seed=500000 + args.demo_seed,
            reward_mode=reward_mode,
            metadata={"model": str(args.model), "epsilon": 0.0},
        )
        print("Demo: " + json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
