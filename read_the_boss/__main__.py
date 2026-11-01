"""Reproducible environment demonstrations, human play, and heuristic checks.

Run ``python -m read_the_boss --help`` from the project root. No RL agent is
trained by these commands. The smoke suite deliberately excludes held-out bosses.
"""

import argparse
import json
import platform
from pathlib import Path

import gymnasium
import numpy as np

from . import Action, Distance, Intent, ReadTheBossEnv
from .controllers import OPPONENTS, TRAIN_OPPONENTS
from .policies import POLICIES


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_value(payload), indent=2) + "\n", encoding="utf-8")


def _state_text(observation) -> str:
    hp, boss_hp, stamina, distance, intent, turns = map(int, observation)
    return (
        f"HP={hp} BOSS={boss_hp} EN={stamina} "
        f"D={Distance(distance).name} NEXT={Intent(intent).name} LEFT={turns}"
    )


def _versions() -> dict:
    return {
        "python": platform.python_version(),
        "gymnasium": gymnasium.__version__,
        "numpy": np.__version__,
    }


def _save_frame(env, path: Path) -> None:
    from PIL import Image

    frame = env.render()
    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise RuntimeError("rgb_array rendering did not return an image array.")
    Image.fromarray(frame).save(path)


def run_demo(args) -> int:
    """Save one explicitly seeded baseline episode, including every transition."""
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    env = ReadTheBossEnv(opponent=args.opponent, render_mode="rgb_array")
    policy = POLICIES[args.policy]
    policy_rng = np.random.default_rng(np.random.SeedSequence([args.seed, 101]))
    observation, info = env.reset(seed=args.seed)
    trace = {
        "schema_version": 1,
        "kind": "heuristic_environment_demo",
        "trained_policy": False,
        "seed": args.seed,
        "opponent": args.opponent,
        "policy": args.policy,
        "versions": _versions(),
        "initial_observation": observation.copy(),
        "initial_info": info,
        "steps": [],
        "frames": [],
    }
    lines = [
        "READ THE BOSS | Milestone 2 environment demonstration",
        f"Heuristic policy={args.policy}; opponent={args.opponent}; seed={args.seed}",
        "No reinforcement learning training is performed in this demo.",
        "RESET " + _state_text(observation),
    ]
    if not args.no_images:
        _save_frame(env, output / "frame_000.png")
        trace["frames"].append("frame_000.png")
    total_return = 0.0
    total_stamina = 0
    invalid_actions = 0
    for turn in range(1, 41):
        before = observation.copy()
        action = policy(before, info["action_mask"], policy_rng)
        observation, reward, terminated, truncated, info = env.step(action)
        total_return += float(reward)
        total_stamina += int(info["stamina_spent"])
        invalid_actions += int(info["invalid_action"])
        trace["steps"].append(
            {
                "turn": turn,
                "action": int(action),
                "action_name": Action(action).name,
                "observation_before": before,
                "observation_after": observation.copy(),
                "reward": float(reward),
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "info": info,
            }
        )
        lines.append(
            f"t={turn:02d} {Action(action).name:<8} {_state_text(observation)} "
            f"R={reward:+.3f}"
        )
        lines.append("     " + "; ".join(info["events"]))
        if not args.no_images:
            name = f"frame_{turn:03d}.png"
            _save_frame(env, output / name)
            trace["frames"].append(name)
        if terminated or truncated:
            break
    else:
        raise RuntimeError("The environment failed to terminate within 40 turns.")
    trace["summary"] = {
        "outcome": info["outcome"],
        "turns": len(trace["steps"]),
        "return": total_return,
        "remaining_player_health": int(observation[0]),
        "remaining_boss_health": int(observation[1]),
        "stamina_spent": total_stamina,
        "invalid_actions": invalid_actions,
    }
    lines.append(
        f"END outcome={info['outcome']} turns={len(trace['steps'])} "
        f"return={total_return:+.3f} invalid_actions={invalid_actions}"
    )
    _write_json(output / "trace.json", trace)
    text = "\n".join(lines) + "\n"
    (output / "trace.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    print(f"Saved reproducible trace to {output / 'trace.json'}")
    if not args.no_images:
        print(f"Saved {len(trace['frames'])} actual environment frames to {output}")
    env.close()
    return 0


def _evaluate_policy(opponent: str, policy_name: str, episodes: int, first_seed: int):
    policy = POLICIES[policy_name]
    outcomes = {"victory": 0, "death": 0, "timeout": 0}
    returns, lengths, health_on_wins, lengths_on_wins = [], [], [], []
    total_stamina = 0
    invalid_actions = 0
    env = ReadTheBossEnv(opponent=opponent)
    for episode in range(episodes):
        seed = first_seed + episode
        observation, info = env.reset(seed=seed)
        rng = np.random.default_rng(np.random.SeedSequence([seed, 202]))
        episode_return = 0.0
        for turn in range(1, 41):
            action = policy(observation, info["action_mask"], rng)
            observation, reward, terminated, truncated, info = env.step(action)
            episode_return += float(reward)
            total_stamina += int(info["stamina_spent"])
            invalid_actions += int(info["invalid_action"])
            if terminated or truncated:
                break
        else:
            raise RuntimeError("The environment failed to terminate within 40 turns.")
        outcomes[info["outcome"]] += 1
        returns.append(episode_return)
        lengths.append(turn)
        if info["outcome"] == "victory":
            health_on_wins.append(int(observation[0]))
            lengths_on_wins.append(turn)
    env.close()
    return {
        "opponent": opponent,
        "policy": policy_name,
        "episodes": episodes,
        "wins": outcomes["victory"],
        "deaths": outcomes["death"],
        "timeouts": outcomes["timeout"],
        "win_rate": outcomes["victory"] / episodes,
        "mean_return": float(np.mean(returns)),
        "mean_turns": float(np.mean(lengths)),
        "total_environment_steps": int(sum(lengths)),
        "mean_health_among_wins": float(np.mean(health_on_wins)) if health_on_wins else None,
        "mean_turns_among_wins": float(np.mean(lengths_on_wins)) if lengths_on_wins else None,
        "total_stamina_spent": total_stamina,
        "invalid_actions": invalid_actions,
    }


def run_smoke(args) -> int:
    """Exercise all training controllers without contaminating held-out tests."""
    opponents = tuple(TRAIN_OPPONENTS) + ("mixed",)
    payload = {
        "schema_version": 1,
        "kind": "heuristic_environment_smoke_check",
        "trained_policy": False,
        "held_out_controllers_evaluated": False,
        "versions": _versions(),
        "settings": {
            "episodes_per_policy_and_opponent": args.episodes,
            "first_episode_seed": args.seed,
            "last_episode_seed": args.seed + args.episodes - 1,
            "policies": list(POLICIES),
            "opponents": list(opponents),
            "paired_episode_seed_lists": True,
        },
        "results": [],
    }
    print("HEURISTIC ENVIRONMENT SMOKE CHECK (no trained RL policy)")
    print("Held-out controllers excluded; shared episode seeds for every comparison.")
    print(f"Episodes per row: {args.episodes}; seeds {args.seed}..{args.seed + args.episodes - 1}")
    print(f"{'Opponent':<16} {'Policy':<11} {'Wins':>5} {'Death':>6} {'Time':>5} {'Win %':>7} {'Turns':>7}")
    for opponent in opponents:
        for policy_name in POLICIES:
            result = _evaluate_policy(opponent, policy_name, args.episodes, args.seed)
            payload["results"].append(result)
            print(
                f"{opponent:<16} {policy_name:<11} {result['wins']:>5} "
                f"{result['deaths']:>6} {result['timeouts']:>5} "
                f"{100 * result['win_rate']:>7.1f} {result['mean_turns']:>7.2f}"
            )
    payload["summary"] = {
        "episodes": len(payload["results"]) * args.episodes,
        "environment_steps": sum(row["total_environment_steps"] for row in payload["results"]),
        "invalid_actions": sum(row["invalid_actions"] for row in payload["results"]),
    }
    _write_json(Path(args.output), payload)
    print(f"Saved measured results to {args.output}")
    return 0


def run_play(args) -> int:
    """Play the actual environment through a terminal; no GUI is required."""
    env = ReadTheBossEnv(opponent=args.opponent, render_mode="ansi")
    observation, info = env.reset(seed=args.seed)
    print("READ THE BOSS | Human play. Enter an action number or name; q quits.")
    try:
        while True:
            print(env.render())
            legal = [Action(index) for index, allowed in enumerate(info["action_mask"]) if allowed]
            print("Legal actions: " + ", ".join(f"{int(action)}={action.name.lower()}" for action in legal))
            answer = input("> ").strip().lower()
            if answer in {"q", "quit", "exit"}:
                print("Stopped by user; this unfinished episode is not a scored timeout.")
                return 0
            try:
                action = Action(int(answer)) if answer.isdigit() else Action[answer.upper()]
            except (ValueError, KeyError):
                print("Choose one of the displayed legal actions.")
                continue
            if action not in legal:
                print("That action is unavailable in the current state.")
                continue
            observation, reward, terminated, truncated, info = env.step(int(action))
            print(f"Reward: {reward:+.3f}; " + "; ".join(info["events"]))
            if terminated or truncated:
                print(env.render())
                print(f"Fight ended: {info['outcome']}")
                return 0
    except (EOFError, KeyboardInterrupt):
        print("\nStopped by user; no completed outcome recorded.")
        return 0
    finally:
        env.close()


def _positive(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("must be a nonnegative integer")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    demo = commands.add_parser("demo", help="Record one seeded heuristic episode.")
    demo.add_argument("--opponent", choices=OPPONENTS, default="balanced")
    demo.add_argument("--policy", choices=tuple(POLICIES), default="reactive")
    demo.add_argument("--seed", type=_nonnegative, default=7)
    demo.add_argument("--output", default="evidence/demo")
    demo.add_argument("--no-images", action="store_true", help="Write text and JSON only.")
    demo.set_defaults(func=run_demo)
    play = commands.add_parser("play", help="Play interactively in a terminal.")
    play.add_argument("--opponent", choices=OPPONENTS, default="mixed")
    play.add_argument("--seed", type=_nonnegative, default=7)
    play.set_defaults(func=run_play)
    smoke = commands.add_parser("smoke", help="Run heuristic checks on training controllers only.")
    smoke.add_argument("--episodes", type=_positive, default=100)
    smoke.add_argument("--seed", type=_nonnegative, default=1000)
    smoke.add_argument("--output", default="evidence/smoke_results.json")
    smoke.set_defaults(func=run_smoke)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
