"""Greedy, reproducible evaluation with raw episode-level evidence.

Evaluation has its own environment and RNGs. Learned policies are never updated;
hash checks guard both action values and visit counts against accidental mutation.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .env import Action, ReadTheBossEnv
from .policies import POLICIES


EPISODE_FIELDS = [
    "policy", "algorithm", "training_opponent", "training_seed", "stage",
    "checkpoint_episode", "opponent", "episode_index", "evaluation_seed",
    "policy_seed", "outcome", "return", "turns", "player_health", "boss_health",
    "stamina_spent", "zero_stamina_turns", "invalid_actions", "terminated",
    "truncated",
]


def json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    return value


def write_json(path: str | Path, value: Any) -> None:
    """Write complete JSON atomically, without nonstandard NaN values."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(json_value(value), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def write_episode_csv(path: str | Path, rows: list[dict]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=EPISODE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _positive_integer(value: int, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer.")
    minimum = 0 if allow_zero else 1
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}.")
    return int(value)


def _table_fingerprint(policy) -> dict[str, str]:
    result = {}
    for name in ("q_values", "visit_counts"):
        if hasattr(policy, name):
            table = np.asarray(getattr(policy, name))
            digest = hashlib.sha256()
            digest.update(str(table.dtype).encode())
            digest.update(str(table.shape).encode())
            digest.update(table.tobytes(order="C"))
            result[name] = digest.hexdigest()
    if hasattr(policy, "updates"):
        result["updates"] = str(policy.updates)
    return result


def aggregate_episodes(rows: list[dict]) -> dict:
    """Summarize completed episodes; conditional means are null if no wins occur."""
    if not rows:
        raise ValueError("At least one completed episode is required.")
    count = len(rows)
    wins = [row for row in rows if row["outcome"] == "victory"]
    deaths = sum(row["outcome"] == "death" for row in rows)
    timeouts = sum(row["outcome"] == "timeout" for row in rows)
    if len(wins) + deaths + timeouts != count:
        raise ValueError("Evaluation rows must have completed terminal outcomes.")
    steps = sum(row["turns"] for row in rows)

    def mean(key: str, selected=rows):
        return float(np.mean([row[key] for row in selected])) if selected else None

    identifiers = {}
    for key in ("policy", "algorithm", "training_opponent", "training_seed",
                "opponent", "stage", "checkpoint_episode"):
        values = {row.get(key) for row in rows}
        identifiers[key] = values.pop() if len(values) == 1 else None
    return {
        **identifiers,
        "episodes": count,
        "wins": len(wins),
        "deaths": deaths,
        "timeouts": timeouts,
        "win_rate": len(wins) / count,
        "death_rate": deaths / count,
        "timeout_rate": timeouts / count,
        "mean_return": mean("return"),
        "mean_turns": mean("turns"),
        "mean_health_among_wins": mean("player_health", wins),
        "mean_turns_among_wins": mean("turns", wins),
        "mean_stamina_spent": mean("stamina_spent"),
        "mean_stamina_spent_among_wins": mean("stamina_spent", wins),
        "zero_stamina_turn_fraction": sum(row["zero_stamina_turns"] for row in rows) / steps,
        "invalid_actions": sum(row["invalid_actions"] for row in rows),
        "total_environment_steps": steps,
    }


def evaluate_policy(
    policy,
    opponent: str,
    *,
    episodes: int = 100,
    seed_start: int = 100000,
    policy_seed_start: int = 200000,
    policy_name: str | None = None,
    algorithm: str | None = None,
    training_opponent: str | None = None,
    training_seed: int | None = None,
    stage: str = "evaluation",
    checkpoint_episode: int | None = None,
    reward_mode: str = "full",
) -> dict:
    """Evaluate a TabularAgent or one of the named heuristic policies.

    Learned agents act with epsilon=0 (greedy with their documented tie rule).
    Environment seeds and action/tie RNG seeds are independent and reset for
    every episode. Episode return is the undiscounted sum of received rewards.
    """
    episodes = _positive_integer(episodes, "episodes")
    seed_start = _positive_integer(seed_start, "seed_start", allow_zero=True)
    policy_seed_start = _positive_integer(policy_seed_start, "policy_seed_start", allow_zero=True)
    if isinstance(policy, str):
        if policy not in POLICIES:
            raise ValueError(f"Unknown heuristic policy: {policy}")
        heuristic = POLICIES[policy]
        policy_name = policy_name or policy
        algorithm = algorithm or "heuristic"
        learned = False
    elif callable(getattr(policy, "act", None)):
        heuristic = None
        algorithm = algorithm or getattr(policy, "algorithm", "tabular")
        policy_name = policy_name or algorithm
        learned = True
    else:
        raise TypeError("policy must be a TabularAgent or a named heuristic.")
    before = _table_fingerprint(policy) if learned else {}
    env = ReadTheBossEnv(opponent=opponent, reward_mode=reward_mode)
    rows = []
    try:
        for index in range(episodes):
            episode_seed = seed_start + index
            action_seed = policy_seed_start + index
            rng = np.random.default_rng(action_seed)
            observation, info = env.reset(seed=episode_seed)
            total_return = 0.0
            stamina_spent = zero_stamina_turns = invalid_actions = 0
            for turn in range(1, 41):
                zero_stamina_turns += int(observation[2] == 0)
                if learned:
                    action = policy.act(observation, info["action_mask"], rng, epsilon=0.0)
                else:
                    action = heuristic(observation, info["action_mask"], rng)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total_return += float(reward)
                stamina_spent += int(info["stamina_spent"])
                invalid_actions += int(info["invalid_action"])
                if terminated or truncated:
                    break
            else:
                raise RuntimeError("Environment failed to end within its 40-turn horizon.")
            rows.append({
                "policy": policy_name,
                "algorithm": algorithm,
                "training_opponent": training_opponent,
                "training_seed": training_seed,
                "stage": stage,
                "checkpoint_episode": checkpoint_episode,
                "opponent": opponent,
                "episode_index": index,
                "evaluation_seed": episode_seed,
                "policy_seed": action_seed,
                "outcome": info["outcome"],
                "return": total_return,
                "turns": turn,
                "player_health": int(observation[0]),
                "boss_health": int(observation[1]),
                "stamina_spent": stamina_spent,
                "zero_stamina_turns": zero_stamina_turns,
                "invalid_actions": invalid_actions,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
            })
    finally:
        env.close()
    if learned and before != _table_fingerprint(policy):
        raise RuntimeError("Evaluation unexpectedly mutated the learned policy.")
    return {"episodes": rows, "summary": aggregate_episodes(rows)}


def make_checkpoint_callback(
    output_dir: str | Path,
    opponent: str,
    episodes: int,
    seed: int,
    *,
    policy_seed_start: int | None = None,
    algorithm: str | None = None,
    training_seed: int | None = None,
    run_id: str | None = None,
    reward_mode: str = "full",
):
    """Build a non-learning callback for the training loop's scheduled points."""
    output = Path(output_dir)
    points, raw_rows = [], []
    settings = {
        "opponent": opponent,
        "episodes": episodes,
        "seed_start": seed,
        "policy_seed_start": seed + 100000 if policy_seed_start is None else policy_seed_start,
        "epsilon": 0.0,
        "reward_mode": reward_mode,
        "selection": "None; final training episode is selected in advance.",
    }

    def callback(agent, episode, environment_steps, epsilon):
        result = evaluate_policy(
            agent, opponent, episodes=episodes, seed_start=seed,
            policy_seed_start=settings["policy_seed_start"], algorithm=algorithm,
            training_opponent=opponent, training_seed=training_seed,
            stage="checkpoint", checkpoint_episode=episode,
            reward_mode=reward_mode,
        )
        raw_rows.extend(result["episodes"])
        points.append({
            **result["summary"],
            "run_id": run_id,
            "environment_steps": int(environment_steps),
            "epsilon": float(epsilon),
        })
        write_episode_csv(output / "learning_curve_episodes.csv", raw_rows)
        write_json(output / "learning_curve.json", {"evaluation": settings, "points": points})
        label = run_id or f"{algorithm or agent.algorithm}_{opponent}"
        print(f"  {label}: episode={episode:,} steps={environment_steps:,} "
              f"greedy_win_rate={result['summary']['win_rate']:.3f}", flush=True)

    callback.points = points
    return callback


def record_trained_demo(
    agent,
    output_dir: str | Path,
    *,
    opponent: str = "balanced",
    seed: int = 7,
    policy_seed: int = 500007,
    metadata: dict | None = None,
    reward_mode: str = "full",
) -> dict:
    """Record the configured single episode; never search for a winning seed."""
    from PIL import Image

    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to replace existing demo evidence: {output}")
    output.mkdir(parents=True, exist_ok=True)
    before = _table_fingerprint(agent)
    rng = np.random.default_rng(policy_seed)
    env = ReadTheBossEnv(opponent=opponent, render_mode="rgb_array", reward_mode=reward_mode)
    observation, info = env.reset(seed=seed)
    trace = {
        "kind": "trained_greedy_policy_demo", "seed": seed,
        "policy_seed": policy_seed, "opponent": opponent, "reward_mode": reward_mode,
        "selection": "Model and episode seed predetermined in experiment config.",
        "metadata": metadata or {}, "initial_observation": observation.copy(), "steps": [],
    }
    frames = [Image.fromarray(env.render())]
    lines = [f"TRAINED GREEDY POLICY | opponent={opponent} seed={seed}",
             f"RESET {observation.tolist()}"]
    total_return = 0.0
    try:
        for turn in range(1, 41):
            previous = observation.copy()
            action = int(agent.act(observation, info["action_mask"], rng, epsilon=0.0))
            observation, reward, terminated, truncated, info = env.step(action)
            total_return += float(reward)
            trace["steps"].append({
                "turn": turn, "action": action, "action_name": Action(action).name,
                "observation_before": previous, "observation_after": observation.copy(),
                "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "info": info,
            })
            frames.append(Image.fromarray(env.render()))
            lines.append(f"{turn:02d} {Action(action).name:<8} {observation.tolist()} reward={reward:+.3f}")
            if terminated or truncated:
                break
        else:
            raise RuntimeError("Demo failed to end within 40 turns.")
    finally:
        env.close()
    if before != _table_fingerprint(agent):
        raise RuntimeError("Demo unexpectedly mutated the learned policy.")
    trace["summary"] = {"outcome": info["outcome"], "turns": len(trace["steps"]),
                        "return": total_return, "player_health": int(observation[0])}
    lines.append(f"END {info['outcome']} return={total_return:+.3f}")
    write_json(output / "trace.json", trace)
    (output / "trace.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    frames[0].save(output / "initial.png")
    frames[-1].save(output / "final.png")
    frames[0].save(output / "demo.gif", save_all=True, append_images=frames[1:],
                   duration=[900] * (len(frames) - 1) + [2500], loop=0, disposal=2)
    return trace["summary"]
