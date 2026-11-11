"""Run the predetermined Milestone 3 experiment matrix and preserve evidence.

Example:
    python -m read_the_boss.experiments --config configs/milestone3.json \
        --output experiments/milestone3

Use --quick for a separate integration run. It never evaluates held-out bosses.
Existing nonempty output directories are rejected, rather than overwritten.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .agents import TabularAgent
from .controllers import HELD_OUT_OPPONENTS, OPPONENTS, TRAIN_OPPONENTS
from .evaluation import (
    aggregate_episodes, evaluate_policy, make_checkpoint_callback,
    record_trained_demo, write_episode_csv, write_json,
)
from .policies import POLICIES
from .training import TrainingConfig, train


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git(repo: Path, *arguments: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *arguments], capture_output=True, text=True, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _provenance(config_path: Path, config: dict, quick: bool) -> dict:
    repo = Path(__file__).resolve().parents[1]
    source_files = sorted(
        list((repo / "read_the_boss").glob("*.py"))
        + list((repo / "tests").glob("*.py"))
        + list(repo.glob("requirements*.txt"))
        + list((repo / "configs").glob("*.json"))
    )
    versions = {"python": platform.python_version()}
    for name in ("gymnasium", "numpy", "Pillow", "matplotlib"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    status = _git(repo, "status", "--short")
    return {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": _utc_now(),
        "quick_integration_run": quick,
        "base_commit": _git(repo, "rev-parse", "HEAD"),
        "git_worktree_dirty": bool(status),
        "git_status_at_start": status,
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_files
        },
        "input_config_path": str(config_path),
        "input_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "resolved_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode("utf-8")
        ).hexdigest(),
        "versions": versions,
        "platform": platform.platform(),
        "command": sys.argv,
        "config": config,
        "protocol": {
            "selection": "Use the final configured training episode; no checkpoint or held-out selection.",
            "checkpoint_policy": "Greedy; evaluated only on the run's training opponent.",
            "training_seed_stream": "Environment seeds in [2**31, 2**32); independent action RNG.",
            "evaluation_seed_stream": "env_seed_start+i; separate action/tie RNG policy_seed_start+i.",
            "paired_evaluation": "Same final environment and action seed lists for all models and baselines.",
            "uncertainty": "Sample standard deviation across independent training seeds, ddof=1; not episode-level SEM.",
            "baselines": "Each heuristic/opponent pair evaluated once, not replicated as training seeds.",
            "returns": "Undiscounted episode reward sums; discount factor recorded separately.",
            "budget": "Equal training episodes in this initial study; actual environment steps are recorded.",
        },
        "completed_runs": [],
    }


def _integer(value, label: str, *, positive: bool = True) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer.")
    if value < (1 if positive else 0):
        raise ValueError(f"{label} is out of range.")
    return value


def _distinct_nonempty(values, label: str):
    if not isinstance(values, list) or not values or len(set(values)) != len(values):
        raise ValueError(f"{label} must be a nonempty list with no duplicates.")


def load_config(path: str | Path, *, quick: bool = False) -> dict:
    """Validate the protocol and apply explicit, separately labelled quick settings."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported experiment configuration schema.")
    training = config["training"]
    for key in ("algorithms", "opponents", "seeds"):
        _distinct_nonempty(training[key], f"training.{key}")
    if not set(training["algorithms"]) <= {"q_learning", "sarsa"}:
        raise ValueError("Training algorithms must be q_learning and/or sarsa.")
    if not set(training["opponents"]) <= set(TRAIN_OPPONENTS) | {"mixed"}:
        raise ValueError("Held-out controllers cannot be used for training.")
    for seed in training["seeds"]:
        _integer(seed, "training seed", positive=False)
    for algorithm, opponent, seed in itertools.product(
        training["algorithms"], training["opponents"], training["seeds"]
    ):
        TrainingConfig(algorithm=algorithm, opponent=opponent, seed=seed,
                       **{key: value for key, value in training.items()
                          if key not in {"algorithms", "opponents", "seeds"}})
    for name in ("checkpoint_evaluation", "final_evaluation"):
        item = config[name]
        _integer(item["episodes"], f"{name}.episodes")
        _integer(item["env_seed_start"], f"{name}.env_seed_start", positive=False)
        _integer(item["policy_seed_start"], f"{name}.policy_seed_start", positive=False)
    _distinct_nonempty(config["final_evaluation"]["opponents"], "final opponents")
    if not set(config["final_evaluation"]["opponents"]) <= set(OPPONENTS):
        raise ValueError("Unknown final evaluation opponent.")
    _distinct_nonempty(config["baselines"], "baselines")
    if not set(config["baselines"]) <= set(POLICIES):
        raise ValueError("Unknown baseline policy.")
    cp = config["checkpoint_evaluation"]
    final = config["final_evaluation"]
    checkpoint_range = range(cp["env_seed_start"], cp["env_seed_start"] + cp["episodes"])
    final_range = range(final["env_seed_start"], final["env_seed_start"] + final["episodes"])
    if max(checkpoint_range.start, final_range.start) < min(checkpoint_range.stop, final_range.stop):
        raise ValueError("Checkpoint and final environment seed lists must be disjoint.")
    for name, settings in (("checkpoint", cp), ("final", final)):
        if settings["env_seed_start"] + settings["episodes"] > 2**31:
            raise ValueError(f"{name} evaluation seeds overlap the training seed domain.")
    demo = config.get("demo", {"enabled": False})
    if demo.get("enabled"):
        if (demo["algorithm"] not in training["algorithms"]
                or demo["training_opponent"] not in training["opponents"]
                or demo["training_seed"] not in training["seeds"]):
            raise ValueError("Configured demo model is not in the training matrix.")
        if demo["opponent"] not in OPPONENTS:
            raise ValueError("Unknown demonstration opponent.")
        _integer(demo["seed"], "demo.seed", positive=False)
        _integer(demo["policy_seed"], "demo.policy_seed", positive=False)
    if quick:
        config = copy.deepcopy(config)
        config["experiment_name"] += "_quick_integration"
        config["training"]["episodes"] = 40
        config["training"]["checkpoint_every"] = 20
        config["training"]["seeds"] = [config["training"]["seeds"][0]]
        config["checkpoint_evaluation"]["episodes"] = 5
        config["final_evaluation"]["episodes"] = 5
        config["final_evaluation"]["opponents"] = list(TRAIN_OPPONENTS) + ["mixed"]
        config["demo"] = {"enabled": False}
        config["description"] = "Integration test only; no held-out-controller evaluation or scientific result."
    return config


def _sample_mean_sd(values: list[float]) -> tuple[float | None, float | None]:
    selected = [value for value in values if value is not None]
    if not selected:
        return None, None
    return (float(np.mean(selected)),
            float(np.std(selected, ddof=1)) if len(selected) > 1 else None)


def _condition_aggregates(raw_rows: list[dict]) -> list[dict]:
    keys = sorted({(row["algorithm"], row["training_opponent"], row["opponent"])
                   for row in raw_rows})
    result = []
    metrics = (
        "win_rate", "death_rate", "timeout_rate", "mean_return", "mean_turns",
        "mean_health_among_wins", "mean_turns_among_wins", "mean_stamina_spent",
        "mean_stamina_spent_among_wins", "zero_stamina_turn_fraction",
    )
    for algorithm, training_opponent, opponent in keys:
        rows = [row for row in raw_rows if
                (row["algorithm"], row["training_opponent"], row["opponent"])
                == (algorithm, training_opponent, opponent)]
        seeds = sorted({row["training_seed"] for row in rows})
        summaries = [aggregate_episodes([row for row in rows if row["training_seed"] == seed])
                     for seed in seeds]
        record = {
            "algorithm": algorithm, "training_opponent": training_opponent,
            "opponent": opponent, "training_seeds": seeds, "n_training_seeds": len(seeds),
            "pooled": aggregate_episodes(rows), "by_seed": summaries,
        }
        for metric in metrics:
            mean, sd = _sample_mean_sd([item[metric] for item in summaries])
            record[f"{metric}_mean"] = mean
            record[f"{metric}_sd"] = sd
        result.append(record)
    return result


def _held_out_aggregates(training_runs: list[dict]) -> list[dict]:
    groups = {}
    for run in training_runs:
        summaries = [summary for summary in run["final_evaluation"]
                     if summary["opponent"] in HELD_OUT_OPPONENTS]
        if not summaries:
            continue
        record = {
            "training_seed": run["training_seed"],
            "opponents": [item["opponent"] for item in summaries],
            "mean_win_rate": float(np.mean([item["win_rate"] for item in summaries])),
            "worst_win_rate": min(item["win_rate"] for item in summaries),
        }
        key = (run["algorithm"], run["training_opponent"])
        groups.setdefault(key, []).append(record)
    result = []
    for (algorithm, opponent), records in sorted(groups.items()):
        mean_mean, mean_sd = _sample_mean_sd([item["mean_win_rate"] for item in records])
        worst_mean, worst_sd = _sample_mean_sd([item["worst_win_rate"] for item in records])
        result.append({
            "algorithm": algorithm, "training_opponent": opponent,
            "training_seeds": [item["training_seed"] for item in records],
            "n_training_seeds": len(records), "by_seed": records,
            "mean_win_rate_mean": mean_mean, "mean_win_rate_sd": mean_sd,
            "worst_win_rate_mean": worst_mean, "worst_win_rate_sd": worst_sd,
        })
    return result


def run_experiments(config_path: str | Path, output_dir: str | Path, *, quick: bool = False) -> dict:
    """Train the fixed matrix, then evaluate only final models and fixed baselines."""
    config_path = Path(config_path).resolve()
    output = Path(output_dir).resolve()
    config = load_config(config_path, quick=quick)
    if quick and "quick" not in output.name.lower():
        raise ValueError("--quick output directory name must contain 'quick' to separate pilot evidence.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty; choose a new directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = _provenance(config_path, config, quick)
    write_json(output / "manifest.json", manifest)
    write_json(output / "resolved_config.json", config)
    training = config["training"]
    checkpoint = config["checkpoint_evaluation"]
    final = config["final_evaluation"]
    runs, curves, all_final_rows = [], [], []
    matrix = list(itertools.product(training["algorithms"], training["opponents"], training["seeds"]))
    try:
        # Final/held-out evaluation starts only after every training run is complete.
        for index, (algorithm, opponent, seed) in enumerate(matrix, 1):
            run_id = f"{algorithm}_{opponent}_seed{seed}"
            run_dir = output / "runs" / run_id
            print(f"TRAIN {index}/{len(matrix)} {run_id}", flush=True)
            callback = make_checkpoint_callback(
                run_dir, opponent, checkpoint["episodes"], checkpoint["env_seed_start"],
                policy_seed_start=checkpoint["policy_seed_start"], algorithm=algorithm,
                training_seed=seed, run_id=run_id,
                reward_mode=training["reward_mode"],
            )
            train_config = TrainingConfig(
                algorithm=algorithm, opponent=opponent, seed=seed,
                **{key: value for key, value in training.items()
                   if key not in {"algorithms", "opponents", "seeds"}},
            )
            train_summary = train(train_config, run_dir, checkpoint_callback=callback)
            record = {
                "run_id": run_id, "algorithm": algorithm, "training_opponent": opponent,
                "training_seed": seed, "training_summary": train_summary,
                "environment_steps": callback.points[-1]["environment_steps"],
                "model_path": str((run_dir / "agent.npz").relative_to(output)),
                "final_evaluation": [],
            }
            runs.append(record)
            curves.extend(callback.points)
            manifest["completed_runs"].append(run_id)
            write_json(output / "manifest.json", manifest)
            write_json(output / "learning_curves.json", {"points": curves})
        for index, run in enumerate(runs, 1):
            run_dir = output / "runs" / run["run_id"]
            agent = TabularAgent.load(output / run["model_path"])
            raw_rows = []
            for opponent in final["opponents"]:
                result = evaluate_policy(
                    agent, opponent, episodes=final["episodes"],
                    seed_start=final["env_seed_start"], policy_seed_start=final["policy_seed_start"],
                    algorithm=run["algorithm"], training_opponent=run["training_opponent"],
                    training_seed=run["training_seed"], stage="final",
                    checkpoint_episode=training["episodes"], reward_mode=training["reward_mode"],
                )
                raw_rows.extend(result["episodes"])
                run["final_evaluation"].append(result["summary"])
                print(f"FINAL {index}/{len(runs)} {run['run_id']} vs {opponent}: "
                      f"win_rate={result['summary']['win_rate']:.3f}", flush=True)
            all_final_rows.extend(raw_rows)
            write_episode_csv(run_dir / "final_evaluation_episodes.csv", raw_rows)
            write_json(run_dir / "final_evaluation.json", {"results": run["final_evaluation"]})
        baseline_rows, baseline_results = [], []
        for policy in config["baselines"]:
            for opponent in final["opponents"]:
                result = evaluate_policy(
                    policy, opponent, episodes=final["episodes"],
                    seed_start=final["env_seed_start"], policy_seed_start=final["policy_seed_start"],
                    stage="baseline", reward_mode=training["reward_mode"],
                )
                baseline_rows.extend(result["episodes"])
                baseline_results.append(result["summary"])
                print(f"BASELINE {policy} vs {opponent}: win_rate={result['summary']['win_rate']:.3f}", flush=True)
        write_episode_csv(output / "baseline_episodes.csv", baseline_rows)
        write_json(output / "baseline_results.json", {"settings": final, "results": baseline_results})
        summary = {
            "schema_version": 1, "status": "complete", "quick_integration_run": quick,
            "config": config, "training_runs": runs, "learning_curves": curves,
            "baseline_results": baseline_results,
            "condition_aggregates": _condition_aggregates(all_final_rows),
            "held_out_aggregates": _held_out_aggregates(runs),
            "counts": {
                "training_runs": len(runs),
                "training_episodes": len(runs) * training["episodes"],
                "training_environment_steps": sum(run["environment_steps"] for run in runs),
                "checkpoint_evaluation_episodes": sum(point["episodes"] for point in curves),
                "final_learned_evaluation_episodes": len(all_final_rows),
                "baseline_evaluation_episodes": len(baseline_rows),
                "invalid_evaluation_actions": sum(row["invalid_actions"] for row in all_final_rows + baseline_rows),
            },
        }
        demo = config.get("demo", {})
        if demo.get("enabled"):
            model_id = f"{demo['algorithm']}_{demo['training_opponent']}_seed{demo['training_seed']}"
            agent = TabularAgent.load(output / "runs" / model_id / "agent.npz")
            summary["demo"] = record_trained_demo(
                agent, output / "demo", opponent=demo["opponent"], seed=demo["seed"],
                reward_mode=training["reward_mode"],
                policy_seed=demo["policy_seed"], metadata={"run_id": model_id,
                    "training_episodes": training["episodes"], "epsilon": 0.0},
            )
        write_json(output / "final_summary.json", summary)
        manifest["status"] = "complete"
        manifest["finished_at_utc"] = _utc_now()
        manifest["elapsed_seconds"] = time.perf_counter() - started
        manifest["counts"] = summary["counts"]
        manifest["output_sha256"] = {
            str(path.relative_to(output)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file() and path.name != "manifest.json"
        }
        write_json(output / "manifest.json", manifest)
        print(f"COMPLETE: {output / 'final_summary.json'}", flush=True)
        return summary
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["finished_at_utc"] = _utc_now()
        manifest["error"] = f"{type(error).__name__}: {error}"
        write_json(output / "manifest.json", manifest)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/milestone3.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick", action="store_true", help="Use a separate, tiny integration run; excludes held-out bosses.")
    args = parser.parse_args()
    output = args.output or ("experiments/milestone3_quick" if args.quick else "experiments/milestone3")
    run_experiments(args.config, output, quick=args.quick)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
