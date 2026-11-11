"""Numerical and integration tests for learning; no win-rate threshold is assumed.

The short runs verify update semantics, legal actions, reproducibility, and
artifact round trips. They are not intended as learning-performance benchmarks.
"""

import copy
import csv
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from read_the_boss import Action, ReadTheBossEnv
from read_the_boss.agents import TabularAgent
from read_the_boss.evaluation import evaluate_policy
from read_the_boss.training import TrainingConfig, train


START = np.array([6, 8, 4, 1, 3, 40], dtype=np.int64)
NEXT = np.array([6, 8, 4, 0, 0, 39], dtype=np.int64)
START_MASK = np.array([1, 1, 0, 1, 1], dtype=np.int8)
NEXT_MASK = np.array([0, 1, 1, 1, 1], dtype=np.int8)


def numerical_agent(algorithm):
    """A hand-worked update fixture with an attractive but illegal next action."""
    agent = TabularAgent(algorithm=algorithm, alpha=0.25, gamma=0.9)
    agent.q_values[tuple(START)][Action.APPROACH] = 2.0
    agent.q_values[tuple(NEXT)] = [1000.0, -10.0, 7.0, -10.0, 3.0]
    return agent


class TemporalDifferenceTests(unittest.TestCase):
    def test_q_learning_uses_best_legal_next_value(self):
        agent = numerical_agent("q_learning")
        td_error = agent.update(
            START, Action.APPROACH, -1.0, NEXT, NEXT_MASK, terminated=False
        )
        # Target=-1+.9*7=5.3, TD error=3.3, new Q=2+.25*3.3=2.825.
        self.assertAlmostEqual(td_error, 3.3)
        self.assertAlmostEqual(agent.q_values[tuple(START)][Action.APPROACH], 2.825)
        self.assertEqual(agent.visit_counts[tuple(START)][Action.APPROACH], 1)
        self.assertEqual(agent.updates, 1)

    def test_sarsa_uses_selected_action_instead_of_maximum(self):
        agent = numerical_agent("sarsa")
        td_error = agent.update(
            START, Action.APPROACH, -1.0, NEXT, NEXT_MASK,
            terminated=False, next_action=Action.REST,
        )
        # The selected REST value is 3, although legal STRIKE has the higher value 7.
        # Target=-1+.9*3=1.7, TD error=-.3, new Q=1.925.
        self.assertAlmostEqual(td_error, -0.3)
        self.assertAlmostEqual(agent.q_values[tuple(START)][Action.APPROACH], 1.925)
        self.assertEqual(agent.visit_counts[tuple(START)][Action.APPROACH], 1)

    def test_sarsa_rejects_a_masked_bootstrap_action(self):
        agent = numerical_agent("sarsa")
        with self.assertRaises(ValueError):
            agent.update(
                START, Action.APPROACH, -1.0, NEXT, NEXT_MASK,
                terminated=False, next_action=Action.APPROACH,
            )
        self.assertEqual(agent.q_values[tuple(START)][Action.APPROACH], 2.0)
        self.assertEqual(agent.updates, 0)

    def test_all_real_terminal_outcomes_suppress_bootstrap(self):
        fixtures = [
            ("victory", [6, 2, 2, 0, 1, 1], Action.STRIKE),
            ("death", [1, 8, 4, 0, 1, 1], Action.REST),
            ("timeout", [6, 8, 4, 2, 3, 1], Action.REST),
        ]
        for algorithm in ("q_learning", "sarsa"):
            for expected_outcome, state, action in fixtures:
                with self.subTest(algorithm=algorithm, outcome=expected_outcome):
                    env = ReadTheBossEnv()
                    obs, _ = env.reset(seed=5, options={"state": state})
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    self.assertTrue(terminated)
                    self.assertFalse(truncated)
                    self.assertEqual(info["outcome"], expected_outcome)
                    self.assertFalse(np.any(info["action_mask"]))
                    agent = TabularAgent(algorithm=algorithm, alpha=0.25, gamma=0.9)
                    agent.q_values[tuple(obs)][action] = 2.0
                    agent.q_values[tuple(next_obs)] = 10000.0
                    td_error = agent.update(
                        obs, action, reward, next_obs, info["action_mask"],
                        terminated=terminated,
                    )
                    self.assertAlmostEqual(td_error, reward - 2.0)
                    self.assertAlmostEqual(
                        agent.q_values[tuple(obs)][action],
                        2.0 + 0.25 * (reward - 2.0),
                    )
                    env.close()

    def test_update_changes_only_the_visited_state_action_entry(self):
        agent = numerical_agent("q_learning")
        before = agent.q_values.copy()
        agent.update(START, Action.APPROACH, -1.0, NEXT, NEXT_MASK, terminated=False)
        changed = np.argwhere(agent.q_values != before)
        np.testing.assert_array_equal(changed, [tuple(START) + (int(Action.APPROACH),)])
        self.assertEqual(np.count_nonzero(agent.visit_counts), 1)


class ActionSelectionTests(unittest.TestCase):
    def test_greedy_action_masks_even_when_all_legal_values_are_negative(self):
        agent = TabularAgent()
        agent.q_values[tuple(START)] = [-7.0, -3.0, 1000000.0, -5.0, -4.0]
        rng = np.random.default_rng(17)
        for _ in range(20):
            self.assertEqual(agent.act(START, START_MASK, rng, epsilon=0.0), Action.RETREAT)
        self.assertEqual(agent.updates, 0)
        self.assertEqual(np.count_nonzero(agent.visit_counts), 0)

    def test_exploration_samples_only_legal_actions(self):
        agent = TabularAgent()
        agent.q_values[tuple(START)] = [0.0, 0.0, 1000000.0, 0.0, 0.0]
        rng = np.random.default_rng(23)
        seen = {agent.act(START, START_MASK, rng, epsilon=1.0) for _ in range(400)}
        self.assertEqual(seen, {0, 1, 3, 4})
        one_legal = np.array([0, 0, 0, 0, 1], dtype=np.int8)
        for epsilon in (0.0, 0.5, 1.0):
            for _ in range(20):
                self.assertEqual(agent.act(START, one_legal, rng, epsilon=epsilon), Action.REST)

    def test_greedy_ties_never_select_a_lower_value_or_masked_action(self):
        agent = TabularAgent()
        agent.q_values[tuple(START)] = [5.0, 4.0, 999.0, 5.0, 0.0]
        rng = np.random.default_rng(29)
        seen = {agent.act(START, START_MASK, rng, epsilon=0.0) for _ in range(200)}
        self.assertTrue(seen)
        self.assertTrue(seen.issubset({int(Action.APPROACH), int(Action.GUARD)}))

    def test_selection_rejects_empty_legal_mask(self):
        agent = TabularAgent()
        with self.assertRaises(ValueError):
            agent.act(START, np.zeros(5, dtype=np.int8), np.random.default_rng(0))


class PersistenceAndEvaluationTests(unittest.TestCase):
    def test_checkpoint_preserves_values_counts_configuration_and_metadata(self):
        agent = numerical_agent("sarsa")
        agent.update(
            START, Action.APPROACH, -1.0, NEXT, NEXT_MASK,
            terminated=False, next_action=Action.REST,
        )
        metadata = {"purpose": "roundtrip-test", "training_seed": 17}
        with tempfile.TemporaryDirectory() as temp:
            path = agent.save(Path(temp) / "agent.npz", metadata=metadata)
            loaded = TabularAgent.load(path, env=ReadTheBossEnv())
        self.assertEqual(loaded.algorithm, "sarsa")
        self.assertEqual(loaded.alpha, 0.25)
        self.assertEqual(loaded.gamma, 0.9)
        self.assertEqual(loaded.updates, agent.updates)
        self.assertEqual(loaded.metadata["purpose"], metadata["purpose"])
        self.assertEqual(loaded.metadata["training_seed"], 17)
        np.testing.assert_array_equal(loaded.q_values, agent.q_values)
        np.testing.assert_array_equal(loaded.visit_counts, agent.visit_counts)
        for state, mask in ((START, START_MASK), (NEXT, NEXT_MASK)):
            a = agent.act(state, mask, np.random.default_rng(3), epsilon=0.0)
            b = loaded.act(state, mask, np.random.default_rng(3), epsilon=0.0)
            self.assertEqual(a, b)

    def test_greedy_evaluation_is_reproducible_and_does_not_train(self):
        agent = numerical_agent("q_learning")
        values, visits = agent.q_values.copy(), agent.visit_counts.copy()
        updates = agent.updates
        metadata = copy.deepcopy(agent.metadata)
        kwargs = dict(episodes=12, seed_start=910000, policy_seed_start=920000)
        first = evaluate_policy(agent, "balanced", **kwargs)
        second = evaluate_policy(agent, "balanced", **kwargs)
        self.assertEqual(first["episodes"], second["episodes"])
        self.assertEqual(first["summary"], second["summary"])
        np.testing.assert_array_equal(agent.q_values, values)
        np.testing.assert_array_equal(agent.visit_counts, visits)
        self.assertEqual(agent.updates, updates)
        self.assertEqual(agent.metadata, metadata)
        self.assertEqual(first["summary"]["episodes"], 12)
        self.assertEqual(first["summary"]["invalid_actions"], 0)
        summary = first["summary"]
        self.assertEqual(summary["wins"] + summary["deaths"] + summary["timeouts"], 12)
        self.assertLessEqual(summary["total_environment_steps"], 12 * 40)


class TrainingIntegrationTests(unittest.TestCase):
    def test_short_training_and_evaluation_are_complete_and_legal(self):
        with tempfile.TemporaryDirectory() as temp:
            for algorithm in ("q_learning", "sarsa"):
                with self.subTest(algorithm=algorithm):
                    config = TrainingConfig(
                        algorithm=algorithm, opponent="mixed", episodes=12, seed=9,
                        alpha=0.2, gamma=1.0, checkpoint_every=0,
                    )
                    result = train(config, Path(temp) / algorithm)
                    agent = TabularAgent.load(result["agent_path"])
                    with Path(result["episodes_csv"]).open(newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 12)
                    self.assertTrue(all(int(row["invalid_actions"]) == 0 for row in rows))
                    self.assertTrue(all(row["outcome"] in ("victory", "death", "timeout")
                                        for row in rows))
                    self.assertGreater(agent.updates, 0)
                    self.assertTrue(np.all(np.isfinite(agent.q_values)))
                    evaluation = evaluate_policy(
                        agent, "balanced", episodes=8,
                        seed_start=930000, policy_seed_start=940000,
                    )
                    summary = evaluation["summary"]
                    self.assertEqual(summary["episodes"], 8)
                    self.assertEqual(summary["invalid_actions"], 0)
                    self.assertEqual(summary["wins"] + summary["deaths"] + summary["timeouts"], 8)

    def test_same_seed_reproduces_training_table_and_episode_log(self):
        with tempfile.TemporaryDirectory() as temp:
            config = TrainingConfig(algorithm="q_learning", episodes=14, seed=41,
                                    opponent="mixed", checkpoint_every=0)
            a = train(config, Path(temp) / "a")
            b = train(config, Path(temp) / "b")
            first, second = TabularAgent.load(a["agent_path"]), TabularAgent.load(b["agent_path"])
            np.testing.assert_array_equal(first.q_values, second.q_values)
            np.testing.assert_array_equal(first.visit_counts, second.visit_counts)
            self.assertEqual(first.updates, second.updates)
            self.assertEqual(Path(a["episodes_csv"]).read_bytes(), Path(b["episodes_csv"]).read_bytes())

    def test_evaluation_frequency_does_not_change_learning(self):
        def evaluate_checkpoint(agent, episode, environment_steps, epsilon):
            del episode, environment_steps, epsilon
            evaluate_policy(agent, "balanced", episodes=3,
                            seed_start=950000, policy_seed_start=960000)

        with tempfile.TemporaryDirectory() as temp:
            for algorithm in ("q_learning", "sarsa"):
                with self.subTest(algorithm=algorithm):
                    results = []
                    for every in (3, 5):
                        config = TrainingConfig(
                            algorithm=algorithm, episodes=12, seed=51,
                            opponent="mixed", checkpoint_every=every,
                        )
                        results.append(train(config, Path(temp) / f"{algorithm}-{every}",
                                             checkpoint_callback=evaluate_checkpoint))
                    a, b = [TabularAgent.load(r["agent_path"]) for r in results]
                    np.testing.assert_array_equal(a.q_values, b.q_values)
                    np.testing.assert_array_equal(a.visit_counts, b.visit_counts)
                    self.assertEqual(a.updates, b.updates)
                    self.assertEqual(Path(results[0]["episodes_csv"]).read_bytes(),
                                     Path(results[1]["episodes_csv"]).read_bytes())

    def test_sarsa_carries_the_selected_next_action_into_the_next_step(self):
        # Spy on the actual training loop rather than a second imitation of SARSA.
        # Each nonterminal update promises an action for the following env.step.
        actual_update = TabularAgent.update
        actual_act = TabularAgent.act
        observed = {"pending": None, "steps": 0, "choices": 0, "checked": 0}

        class TrackingEnvironment(ReadTheBossEnv):
            def step(self, action):
                if observed["pending"] is not None:
                    self_test.assertEqual(int(action), int(observed["pending"]))
                    observed["checked"] += 1
                    observed["pending"] = None
                self_test.assertTrue(self.action_masks()[action])
                observed["steps"] += 1
                return super().step(action)

        def tracking_act(agent, *args, **kwargs):
            observed["choices"] += 1
            return actual_act(agent, *args, **kwargs)

        def tracking_update(agent, obs, action, reward, next_obs, next_mask,
                            terminated, next_action=None):
            if not terminated:
                self.assertIsNotNone(next_action)
                observed["pending"] = next_action
            return actual_update(agent, obs, action, reward, next_obs, next_mask,
                                 terminated=terminated, next_action=next_action)

        self_test = self
        config = TrainingConfig(
            algorithm="sarsa", opponent="mixed", episodes=7, seed=61,
            epsilon_start=1.0, epsilon_end=1.0, checkpoint_every=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            with patch("read_the_boss.training.ReadTheBossEnv", TrackingEnvironment), \
                    patch.object(TabularAgent, "act", tracking_act), \
                    patch.object(TabularAgent, "update", tracking_update):
                train(config, Path(temp))
        self.assertEqual(observed["choices"], observed["steps"])
        self.assertEqual(observed["checked"], observed["steps"] - 7)
        self.assertIsNone(observed["pending"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
