"""Behavioral tests for the real combat simulator (no trained policy required).

Run from the project root: python -m unittest discover -s tests -v
"""

import unittest

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from read_the_boss import Action, Distance, Intent, ReadTheBossEnv
from read_the_boss.controllers import (
    HELD_OUT_OPPONENTS,
    TRAIN_OPPONENTS,
    choose_next_intent,
    controller_parameters,
)


class CombatRulesTests(unittest.TestCase):
    def setUp(self):
        self.env = ReadTheBossEnv()
        self.env.reset(seed=123)

    def tearDown(self):
        self.env.close()

    def start(self, hp=6, boss=8, stamina=4, distance=Distance.MEDIUM,
              intent=Intent.RECOVERY, turns=40):
        return self.env.reset(seed=123, options={
            "state": [hp, boss, stamina, int(distance), int(intent), turns]
        })

    def test_reset_restores_complete_initial_state(self):
        self.env.step(Action.APPROACH)
        obs, info = self.env.reset(seed=123)
        np.testing.assert_array_equal(obs, [6, 8, 4, 1, 3, 40])
        self.assertTrue(self.env.observation_space.contains(obs))
        self.assertEqual(info["outcome"], "ongoing")
        self.assertEqual(info["stamina_spent"], 0)
        self.assertEqual(info["damage_dealt"], 0)
        self.assertEqual(info["damage_taken"], 0)

    def test_observation_is_a_copy_not_mutable_environment_storage(self):
        obs, info = self.env.reset(seed=123)
        obs[0] = 0
        info["action_mask"][:] = 0
        self.assertEqual(self.env.state[0], 6)
        self.assertTrue(self.env.action_masks()[Action.REST])

    def test_action_mask_covers_range_boundaries_and_affordability(self):
        cases = [
            (Distance.CLOSE, 4, [0, 1, 1, 1, 1]),
            (Distance.MEDIUM, 4, [1, 1, 0, 1, 1]),
            (Distance.FAR, 4, [1, 0, 0, 1, 1]),
            (Distance.CLOSE, 0, [0, 0, 0, 0, 1]),
            (Distance.MEDIUM, 0, [1, 0, 0, 0, 1]),
            (Distance.CLOSE, 1, [0, 1, 0, 1, 1]),
        ]
        for distance, stamina, expected in cases:
            with self.subTest(distance=distance, stamina=stamina):
                _, info = self.start(stamina=stamina, distance=distance)
                np.testing.assert_array_equal(info["action_mask"], expected)
                np.testing.assert_array_equal(self.env.action_masks(), expected)

    def test_approach_resolves_slash_at_the_new_distance(self):
        self.start(intent=Intent.SLASH)
        obs, reward, terminated, truncated, info = self.env.step(Action.APPROACH)
        self.assertEqual(tuple(obs[[0, 2, 3, 5]]), (4, 4, 0, 39))
        self.assertEqual(info["damage_taken"], 2)
        self.assertAlmostEqual(reward, -0.1)
        self.assertFalse(terminated or truncated)

    def test_retreat_can_escape_a_close_slash(self):
        self.start(distance=Distance.CLOSE, intent=Intent.SLASH)
        obs, reward, _, _, info = self.env.step(Action.RETREAT)
        self.assertEqual(tuple(obs[[0, 2, 3]]), (6, 3, 1))
        self.assertEqual(info["stamina_spent"], 1)
        self.assertEqual(info["damage_taken"], 0)
        self.assertAlmostEqual(reward, -0.12)

    def test_retreat_from_medium_escapes_sweep(self):
        self.start(distance=Distance.MEDIUM, intent=Intent.SWEEP)
        obs, _, _, _, info = self.env.step(Action.RETREAT)
        self.assertEqual(tuple(obs[[0, 3]]), (6, 2))
        self.assertEqual(info["damage_taken"], 0)

    def test_retreat_from_close_does_not_escape_sweep(self):
        self.start(distance=Distance.CLOSE, intent=Intent.SWEEP)
        obs, _, _, _, info = self.env.step(Action.RETREAT)
        self.assertEqual(tuple(obs[[0, 3]]), (3, 1))
        self.assertEqual(info["damage_taken"], 3)

    def test_lunge_can_catch_a_retreating_player(self):
        self.start(distance=Distance.CLOSE, intent=Intent.LUNGE)
        obs, _, _, _, info = self.env.step(Action.RETREAT)
        self.assertEqual(tuple(obs[[0, 2, 3]]), (5, 3, 0))
        self.assertEqual(info["damage_taken"], 1)

    def test_lunge_from_far_closes_distance_without_hitting(self):
        self.start(distance=Distance.FAR, intent=Intent.LUNGE)
        obs, _, _, _, info = self.env.step(Action.REST)
        self.assertEqual(tuple(obs[[0, 3]]), (6, 1))
        self.assertEqual(info["damage_taken"], 0)

    def test_guard_fully_blocks_slash_and_lunge(self):
        for intent in (Intent.SLASH, Intent.LUNGE):
            with self.subTest(intent=intent):
                self.start(distance=Distance.CLOSE, intent=intent)
                obs, reward, _, _, info = self.env.step(Action.GUARD)
                self.assertEqual(tuple(obs[[0, 2]]), (6, 3))
                self.assertEqual(info["damage_taken"], 0)
                self.assertAlmostEqual(reward, -0.12)

    def test_guard_only_partially_blocks_heavy_sweep(self):
        self.start(distance=Distance.MEDIUM, intent=Intent.SWEEP)
        obs, reward, _, _, info = self.env.step(Action.GUARD)
        self.assertEqual(tuple(obs[[0, 2]]), (5, 3))
        self.assertEqual(info["damage_taken"], 1)
        self.assertAlmostEqual(reward, -0.12)

    def test_rest_caps_stamina_and_does_not_protect_health(self):
        self.start(stamina=3, distance=Distance.CLOSE, intent=Intent.SLASH)
        obs, reward, _, _, info = self.env.step(Action.REST)
        self.assertEqual(tuple(obs[[0, 2]]), (4, 4))
        self.assertEqual(info["stamina_spent"], 0)
        self.assertAlmostEqual(reward, -0.1)

    def test_nonlethal_strike_does_not_cancel_boss_attack(self):
        self.start(distance=Distance.CLOSE, intent=Intent.SWEEP)
        obs, reward, terminated, _, info = self.env.step(Action.STRIKE)
        self.assertEqual(tuple(obs[:3]), (3, 6, 2))
        self.assertEqual(info["damage_dealt"], 2)
        self.assertEqual(info["damage_taken"], 3)
        self.assertAlmostEqual(reward, -0.14)
        self.assertFalse(terminated)

    def test_lethal_strike_precedes_retaliation_and_deadline(self):
        self.start(hp=1, boss=2, stamina=2, distance=Distance.CLOSE,
                   intent=Intent.SWEEP, turns=1)
        obs, reward, terminated, truncated, info = self.env.step(Action.STRIKE)
        self.assertEqual(tuple(obs[:3]), (1, 0, 0))
        self.assertEqual(obs[5], 0)
        self.assertEqual(info["damage_taken"], 0)
        self.assertEqual(info["outcome"], "victory")
        self.assertAlmostEqual(reward, 100 + 10 / 6 - 0.14)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_death_precedes_timeout_and_health_is_clamped(self):
        self.start(hp=1, distance=Distance.CLOSE, intent=Intent.SWEEP, turns=1)
        obs, reward, terminated, truncated, info = self.env.step(Action.REST)
        self.assertEqual(obs[0], 0)
        self.assertEqual(info["outcome"], "death")
        self.assertEqual(info["damage_taken"], 1)
        self.assertAlmostEqual(reward, -100.1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_intrinsic_timeout_is_a_loss_and_termination(self):
        self.start(distance=Distance.FAR, turns=1)
        obs, reward, terminated, truncated, info = self.env.step(Action.REST)
        self.assertEqual(tuple(obs[[0, 1, 5]]), (6, 8, 0))
        self.assertEqual(info["outcome"], "timeout")
        self.assertAlmostEqual(reward, -100.1)
        self.assertTrue(terminated)
        self.assertFalse(truncated)

    def test_masked_strike_wastes_turn_without_spending_stamina(self):
        self.start(stamina=1, distance=Distance.CLOSE, intent=Intent.SLASH)
        obs, reward, _, _, info = self.env.step(Action.STRIKE)
        self.assertEqual(tuple(obs[[0, 1, 2, 5]]), (4, 8, 1, 39))
        self.assertTrue(info["invalid_action"])
        self.assertEqual(info["damage_dealt"], 0)
        self.assertEqual(info["stamina_spent"], 0)
        self.assertAlmostEqual(reward, -0.1)

    def test_out_of_range_strike_is_masked_and_cannot_damage_boss(self):
        self.start(distance=Distance.FAR)
        obs, reward, _, _, info = self.env.step(Action.STRIKE)
        self.assertEqual(tuple(obs[[1, 2, 3, 5]]), (8, 4, 2, 39))
        self.assertTrue(info["invalid_action"])
        self.assertAlmostEqual(reward, -0.1)

    def test_invalid_action_types_do_not_advance_state_or_rng(self):
        other = ReadTheBossEnv()
        other.reset(seed=123)
        for action in (-1, 5, 1.5, True, None, "2"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    self.env.step(action)
                self.assertEqual(self.env.state, (6, 8, 4, 1, 3, 40))
        actual = self.env.step(Action.APPROACH)
        expected = other.step(Action.APPROACH)
        np.testing.assert_array_equal(actual[0], expected[0])
        self.assertEqual(actual[1:4], expected[1:4])
        other.close()

    def test_step_requires_reset_and_refuses_finished_episode(self):
        fresh = ReadTheBossEnv()
        with self.assertRaises(RuntimeError):
            fresh.step(Action.REST)
        fresh.close()
        self.start(turns=1)
        self.env.step(Action.REST)
        with self.assertRaises(RuntimeError):
            self.env.step(Action.REST)

    def test_invalid_debug_states_are_rejected(self):
        invalid_states = [
            [0, 8, 4, 1, 3, 40], [6, 0, 4, 1, 3, 40],
            [7, 8, 4, 1, 3, 40], [6, 9, 4, 1, 3, 40],
            [6, 8, -1, 1, 3, 40], [6, 8, 5, 1, 3, 40],
            [6, 8, 4, 3, 3, 40], [6, 8, 4, 1, 4, 40],
            [6, 8, 4, 1, 3, 0], [6, 8, 4, 1, 3, 41],
            [6, 8, 4, 1, 3], [6.5, 8, 4, 1, 3, 40],
        ]
        for state in invalid_states:
            with self.subTest(state=state):
                with self.assertRaises(ValueError):
                    self.env.reset(options={"state": state})
        with self.assertRaises(ValueError):
            self.env.reset(options={"unknown_option": 1})

    def test_reward_ablation_removes_only_health_and_stamina_terms(self):
        alternative = ReadTheBossEnv(reward_mode="terminal_time")
        state = [6, 2, 2, 0, 1, 40]
        self.env.reset(seed=8, options={"state": state})
        alternative.reset(seed=8, options={"state": state})
        full = self.env.step(Action.STRIKE)
        ablated = alternative.step(Action.STRIKE)
        np.testing.assert_array_equal(full[0], ablated[0])
        self.assertAlmostEqual(full[1], 109.86)
        self.assertAlmostEqual(ablated[1], 99.9)
        self.assertEqual(full[2:4], ablated[2:4])
        alternative.close()


class StochasticControllerTests(unittest.TestCase):
    def test_all_controller_probabilities_are_valid(self):
        for name in tuple(TRAIN_OPPONENTS) + tuple(HELD_OUT_OPPONENTS):
            for distance in range(3):
                with self.subTest(controller=name, distance=distance):
                    pc, weights = controller_parameters(name, (6, 8, 4, distance, 0, 40))
                    self.assertGreaterEqual(pc, 0)
                    self.assertLessEqual(pc, 1)
                    self.assertEqual(np.asarray(weights).shape, (3,))
                    self.assertTrue(np.all(np.asarray(weights) > 0))
                    self.assertAlmostEqual(float(sum(weights)), 1.0)

    def test_enraged_probability_changes_at_observed_boss_health(self):
        normal, _ = controller_parameters("enraged", (6, 5, 4, 1, 0, 40))
        enraged, _ = controller_parameters("enraged", (6, 4, 4, 1, 0, 40))
        self.assertAlmostEqual(normal, 0.35)
        self.assertAlmostEqual(enraged, 0.85)

    def test_recovery_is_always_followed_by_an_offensive_intent(self):
        rng = np.random.default_rng(2026)
        for name in tuple(TRAIN_OPPONENTS) + tuple(HELD_OUT_OPPONENTS) + ("mixed",):
            with self.subTest(controller=name):
                samples = [choose_next_intent(name, (6, 8, 4, 1, 3, 40), rng)
                           for _ in range(200)]
                self.assertTrue(all(x in (0, 1, 2) for x in samples))

    def test_mixed_attack_probabilities_include_combo_weighting(self):
        # After an attack, controller choice and combo continuation are coupled.
        # Exact unconditional probabilities at close range, derived on paper:
        # slash=43/180; sweep=17/144; lunge=103/720; recovery=1/2.
        rng = np.random.default_rng(7001)
        state = (6, 8, 4, 0, 0, 40)
        samples = [choose_next_intent("mixed", state, rng) for _ in range(25000)]
        frequencies = np.bincount(samples, minlength=4) / len(samples)
        expected = [43 / 180, 17 / 144, 103 / 720, 0.5]
        np.testing.assert_allclose(frequencies, expected, atol=0.018, rtol=0)

    def test_mixed_after_recovery_uses_unweighted_controller_mixture(self):
        rng = np.random.default_rng(7002)
        state = (6, 8, 4, 0, 3, 40)
        samples = [choose_next_intent("mixed", state, rng) for _ in range(25000)]
        frequencies = np.bincount(samples, minlength=4) / len(samples)
        np.testing.assert_allclose(
            frequencies, [23 / 45, 41 / 180, 47 / 180, 0], atol=0.018, rtol=0
        )

    def test_distance_aware_far_range_prefers_lunging(self):
        rng = np.random.default_rng(7003)
        state = (6, 8, 4, 2, 3, 40)
        samples = [choose_next_intent("distance_aware", state, rng) for _ in range(10000)]
        frequencies = np.bincount(samples, minlength=4) / len(samples)
        np.testing.assert_allclose(frequencies, [0.15, 0.15, 0.7, 0], atol=0.025, rtol=0)


class EnvironmentIntegrationTests(unittest.TestCase):
    def test_gymnasium_environment_checker(self):
        env = gym.make("ReadTheBoss-v0")
        self.assertIsNone(env.spec.max_episode_steps)
        check_env(env.unwrapped)
        env.close()

    def test_seeded_trajectories_reproduce(self):
        for opponent in ("balanced", "mixed", "enraged"):
            with self.subTest(opponent=opponent):
                envs = [ReadTheBossEnv(opponent=opponent) for _ in range(2)]
                starts = [env.reset(seed=808) for env in envs]
                np.testing.assert_array_equal(starts[0][0], starts[1][0])
                selector = np.random.default_rng(919)
                for _ in range(40):
                    action = int(selector.choice(np.flatnonzero(envs[0].action_masks())))
                    a, b = [env.step(action) for env in envs]
                    np.testing.assert_array_equal(a[0], b[0])
                    self.assertEqual(a[1:4], b[1:4])
                    self.assertEqual(a[4]["events"], b[4]["events"])
                    if a[2] or a[3]:
                        break
                for env in envs:
                    env.close()

    def test_random_rollouts_preserve_invariants_and_finish(self):
        # 280 complete episodes, covering every implemented controller and mixed mode.
        opponents = tuple(TRAIN_OPPONENTS) + tuple(HELD_OUT_OPPONENTS) + ("mixed",)
        rng = np.random.default_rng(991)
        for opponent in opponents:
            env = ReadTheBossEnv(opponent=opponent)
            for seed in range(40):
                with self.subTest(opponent=opponent, seed=seed):
                    obs, info = env.reset(seed=seed)
                    total_reward = 0.0
                    for turn in range(1, 41):
                        action = int(rng.choice(np.flatnonzero(info["action_mask"])))
                        obs, reward, terminated, truncated, info = env.step(action)
                        total_reward += reward
                        self.assertTrue(env.observation_space.contains(obs))
                        self.assertEqual(obs[5], 40 - turn)
                        self.assertFalse(info["invalid_action"])
                        self.assertTrue(np.isfinite(reward))
                        self.assertAlmostEqual(sum(info["reward_terms"].values()), reward)
                        self.assertFalse(truncated)
                        if terminated:
                            self.assertIn(info["outcome"], ("victory", "death", "timeout"))
                            if info["outcome"] == "victory":
                                self.assertEqual(obs[1], 0)
                                self.assertGreater(obs[0], 0)
                                self.assertGreater(total_reward, 90)
                            else:
                                self.assertLess(total_reward, -100)
                            break
                        self.assertTrue(info["action_mask"][Action.REST])
                    else:
                        self.fail("Episode exceeded the 40-turn intrinsic horizon")
            env.close()

    def test_rendering_does_not_change_state_or_future_randomness(self):
        a = ReadTheBossEnv(opponent="mixed", render_mode="ansi")
        b = ReadTheBossEnv(opponent="mixed", render_mode="ansi")
        a.reset(seed=72)
        b.reset(seed=72)
        text = a.render()
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 0)
        a.render()
        self.assertEqual(a.state, b.state)
        x, y = a.step(Action.APPROACH), b.step(Action.APPROACH)
        np.testing.assert_array_equal(x[0], y[0])
        self.assertEqual(x[1:4], y[1:4])
        a.close()
        b.close()

    def test_rgb_render_returns_a_valid_image(self):
        env = ReadTheBossEnv(render_mode="rgb_array")
        env.reset(seed=5)
        frame = env.render()
        self.assertIsInstance(frame, np.ndarray)
        self.assertEqual(frame.ndim, 3)
        self.assertEqual(frame.shape[2], 3)
        self.assertEqual(frame.dtype, np.uint8)
        self.assertGreaterEqual(frame.shape[0], 200)
        self.assertGreaterEqual(frame.shape[1], 300)
        env.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
