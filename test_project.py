"""Fast sanity tests; run with `python -m unittest test_project.py`."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import torch
from waste_sorting_env import WasteSortingEnv
from ppo_numpy import LinearPPO

class ProjectTests(unittest.TestCase):
    def test_observation_and_valid_action_mask(self):
        env = WasteSortingEnv(stage=1, seed=1)
        obs = env.reset()
        self.assertEqual(obs.shape, (6,))
        p = LinearPPO(seed=1).probs(obs, valid_actions=env.n + 1)
        self.assertAlmostEqual(float(p.sum()), 1.0, places=7)
        self.assertEqual(float(p[-1]), 0.0)

    def test_reward_curriculum_targets(self):
        env1 = WasteSortingEnv(stage=1, seed=1)
        self.assertGreater(env1._empty_reward(env1.HIGH_PEAK), env1._empty_reward(env1.LOW_PEAK))
        env2 = WasteSortingEnv(stage=2, seed=1)
        self.assertGreater(env2._empty_reward(env2.HIGH_PEAK), 0)
        env3 = WasteSortingEnv(stage=3, seed=1)
        self.assertEqual(env3._empty_reward(env3.HIGH_PEAK), 1.0)
        self.assertEqual(env3._empty_reward(env3.LOW_PEAK), 1.0)


    def test_backprop_updates_network_weights(self):
        agent = LinearPPO(seed=11)
        env = WasteSortingEnv(stage=1, seed=11, horizon=12)
        before = [p.detach().clone() for p in agent.model.parameters()]
        obs, done, trajectory = env.reset(), False, []
        while not done:
            action, logp, value = agent.act(obs, valid_actions=env.n + 1)
            nxt, reward, done, _ = env.step(action)
            trajectory.append(dict(obs=obs.copy(), action=action, logp=logp, value=value,
                                   reward=reward, done=done,
                                   action_mask=np.arange(4) < env.n + 1))
            obs = nxt
        returns, advantages = agent._returns_and_advantages(
            [x['reward'] for x in trajectory], [x['value'] for x in trajectory],
            [x['done'] for x in trajectory])
        for i, tr in enumerate(trajectory):
            tr['return'], tr['advantage'] = returns[i], advantages[i]
        metrics = agent.update(trajectory, epochs=2, minibatch_size=8)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(before, agent.model.parameters())))
        self.assertIn('policy_loss', metrics)

    def test_checkpoint_roundtrip(self):
        agent = LinearPPO(seed=5)
        obs = np.zeros(6, dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'agent.pt'
            expected = agent.probs(obs)
            agent.save(path)
            restored = LinearPPO(seed=99)
            restored.load(path)
            np.testing.assert_allclose(expected, restored.probs(obs), atol=1e-7)


    def test_phase_three_is_strict_step_reward_and_noop_is_zero(self):
        env = WasteSortingEnv(stage=3, seed=3)
        self.assertEqual(env._empty_reward(0.20), 0.0)
        self.assertEqual(env._empty_reward(env.LOW_PEAK + env.PEAK_TOLERANCE + 0.01), 0.0)
        obs = env.reset()
        _, reward, _, _ = env.step(0)
        self.assertEqual(reward, 0.0)

    def test_busy_press_rejects_empty_request(self):
        env = WasteSortingEnv(stage=3, seed=9)
        env.busy = 2
        before = env.stats['throughput']
        _, reward, _, _ = env.step(1)
        self.assertEqual(env.stats['throughput'], before)
        self.assertLess(reward, 0.0)

    def test_environment_episode_terminates(self):
        env = WasteSortingEnv(stage=3, seed=2, horizon=5)
        obs = env.reset(); done = False; count = 0
        while not done:
            obs, reward, done, stats = env.step(0); count += 1
        self.assertLessEqual(count, 5)
        self.assertIn('throughput', stats)

if __name__ == '__main__':
    unittest.main()

class CollisionModelTests(unittest.TestCase):
    def test_monte_carlo_data_shape_and_binary_labels(self):
        from collision_model import generate_pairwise_data
        X, y = generate_pairwise_data(n_samples=200, seed=4)
        self.assertEqual(X.shape, (200, 7))
        self.assertEqual(set(np.unique(y)).issubset({0, 1}), True)

    def test_override_only_considered_for_noop(self):
        from collision_model import pairwise_override
        class DummyModel:
            def predict_proba(self, X):
                return np.array([[0.01, 0.99]])
        env = WasteSortingEnv(stage=3, seed=1)
        env.levels[:] = [0.70, 0.72, 0.20]
        env.busy = 2
        action, risk = pairwise_override(env, 2, DummyModel())
        self.assertEqual(action, 2)
        action, risk = pairwise_override(env, 0, DummyModel())
        self.assertEqual(action, 0)  # cannot empty while PU is busy
        env.busy = 0
        action, risk = pairwise_override(env, 0, DummyModel())
        self.assertIn(action, (1, 2))  # high-risk no-op is overridden while idle


class PlayableGameTests(unittest.TestCase):
    def test_manual_game_module_is_importable(self):
        from play_game import run_game
        self.assertTrue(callable(run_game))

    def test_environment_exposes_game_actions_and_state(self):
        env = WasteSortingEnv(stage=3, seed=12, n_containers=3, horizon=4)
        obs = env.reset()
        self.assertEqual(obs.shape, (6,))
        self.assertEqual(env.n + 1, 4)  # no-op plus one action per container
        self.assertTrue(np.all(env.levels >= 0))
        self.assertTrue(np.all(env.levels <= env.CAPACITY))
