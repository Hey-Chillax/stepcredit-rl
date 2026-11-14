"""Step-level advantage with the loop-aware term.

Batch used below (rows are agent steps, 3 response tokens each, the outcome reward on the last token of the last step):
    prompt p0: trajectories a and b both fail; a repeats a no-op at its steps 1 and 2, b never does.
    prompt p1: trajectory c succeeds in 2 steps, d fails in 2 steps; no no-ops.
Anchor states: a = (s0, s1, s2), b = (s0, s1, s3), c = (t0, t1), d = (t0, t2).
"""

import importlib.util
import unittest

HAS_DEPS = all(importlib.util.find_spec(m) is not None for m in ("torch", "verl"))

ROWS = [  # (prompt uid, trajectory uid, step index, anchor state, outcome reward, loop flag)
    ("p0", "a", 0, "s0", 0.0, 0.0),
    ("p0", "a", 1, "s1", 0.0, 1.0),
    ("p0", "a", 2, "s2", 0.0, 1.0),
    ("p0", "b", 0, "s0", 0.0, 0.0),
    ("p0", "b", 1, "s1", 0.0, 0.0),
    ("p0", "b", 2, "s3", 0.0, 0.0),
    ("p1", "c", 0, "t0", 0.0, 0.0),
    ("p1", "c", 1, "t1", 1.0, 0.0),
    ("p1", "d", 0, "t0", 0.0, 0.0),
    ("p1", "d", 1, "t2", 0.0, 0.0),
]
PENALTY = 0.05


@unittest.skipUnless(HAS_DEPS, "needs torch and verl")
class StepAdvantageTest(unittest.TestCase):
    def setUp(self):
        import numpy as np
        import torch

        from agent_r1.trainer.ppo import core_algos

        self.np, self.torch, self.core = np, torch, core_algos
        n, length = len(ROWS), 3
        # (bs, response_length): bs counts step rows; response_mask is 1 on generated tokens (all 3 here)
        self.response_mask = torch.ones(n, length)
        self.token_rewards = torch.zeros(n, length)
        self.token_rewards[:, -1] = torch.tensor([r[4] for r in ROWS])
        self.index = np.array([r[0] for r in ROWS], dtype=object)
        self.traj = np.array([r[1] for r in ROWS], dtype=object)
        self.steps = np.array([r[2] for r in ROWS])
        self.anchor = np.array([r[3] for r in ROWS], dtype=object)
        self.flags = np.array([r[5] for r in ROWS], dtype=np.float32)

    def advantages(self, loop_returns=None, metrics=None):
        step_rewards = self.core.compute_step_discounted_returns(
            self.token_rewards, self.response_mask, self.traj, self.steps, gamma=1.0
        )
        adv, _ = self.core.compute_gigpo_outcome_advantage(
            token_level_rewards=self.token_rewards,
            step_rewards=step_rewards,
            response_mask=self.response_mask,
            anchor_obs=self.anchor,
            index=self.index,
            trajectory_uids=self.traj,
            loop_returns=loop_returns,
            metrics=metrics,
        )
        return adv[:, 0]  # one advantage per step row, broadcast over its response tokens

    def loop_returns(self):
        flags = self.torch.as_tensor(self.flags)
        return PENALTY * self.core.compute_step_suffix_sums(flags, self.traj, self.steps, gamma=1.0)

    def test_suffix_sums_match_a_reference_loop(self):
        rng = self.np.random.default_rng(0)
        traj, steps, values = [], [], []
        for t, length in enumerate([3, 1, 5, 2]):
            for s in range(length):
                traj.append(f"t{t}")
                steps.append(s)
                values.append(float(rng.integers(0, 3)))
        order = rng.permutation(len(values))  # rows of a batch are not sorted by trajectory
        traj, steps, values = [self.np.array(x)[order] for x in (traj, steps, values)]
        got = self.core.compute_step_suffix_sums(self.torch.tensor(values), traj, steps, gamma=0.9)
        for i in range(len(values)):
            expected = sum(
                0.9 ** (steps[j] - steps[i]) * values[j]
                for j in range(len(values))
                if traj[j] == traj[i] and steps[j] >= steps[i]
            )
            self.assertAlmostEqual(float(got[i]), expected, places=5)

    def test_zero_penalty_leaves_the_advantage_unchanged(self):
        plain = self.advantages()
        zero = self.advantages(loop_returns=self.torch.zeros(len(ROWS)))
        self.assertTrue(self.torch.allclose(plain, zero))

    def test_all_failed_group_gets_a_centred_loop_signal(self):
        plain = self.advantages()
        self.assertTrue(self.torch.all(plain[:6] == 0))  # p0: every rollout failed, so no signal at all
        delta = self.advantages(loop_returns=self.loop_returns()) - plain
        # a pays 0.10 from steps 0 and 1 and 0.05 from step 2; b pays nothing. Shared anchors s0 and s1 are centred
        # (+-0.05, not divided by the group std); s2 and s3 are singletons and stay 0.
        expected = self.torch.tensor([-0.05, -0.05, 0.0, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(self.torch.allclose(delta, expected, atol=1e-6))

    def test_outcome_signal_is_untouched(self):
        plain = self.advantages()
        with_loop = self.advantages(loop_returns=self.loop_returns())
        self.assertTrue(self.torch.allclose(plain[6:], with_loop[6:]))  # p1 has no no-ops
        self.assertGreater(float(plain[6]), 1.0)  # c0: trajectory and step level both favour success

    def test_metrics(self):
        metrics = {}
        self.advantages(loop_returns=self.loop_returns(), metrics=metrics)
        self.assertAlmostEqual(metrics["agent/loop/step_signal_ratio_plain"], 0.2)  # c0, d0
        self.assertAlmostEqual(metrics["agent/loop/signal_row_ratio"], 0.4)  # a0, a1, b0, b1
        self.assertAlmostEqual(metrics["agent/loop/step_signal_ratio"], 0.6)

    def test_compute_advantage_end_to_end(self):
        from verl import DataProto

        from agent_r1.trainer.ppo.core_algos import AgentAdvantageEstimator
        from agent_r1.trainer.ppo.ray_trainer import compute_advantage

        def batch(with_flags: bool):
            non_tensors = {"uid": self.index, "trajectory_uids": self.traj, "step_indices": self.steps}
            non_tensors["anchor_obs"] = self.anchor
            if with_flags:
                non_tensors["loop_flag"] = self.flags.astype(object)
            return DataProto.from_dict(
                tensors={"token_level_rewards": self.token_rewards.clone(), "response_mask": self.response_mask},
                non_tensors=non_tensors,
            )

        config = {"gigpo": {"loop_penalty": PENALTY}}
        out = compute_advantage(batch(True), AgentAdvantageEstimator.GIGPO, gamma=1.0, config=config)
        expected = self.advantages(loop_returns=self.loop_returns())
        self.assertTrue(self.torch.allclose(out.batch["advantages"][:, 0], expected, atol=1e-6))
        self.assertAlmostEqual(out.meta_info["advantage_metrics"]["agent/loop/flag_ratio"], 0.2)
        with self.assertRaises(KeyError):
            compute_advantage(batch(False), AgentAdvantageEstimator.GIGPO, gamma=1.0, config=config)


if __name__ == "__main__":
    unittest.main()
