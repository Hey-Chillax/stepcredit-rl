"""ALFWorld outcome reward: success from the rollout's runtime info, with every per-step info key carried over."""

import unittest

from recipes.alfworld.reward_fn import compute_score

STEP_KEYS = {"success", "num_steps", "is_action_valid", "is_noop", "num_noops", "task_id", "split", "task_family"}


class AlfworldRewardTest(unittest.TestCase):
    def score(self, success, **runtime):
        info = {"reward_extra_info": {"success": success, "num_steps": 7, "task_id": "trial_1", **runtime}}
        return compute_score("alfworld_valid_seen", "<tool_call>...</tool_call>", {"success": None}, info)

    def test_success_and_failure(self):
        self.assertEqual(self.score(True)["score"], 1.0)
        self.assertEqual(self.score(False)["score"], 0.0)

    def test_ground_truth_flag_wins_over_runtime_info(self):
        info = {"reward_extra_info": {"success": False}}
        self.assertEqual(compute_score("alfworld_train", "", {"success": True}, info)["score"], 1.0)

    def test_scoring_row_keeps_the_loop_fields(self):
        # The reward worker replaces the scoring row's info with this dict, so it must carry the flow's keys
        result = self.score(False, is_noop=True, num_noops=5)
        self.assertTrue(STEP_KEYS <= result.keys())
        self.assertEqual((result["is_noop"], result["num_noops"], result["num_steps"]), (True, 5, 7))

    def test_missing_loop_fields_default_to_zero(self):
        result = self.score(True)
        self.assertEqual((result["is_noop"], result["num_noops"]), (False, 0))


if __name__ == "__main__":
    unittest.main()
