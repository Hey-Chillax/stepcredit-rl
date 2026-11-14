"""HotpotQA reward: exact match, F1 partial credit, evidence credit and the capped repeated-query penalty."""

import importlib.util
import unittest
from types import SimpleNamespace

from recipes.hotpotqa.reward_fn import (
    _answer_f1,
    _answer_in_passages,
    _normalize_answer,
    compute_score,
    trajectory_stats,
)

HAS_VERL = importlib.util.find_spec("verl") is not None
SOURCE = "hotpotqa_distractor"


class ExactMatchTest(unittest.TestCase):
    def test_normalization(self):
        self.assertEqual(_normalize_answer("The  Eiffel Tower!"), "eiffel tower")

    def test_tagged_answer(self):
        self.assertEqual(compute_score(SOURCE, "<answer>the Eiffel tower</answer>", "Eiffel Tower"), 1.0)
        self.assertEqual(compute_score(SOURCE, "<answer>Louvre</answer>", "Eiffel Tower"), 0.0)

    def test_last_tag_and_aliases(self):
        solution = "<answer>Paris</answer> on second thought <answer>NYC</answer>"
        self.assertEqual(compute_score(SOURCE, solution, "New York City", {"answers": ["NYC"]}), 1.0)

    def test_missing_ground_truth_scores_zero(self):
        self.assertEqual(compute_score(SOURCE, "<answer>x</answer>", ""), 0.0)


class TrajectoryStatsTest(unittest.TestCase):
    def test_token_f1(self):
        self.assertAlmostEqual(_answer_f1("New York City", ["New York"]), 0.8)
        self.assertEqual(_answer_f1("Boston", ["New York"]), 0.0)
        self.assertEqual(_answer_f1("the city", ["New York", "the city"]), 1.0)

    def test_evidence_hit(self):
        self.assertTrue(_answer_in_passages(["The tower is in Paris, France."], ["Paris"]))
        self.assertFalse(_answer_in_passages(["Parisian cafes"], ["Paris"]))  # whole words only

    def test_yes_no_never_counts_as_evidence(self):
        self.assertFalse(_answer_in_passages(["Yes, it is older than the bridge."], ["yes"]))

    def test_stats_of_a_wrong_tagged_answer(self):
        stats = trajectory_stats(
            SOURCE,
            "<answer>New York City</answer>",
            "New York",
            None,
            passages=["He moved to New York in 1990."],
            queries=["new york", "New York!", "brooklyn bridge"],
        )
        self.assertEqual(stats["acc"], 0.0)
        self.assertEqual(stats["answered"], 1.0)
        self.assertAlmostEqual(stats["f1"], 0.8)
        self.assertEqual(stats["evidence_hit"], 1.0)
        self.assertEqual(stats["repeat_queries"], 1.0)  # the first two queries normalize to the same string

    def test_untagged_answer_gets_no_f1(self):
        stats = trajectory_stats(SOURCE, "New York City", "New York", None, passages=[], queries=[])
        self.assertEqual(stats["answered"], 0.0)
        self.assertEqual(stats["f1"], 0.0)


@unittest.skipUnless(HAS_VERL, "needs verl: the shaped reward lives in the agent flow")
class ShapedRewardTest(unittest.TestCase):
    """The setting of the shaped-reward run: F1 weight 0.3, evidence weight 0.1, penalty 0.05 capped at 4 queries."""

    def setUp(self):
        from recipes.hotpotqa.hotpotqa_agent_flow import HotpotQAAgentFlow

        self.shaped_reward = HotpotQAAgentFlow._shaped_reward
        self.flow = SimpleNamespace(f1_weight=0.3, evidence_weight=0.1, repeat_penalty=0.05, repeat_penalty_cap=4)

    def reward(self, **stats):
        base = {"acc": 0.0, "answered": 0.0, "f1": 0.0, "evidence_hit": 0.0, "repeat_queries": 0.0}
        return self.shaped_reward(self.flow, {**base, **stats})

    def test_exact_match_scores_one(self):
        self.assertAlmostEqual(self.reward(acc=1.0, answered=1.0, f1=1.0, evidence_hit=1.0), 1.0)

    def test_partial_credit_stays_below_exact_match(self):
        self.assertAlmostEqual(self.reward(answered=1.0, evidence_hit=1.0), 0.1)
        self.assertAlmostEqual(self.reward(answered=1.0, f1=0.6, evidence_hit=1.0), 0.28)
        self.assertAlmostEqual(self.reward(answered=1.0, f1=1.0, evidence_hit=1.0), 0.4)

    def test_untagged_answer_earns_nothing(self):
        self.assertEqual(self.reward(f1=0.9, evidence_hit=1.0), 0.0)

    def test_repeat_penalty_is_capped(self):
        self.assertAlmostEqual(self.reward(repeat_queries=4.0), -0.2)  # the same query five times
        self.assertAlmostEqual(self.reward(repeat_queries=9.0), -0.2)


if __name__ == "__main__":
    unittest.main()
