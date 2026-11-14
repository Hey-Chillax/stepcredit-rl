"""Paired evaluation: exact McNemar test, item pairing across two validation dumps, and the per-split summary."""

import json
import tempfile
import unittest
from pathlib import Path

from tools.eval_summary import summarize
from tools.paired_eval import compare
from tools.validation_dump import load_entries, load_items, mcnemar_exact


def alfworld_entry(task_id: str, split: str, family: str, success: bool) -> dict:
    steps = [
        {"step_index": 0, "score": 0.0, "task_id": task_id, "split": split, "task_family": family, "success": False},
        {"step_index": 1, "score": float(success), "task_id": task_id, "split": split, "task_family": family,
         "success": success},
    ]
    return {"trajectory_uid": f"u-{task_id}", "input": f"prompt of {task_id}", "score": float(success),
            "num_steps": 2, "steps": steps}


def hotpotqa_entry(question: str, acc: float, shaped_score: float) -> dict:
    # A shaped-reward run logs exact match as `acc`; its score is the training reward, not the evaluation metric
    step = {"step_index": 0, "score": shaped_score, "acc": acc, "f1": 0.5, "repeat_queries": 0.0}
    return {"trajectory_uid": f"u-{question}", "input": question, "score": shaped_score, "num_steps": 1,
            "steps": [step]}


class McNemarTest(unittest.TestCase):
    def test_reported_tests(self):
        self.assertAlmostEqual(mcnemar_exact(26, 9), 0.006, places=3)  # ALFWorld valid_seen, step 200
        self.assertAlmostEqual(mcnemar_exact(21, 23), 0.88, places=2)  # ALFWorld valid_unseen, step 200
        self.assertAlmostEqual(mcnemar_exact(49, 35), 0.156, places=3)  # HotpotQA, 1,000 questions

    def test_symmetric_and_bounded(self):
        self.assertEqual(mcnemar_exact(3, 7), mcnemar_exact(7, 3))
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertEqual(mcnemar_exact(5, 5), 1.0)


class PairingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name: str, entries: list[dict]) -> Path:
        path = self.dir / name
        path.write_text("".join(json.dumps(e) + "\n" for e in entries), encoding="utf-8")
        return path

    def test_alfworld_pairs_by_task_inside_each_split(self):
        a = self.write("a.jsonl", [
            alfworld_entry("t1", "valid_seen", "pick_and_place", True),
            alfworld_entry("t2", "valid_seen", "pick_two_obj_and_place", True),
            alfworld_entry("t3", "valid_unseen", "pick_and_place", False),
        ])
        b = self.write("b.jsonl", [  # same items, different order
            alfworld_entry("t3", "valid_unseen", "pick_and_place", True),
            alfworld_entry("t2", "valid_seen", "pick_two_obj_and_place", False),
            alfworld_entry("t1", "valid_seen", "pick_and_place", True),
        ])
        rows = {r["split"]: r for r in compare(load_items(a), load_items(b))}
        self.assertEqual((rows["valid_seen"]["both_correct"], rows["valid_seen"]["only_a"]), (1, 1))
        self.assertEqual((rows["valid_unseen"]["only_a"], rows["valid_unseen"]["only_b"]), (0, 1))
        self.assertEqual(rows["valid_seen"]["items"], 2)

    def test_hotpotqa_uses_exact_match_not_the_training_reward(self):
        a = self.write("em.jsonl", [hotpotqa_entry("q1", 1.0, 1.0), hotpotqa_entry("q2", 0.0, 0.0)])
        b = self.write("shaped.jsonl", [hotpotqa_entry("q1", 1.0, 1.0), hotpotqa_entry("q2", 0.0, 0.28)])
        (row,) = compare(load_items(a), load_items(b))
        self.assertEqual((row["split"], row["a_correct"], row["b_correct"]), ("all", 1, 1))

    def test_hotpotqa_pairs_on_the_question_not_the_model_history(self):
        # The recorded prompt can already hold the model's earlier turn, which differs between runs
        def prompt(history: str) -> str:
            return f"system\nYou answer questions.\nuser\nQuestion: Where is the tower?\nassistant\n{history}\nuser\n..."

        a_entry = hotpotqa_entry("unused", 1.0, 1.0)
        b_entry = hotpotqa_entry("unused", 0.0, 0.1)
        a_entry["input"], b_entry["input"] = prompt("I will search."), prompt("Let me look it up.")
        (row,) = compare(load_items(self.write("a.jsonl", [a_entry])), load_items(self.write("b.jsonl", [b_entry])))
        self.assertEqual((row["items"], row["only_a"]), (1, 1))

    def test_duplicate_items_are_rejected(self):
        path = self.write("dup.jsonl", [hotpotqa_entry("q1", 1.0, 1.0), hotpotqa_entry("q1", 0.0, 0.0)])
        with self.assertRaises(ValueError):
            load_items(path)

    def test_summary_per_split_and_family(self):
        path = self.write("s.jsonl", [
            alfworld_entry("t1", "valid_seen", "pick_and_place", True),
            alfworld_entry("t2", "valid_seen", "pick_and_place", False),
            alfworld_entry("t3", "valid_unseen", "pick_two_obj_and_place", True),
        ])
        summary = summarize(load_entries(path))
        self.assertEqual((summary["valid_seen"]["solved"], summary["valid_seen"]["total"]), (1, 2))
        self.assertEqual(summary["valid_seen"]["families"]["pick_and_place"], [1, 2])
        self.assertEqual(summary["valid_unseen"]["families"]["pick_two_obj_and_place"], [1, 1])


if __name__ == "__main__":
    unittest.main()
