"""Summarize one validation dump: success per split and per task family, and the mean of every logged metric.

    python -m tools.eval_summary outputs/<run>/validation/200.jsonl [--by task_family]

ALFWorld dumps report solved / total per split and task family; HotpotQA dumps report exact match together with the
behaviour metrics the flow logs (`f1`, `answered`, `evidence_hit`, `repeat_queries`, `num_tool_steps`).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from numbers import Real

from tools.validation_dump import is_correct, item_group, last_step, load_entries


def summarize(entries: list[dict], by: str = "task_family", metric: str = "auto") -> dict:
    """Return {split: {"solved", "total", "families": {family: [solved, total]}, "means": {field: mean}}}."""
    summary: dict = {}
    sums: dict = defaultdict(lambda: defaultdict(float))
    for entry in entries:
        split = item_group(entry)
        info = last_step(entry)
        s = summary.setdefault(split, {"solved": 0, "total": 0, "families": {}, "means": {}})
        correct = is_correct(entry, metric)
        s["solved"] += int(correct)
        s["total"] += 1
        family = info.get(by)
        if family:
            fam = s["families"].setdefault(str(family), [0, 0])
            fam[0] += int(correct)
            fam[1] += 1
        for name, value in info.items():
            if isinstance(value, Real) and name not in ("step_index", "score"):
                sums[split][name] += float(value)
    for split, s in summary.items():
        s["means"] = {name: total / s["total"] for name, total in sorted(sums[split].items())}
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump")
    parser.add_argument("--by", default="task_family", help="field to break results down by")
    parser.add_argument("--metric", default="auto", help="correctness field: auto, acc, success or score")
    args = parser.parse_args(argv)

    for split, s in sorted(summarize(load_entries(args.dump), args.by, args.metric).items()):
        print(f"[{split}] {s['solved']}/{s['total']} = {100 * s['solved'] / s['total']:.1f}%")
        for family, (solved, total) in sorted(s["families"].items()):
            print(f"  {family:<28} {solved}/{total}")
        for name, mean in s["means"].items():
            print(f"  mean {name:<24} {mean:.4f}")


if __name__ == "__main__":
    main()
