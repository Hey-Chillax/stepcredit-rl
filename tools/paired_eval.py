"""Paired comparison of two runs on the same validation items, with an exact McNemar test.

    python -m tools.paired_eval <run_a>/validation/200.jsonl <run_b>/validation/200.jsonl --names A B [--csv out.csv]

Both dumps come from `trainer.validation_data_dir`. Items are matched by `task_id` (ALFWorld) or by the question, the
first user turn of the prompt (HotpotQA), inside each split. With small validation sets and greedy decoding, two runs often differ by a handful of items, so
the test counts the items only one run solves rather than comparing two accuracies as if they were independent.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict

from tools.validation_dump import load_items, mcnemar_exact


def compare(items_a: dict, items_b: dict) -> list[dict]:
    """Return one row per group with the 2 x 2 counts over the items present in both runs and the McNemar p-value."""
    rows = []
    groups = sorted({group for group, _ in items_a} | {group for group, _ in items_b})
    for group in groups:
        keys_a = {key for g, key in items_a if g == group}
        keys_b = {key for g, key in items_b if g == group}
        shared = keys_a & keys_b
        counts: dict[tuple[bool, bool], int] = defaultdict(int)
        for key in shared:
            counts[(items_a[(group, key)].correct, items_b[(group, key)].correct)] += 1
        only_a, only_b = counts[(True, False)], counts[(False, True)]
        rows.append(
            {
                "split": group,
                "items": len(shared),
                "a_correct": counts[(True, True)] + only_a,
                "b_correct": counts[(True, True)] + only_b,
                "both_correct": counts[(True, True)],
                "only_a": only_a,
                "only_b": only_b,
                "neither": counts[(False, False)],
                "p": mcnemar_exact(only_a, only_b),
                "unpaired_a": len(keys_a - shared),
                "unpaired_b": len(keys_b - shared),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump_a")
    parser.add_argument("dump_b")
    parser.add_argument("--names", nargs=2, default=("A", "B"), help="labels of the two runs")
    parser.add_argument("--key", default="auto", help="pairing field: auto, task_id, question or input")
    parser.add_argument("--metric", default="auto", help="correctness field: auto, acc, success or score")
    parser.add_argument("--group", default="auto", help="split field: auto (split), none, or another field")
    parser.add_argument("--test-name", default="", help="value of the `test` column in --csv")
    parser.add_argument("--csv", help="append the rows to this CSV file")
    args = parser.parse_args(argv)

    items_a = load_items(args.dump_a, args.key, args.metric, args.group)
    items_b = load_items(args.dump_b, args.key, args.metric, args.group)
    rows = compare(items_a, items_b)

    name_a, name_b = args.names
    print(f"| split | items | {name_a} | {name_b} | only {name_a} | only {name_b} | McNemar p |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['split']} | {r['items']} | {r['a_correct']} ({100 * r['a_correct'] / max(r['items'], 1):.1f}%) "
            f"| {r['b_correct']} ({100 * r['b_correct'] / max(r['items'], 1):.1f}%) | {r['only_a']} | {r['only_b']} "
            f"| {r['p']:.4f} |"
        )
        if r["unpaired_a"] or r["unpaired_b"]:
            print(f"  note: {r['unpaired_a']} items only in {name_a}, {r['unpaired_b']} only in {name_b}", file=sys.stderr)

    if args.csv:
        fields = ["test", "split", "items", "both_correct", "only_a", "only_b", "a_correct", "b_correct", "p"]
        try:
            with open(args.csv, encoding="utf-8") as f:
                write_header = not f.read(1)
        except FileNotFoundError:
            write_header = True
        with open(args.csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            for r in rows:
                writer.writerow({**r, "test": args.test_name or f"{name_a}_vs_{name_b}", "p": round(r["p"], 4)})


if __name__ == "__main__":
    main()
