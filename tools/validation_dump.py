"""Read the per-trajectory validation dumps written by the trainer (`trainer.validation_data_dir/<step>.jsonl`).

Each line is one validation trajectory: `input` (decoded prompt of its first step), `output` (last response), `gts`,
`score` (sum of the step rewards), `num_steps` and `steps`. Every step row carries the reward info of its task, e.g.
`task_id`, `split`, `task_family`, `success` for ALFWorld and `acc`, `f1`, `answered`, `evidence_hit`,
`repeat_queries` for HotpotQA. The outcome sits on the last step (highest `step_index`).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Item:
    """One evaluation item (task or question) of a dump."""

    key: str  # identifies the item across runs: task_id when the dump has one, else the prompt text
    group: str  # split the item belongs to, e.g. valid_seen; "all" when the dump has no split field
    correct: bool
    info: dict[str, Any] = field(default_factory=dict, compare=False)  # reward info of the last step


def load_entries(path: str | Path) -> list[dict[str, Any]]:
    """Return the trajectory entries of one dump file, in file order."""
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def last_step(entry: dict[str, Any]) -> dict[str, Any]:
    """Return the step row with the highest step_index (the one holding the trajectory's outcome)."""
    steps = entry.get("steps") or []
    return max(steps, key=lambda step: step.get("step_index", 0)) if steps else {}


def first_user_turn(prompt: str) -> str:
    """Return the first user message of a decoded chat prompt.

    Decoding with skip_special_tokens keeps the role names, so a Qwen-style prompt reads
    "system\\n...\\nuser\\n<question>\\nassistant\\n...". The recorded prompt of a trajectory can already contain the
    model's own earlier turns, which differ between runs; the first user turn (the question) does not.
    """
    start = prompt.find("\nuser\n")
    if start >= 0:
        start += len("\nuser\n")
    elif prompt.startswith("user\n"):
        start = len("user\n")
    else:
        return prompt
    end = prompt.find("\nassistant\n", start)
    return prompt[start:] if end < 0 else prompt[start:end]


def item_key(entry: dict[str, Any], key: str = "auto") -> str:
    """Identify the item across runs.

    `auto` uses `task_id` (ALFWorld) when the steps carry one, otherwise the first user turn of the prompt (HotpotQA
    questions). `input` takes the whole recorded prompt, `question` the first user turn, any other name a step field.
    """
    info = last_step(entry)
    if key == "auto":
        key = "task_id" if info.get("task_id") else "question"
    if key == "question":
        value = first_user_turn(entry.get("input") or "")
    elif key == "input":
        value = entry.get("input")
    else:
        value = info.get(key)
    if value in (None, ""):
        raise ValueError(f"trajectory {entry.get('trajectory_uid')} has no '{key}' to pair on")
    return str(value)


def is_correct(entry: dict[str, Any], metric: str = "auto") -> bool:
    """Whether the trajectory solved its item.

    `auto` uses `acc` (exact match) when the steps log it, else `success`, else a positive total score. HotpotQA runs
    with a shaped training reward still log exact match as `acc`, so the comparison never depends on the reward.
    """
    info = last_step(entry)
    if metric == "auto":
        metric = "acc" if "acc" in info else "success" if "success" in info else "score"
    if metric == "score":
        return float(entry.get("score") or 0.0) > 0
    return float(info.get(metric) or 0.0) > 0


def item_group(entry: dict[str, Any], group: str = "auto") -> str:
    """The split of the item (`split` field for `auto`), or "all"."""
    if group == "none":
        return "all"
    field_name = "split" if group == "auto" else group
    return str(last_step(entry).get(field_name) or "all")


def load_items(
    path: str | Path, key: str = "auto", metric: str = "auto", group: str = "auto"
) -> dict[tuple[str, str], Item]:
    """Load a dump as {(group, key): Item}. Fails on duplicate items, which would make pairing ambiguous."""
    items: dict[tuple[str, str], Item] = {}
    for entry in load_entries(path):
        item = Item(
            key=item_key(entry, key),
            group=item_group(entry, group),
            correct=is_correct(entry, metric),
            info=last_step(entry),
        )
        if (item.group, item.key) in items:
            raise ValueError(f"{path}: item {item.key[:60]!r} appears twice in group {item.group!r}")
        items[(item.group, item.key)] = item
    return items


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test: binomial test of the discordant pairs against p = 0.5."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
