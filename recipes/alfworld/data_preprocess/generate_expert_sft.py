#!/usr/bin/env python3
"""
Generate step-level SFT data for ALFWorld from the hand-coded expert.

Every successful expert episode becomes one row per agent step. A row holds the same [system, user] messages the
rollout builds for that step, plus the tool call the expert takes, so SFT sees the prompts RL will see. Only training
games are used. Episodes that fail, exceed the rollout's step limit, or use a command outside the admissible list are
dropped.

The tool schema is stored already rendered into the system message, not as a `tools` column: parquet sorts struct
fields by name, which would change the key order of the tool JSON and so the prompt tokens. Generation fails if the
stored messages do not tokenize to exactly what the rollout feeds the model.

Output, written to --output_dir:
- train.parquet / val.parquet: columns `messages`, `task_id`, `task_family`, `step_index`.
  The validation rows come from held-out training games, not from the valid_seen / valid_unseen evaluation splits.
- stats.json: episode counts and step statistics per task family.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from transformers import AutoTokenizer

from recipes.alfworld.env.alfworld_wrapper import AlfworldTextworldEnv
from recipes.alfworld.prompts import ALFWORLD_SYSTEM_PROMPT, ALFWORLD_TOOL_SCHEMAS
from recipes.alfworld.utils import build_alfworld_messages, extract_task_text

_ENV: AlfworldTextworldEnv | None = None


def format_tool_call(command: str) -> str:
    """Render a command the way the model emits it during rollout (hermes tool-call format)."""
    payload = json.dumps({"name": "env_step", "arguments": {"command": command}})
    return f"<tool_call>\n{payload}\n</tool_call>"


def render_system_message(tokenizer: Any) -> str:
    """Return the system text the chat template builds from the system prompt and the tool schema."""
    rendered = tokenizer.apply_chat_template(
        [{"role": "system", "content": ALFWORLD_SYSTEM_PROMPT}],
        tools=ALFWORLD_TOOL_SCHEMAS,
        add_generation_prompt=False,
        tokenize=False,
    )
    prefix, suffix = "<|im_start|>system\n", "<|im_end|>\n"
    if not (rendered.startswith(prefix) and rendered.endswith(suffix)):
        raise ValueError(f"unexpected chat template output: {rendered[:80]!r} ... {rendered[-40:]!r}")
    return rendered[len(prefix) : -len(suffix)]


def check_matches_rollout(tokenizer: Any, rows: list[dict[str, Any]], system_message: str) -> None:
    """Fail unless stored rows tokenize to the prompt the rollout builds from the same messages with `tools=`."""
    for row in rows:
        rollout_messages = [{"role": "system", "content": ALFWORLD_SYSTEM_PROMPT}, row["messages"][1]]
        rollout_ids = tokenizer.apply_chat_template(
            rollout_messages, tools=ALFWORLD_TOOL_SCHEMAS, add_generation_prompt=True, tokenize=True
        )
        stored_messages = [{"role": "system", "content": system_message}, row["messages"][1]]
        stored_ids = tokenizer.apply_chat_template(stored_messages, add_generation_prompt=True, tokenize=True)
        if stored_ids != rollout_ids:
            raise ValueError(f"SFT prompt differs from the rollout prompt for task {row['task_id']}")


def _init_worker(data_root: str, max_steps: int) -> None:
    global _ENV
    _ENV = AlfworldTextworldEnv(data_root=data_root, max_episode_steps=max_steps, with_expert=True)


def _run_expert_episode(task: dict[str, Any]) -> dict[str, Any]:
    """Play one game with the expert and return its step rows, or the reason it was dropped."""
    result: dict[str, Any] = {"family": task["task_family"], "rows": [], "reason": None, "num_steps": 0}
    try:
        observation, info = _ENV.reset_with_info(game_relative_path=task["game_relative_path"])
    except Exception as e:  # a broken game file must not stop the whole dataset
        result["reason"] = f"reset_error:{type(e).__name__}"
        return result

    task_text = extract_task_text(observation, task.get("goal_text"))
    history_actions: list[str] = []
    rows: list[dict[str, Any]] = []
    done = False

    while not done and len(rows) < task["max_steps"]:
        plan = info.get("extra.expert_plan") or []
        admissible_commands = list(info.get("admissible_commands") or [])
        if not plan:
            result["reason"] = "empty_plan"
            return result
        command = str(plan[0]).strip()
        if command not in admissible_commands:
            result["reason"] = "inadmissible_command"
            return result

        messages = build_alfworld_messages(
            task_text=task_text,
            observation=observation,
            history_actions=history_actions,
            admissible_commands=admissible_commands,
        )
        messages.append({"role": "assistant", "content": format_tool_call(command)})
        rows.append(
            {
                "messages": messages,
                "task_id": task["task_id"],
                "task_family": task["task_family"],
                "step_index": len(rows),
            }
        )

        observation, _, done, info = _ENV.step(command)
        observation = str(observation)
        history_actions.append(command)

    if not info.get("success"):
        result["reason"] = "not_solved_within_limit"
        return result

    result["rows"] = rows
    result["num_steps"] = len(rows)
    return result


def _load_tasks(train_parquet: str, max_steps: int) -> list[dict[str, Any]]:
    tasks = []
    for extra_info in pd.read_parquet(train_parquet)["extra_info"]:
        tasks.append(
            {
                "task_id": str(extra_info.get("task_id", "")),
                "task_family": str(extra_info.get("task_family", "")),
                "goal_text": extra_info.get("goal_text"),
                "game_relative_path": extra_info["game_relative_path"],
                "max_steps": max_steps,
            }
        )
    return tasks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train_parquet", required=True, help="ALFWorld train.parquet")
    parser.add_argument("--data_root", required=True, help="directory holding games/")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--tokenizer_path", required=True, help="model whose chat template renders the tool schema")
    parser.add_argument("--max_steps", type=int, default=20, help="must match max_steps in recipes/alfworld/base.yaml")
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--val_episodes", type=int, default=64, help="solved episodes held out for SFT validation")
    parser.add_argument("--limit", type=int, default=0, help="use only the first N tasks after shuffling; 0 = all")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = _load_tasks(args.train_parquet, args.max_steps)
    random.Random(args.seed).shuffle(tasks)
    if args.limit > 0:
        tasks = tasks[: args.limit]

    with mp.Pool(args.num_workers, initializer=_init_worker, initargs=(args.data_root, args.max_steps)) as pool:
        episodes = pool.map(_run_expert_episode, tasks, chunksize=4)

    solved = [episode for episode in episodes if episode["rows"]]
    if len(solved) <= args.val_episodes:
        raise RuntimeError(f"only {len(solved)} solved episodes, need more than val_episodes={args.val_episodes}")
    val_rows = [row for episode in solved[: args.val_episodes] for row in episode["rows"]]
    train_rows = [row for episode in solved[args.val_episodes :] for row in episode["rows"]]

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    system_message = render_system_message(tokenizer)
    sample_rows = random.Random(args.seed).sample(train_rows, min(200, len(train_rows)))
    check_matches_rollout(tokenizer, sample_rows, system_message)
    for row in train_rows + val_rows:
        row["messages"][0]["content"] = system_message

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(train_rows).to_parquet(output_dir / "train.parquet")
    pd.DataFrame(val_rows).to_parquet(output_dir / "val.parquet")

    per_family: dict[str, dict[str, int]] = defaultdict(lambda: {"tasks": 0, "solved": 0, "rows": 0})
    for episode in episodes:
        family = per_family[episode["family"]]
        family["tasks"] += 1
        family["solved"] += int(bool(episode["rows"]))
        family["rows"] += len(episode["rows"])
    steps = [episode["num_steps"] for episode in solved]
    stats = {
        "tasks": len(episodes),
        "solved": len(solved),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "mean_steps": sum(steps) / len(steps),
        "max_steps": max(steps),
        "dropped": dict(Counter(episode["reason"] for episode in episodes if episode["reason"])),
        "per_family": dict(per_family),
    }
    (output_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
