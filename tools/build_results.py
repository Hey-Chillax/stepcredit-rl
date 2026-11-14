"""Build the result tables in results/ from per-run W&B history CSVs (see tools/wandb_history.py).

    python -m tools.build_results <history_dir> <out_dir>

<history_dir> holds one CSV per run, named after the run labels used in results/:
    gigpo.csv          ALFWorld, step-level advantage, 200 steps
    grpo.csv           ALFWorld, GRPO, 200 steps (same settings otherwise)
    gigpo_4upd.csv     ALFWorld, step-level advantage with 4 mini-batch updates per step, 100 steps
    gigpo_dynsamp.csv  ALFWorld, step-level advantage with dynamic sampling (stopped at step 109)
    sft_gigpo.csv      ALFWorld, RL stage after expert SFT, 100 steps (step 0 = the SFT model)
    hotpotqa_em.csv    HotpotQA, exact-match reward, 200 steps
    hotpotqa_shaped.csv  HotpotQA, shaped reward (F1 partial credit, evidence, repeat penalty), 200 steps
Runs that are missing are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

SEEN = "val-core/alfworld_valid_seen/reward/mean@1"
UNSEEN = "val-core/alfworld_valid_unseen/reward/mean@1"
TRAJECTORIES_PER_STEP = 16 * 8  # prompts per step x rollouts per prompt
ALFWORLD_RUNS = ["gigpo", "grpo", "gigpo_4upd", "gigpo_dynsamp", "sft_gigpo"]
HOTPOTQA_RUNS = {"em": "hotpotqa_em", "shaped": "hotpotqa_shaped"}
HOTPOTQA_PREFIX = "hotpotqa_distractor"

# Training-batch statistics logged every step (agent/* come from agent_r1/trainer/ppo/agent_metrics.py)
TRAIN_KEYS = {
    "agent/reward_info/success/last_step_mean": "train_success",
    "agent/reward_info/num_steps/last_step_mean": "episode_steps",
    "agent/credit/within_traj_adv_std": "within_traj_adv_std",
    "agent/group/zero_std_ratio": "no_signal_groups",
    "agent/group/zero_std_failed_ratio": "all_failed_groups",
    "agent/group/zero_std_solved_ratio": "all_solved_groups",
    "response_length/mean": "response_tokens",
    "actor/entropy": "entropy",
}


def main(history_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def load(name: str) -> pd.DataFrame | None:
        path = history_dir / f"{name}.csv"
        return pd.read_csv(path).sort_values("step") if path.exists() else None

    # ALFWorld validation: one row per run and validation step
    val_rows, train_rows = [], []
    for run in ALFWORLD_RUNS:
        df = load(run)
        if df is None:
            continue
        if run == "gigpo_dynsamp":  # dynamic sampling rolls out up to 3 batches per step
            cumulative = df.set_index("step")["agent/sampling/trajectories_seen"].fillna(0).cumsum()
        for _, r in df.dropna(subset=[SEEN]).iterrows():
            step = int(r.step)
            if step == 0:
                trajectories = 0
            elif run == "gigpo_dynsamp":
                trajectories = int(cumulative.loc[step])
            else:
                trajectories = step * TRAJECTORIES_PER_STEP
            val_rows.append(
                {
                    "run": run,
                    "step": step,
                    "seen": round(r[SEEN] * 100, 1),
                    "unseen": round(r[UNSEEN] * 100, 1),
                    "hours": round(r["_runtime"] / 3600, 1),
                    "trajectories": trajectories,
                }
            )
        for _, r in df[df.step >= 1].iterrows():
            row = {"run": run, "step": int(r.step)}
            for key, name in TRAIN_KEYS.items():
                row[name] = round(float(r[key]), 4) if key in df and pd.notna(r[key]) else None
            train_rows.append(row)
    pd.DataFrame(val_rows).to_csv(out_dir / "alfworld_val.csv", index=False)
    pd.DataFrame(train_rows).to_csv(out_dir / "alfworld_train.csv", index=False)

    # HotpotQA validation: exact match (the evaluation metric for every run), F1 and repeated queries per question
    hp_rows = []
    for run, name in HOTPOTQA_RUNS.items():
        df = load(name)
        if df is None:
            continue
        for _, r in df.dropna(subset=[f"val-core/{HOTPOTQA_PREFIX}/acc/mean@1"]).iterrows():
            hp_rows.append(
                {
                    "run": run,
                    "step": int(r.step),
                    "em": round(r[f"val-core/{HOTPOTQA_PREFIX}/acc/mean@1"] * 100, 1),
                    "f1": round(r[f"val-aux/{HOTPOTQA_PREFIX}/f1/mean@1"], 3),
                    "repeat_queries": round(r[f"val-aux/{HOTPOTQA_PREFIX}/repeat_queries/mean@1"], 3),
                }
            )
    pd.DataFrame(hp_rows).to_csv(out_dir / "hotpotqa_val.csv", index=False)
    print(f"wrote {len(val_rows)} + {len(train_rows)} ALFWorld rows and {len(hp_rows)} HotpotQA rows to {out_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
