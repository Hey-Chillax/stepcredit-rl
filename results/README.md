# Results data

The CSV files behind the tables and figures of the main README. All runs: Qwen3-4B-Instruct-2507, seed 42, one run per
configuration. Success rates are percentages; ALFWorld validation uses valid_seen (140 tasks) and valid_unseen
(134 tasks), HotpotQA a fixed subset of 1,000 dev questions.

| File | Content | Produced by |
|---|---|---|
| `alfworld_val.csv` | Validation success (seen / unseen), wall-clock hours and trajectories rolled out, per run and validation step | `tools/build_results.py` from the W&B history |
| `alfworld_train.csv` | Per training step: training success, episode length, within-trajectory advantage spread, share of groups without signal (all failed / all solved), response length, entropy | `tools/build_results.py` |
| `hotpotqa_val.csv` | Exact match, F1 and repeated queries per question, per run and validation step | `tools/build_results.py` |
| `paired_tests.csv` | Item-level 2 × 2 counts at step 200 and exact McNemar p-values | `tools/paired_eval.py` on the validation records |
| `alfworld_family.csv` | Solved / total per task family: both 200-step runs at step 200, the SFT model, and SFT + 100 RL steps | `tools/eval_summary.py` on the validation records |
| `expert_sft_data.csv` | Training games, games the planner expert solves within 20 steps, and SFT rows per task family | statistics of `recipes/alfworld/data_preprocess/generate_expert_sft.py` |
| `loop_offline.csv` | No-op statistics of the 200-step step-level run in three training phases: share of executed commands that are no-ops (all, failed and successful trajectories), no-ops per failed trajectory, and the share of rows in all-failed groups that the loop-aware reward would reach | offline count over the rollout records |

Run labels: `gigpo` = step-level advantage, 200 steps; `grpo` = GRPO, 200 steps; `gigpo_4upd` = 4 mini-batch updates per
step, 100 steps; `gigpo_dynsamp` = dynamic sampling, stopped at step 109; `sft_gigpo` = RL after expert SFT, 100 steps
(step 0 is the SFT model); HotpotQA `em` = exact-match reward, `shaped` = shaped reward. `gigpo` is the name of the
step-level advantage estimator in the code (`algorithm.adv_estimator=gigpo`).
