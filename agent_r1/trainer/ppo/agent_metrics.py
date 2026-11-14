"""Agent-level training diagnostics computed on the driver.

These metrics complement `compute_data_metrics` with views that only make sense for multi-step agent
trajectories: how advantages are spread over step indices (credit assignment), whether prompt groups carry
any learning signal for group-relative estimators, why trajectories stopped, and the recipe-specific reward
diagnostics (`reward_extra_info`) aggregated per training step. Everything is computed from tensors and arrays
already present in the post-update batch, so it adds no GPU work.
"""

from __future__ import annotations

from collections.abc import Iterable
from numbers import Real
from typing import Any

import numpy as np
import torch

STOP_REASONS = ("done", "max_steps", "prompt_overflow")
# reward_extra_info keys that are not worth averaging: `score` duplicates critic/score/*
_SKIPPED_REWARD_INFO_KEYS = frozenset({"score"})


def _numeric_or_nan(values: np.ndarray) -> np.ndarray | None:
    """Map an object array to float64 (bools count as 0/1, anything non-numeric becomes NaN).

    Returns None when no element is numeric, e.g. for string metadata such as task ids.
    """
    out = np.full(len(values), np.nan, dtype=np.float64)
    for i, value in enumerate(values):
        if isinstance(value, np.bool_ | Real):  # Real covers bool, int, float and numpy numbers
            out[i] = float(value)
    return None if np.isnan(out).all() else out


def compute_agent_metrics(batch: Any, max_step_index: int = 5, reward_extra_keys: Iterable[str] = ()) -> dict[str, Any]:
    """Compute `agent/*` diagnostics from a training batch after advantage computation.

    Each row of `batch` is one agent step, not one trajectory. Rows of the same trajectory share
    `non_tensor_batch["trajectory_uids"]` and are ordered by `step_indices` (0 .. n-1); trajectories rolled out
    from the same prompt share `non_tensor_batch["uid"]` (the GRPO/RLOO group). Pass only valid rows: padding
    rows (`sample_mask=False`) must be removed beforehand.

    Args:
        batch (Any): A DataProto-like object. `batch.batch` holds `response_mask`, `token_level_scores`,
            `token_level_rewards`, `advantages` and `returns`, each of shape (bs, response_length) where bs is the
            number of step rows, plus `values` of the same shape when a critic is used. `response_mask` is 1 for
            LLM-generated tokens; the step reward sits on the last generated token.
        max_step_index (int): Per-step-index metrics are emitted for step indices `0 .. max_step_index - 1`.
        reward_extra_keys (Iterable[str]): Keys of `reward_extra_info` stored in `non_tensor_batch`, one value per
            step row. Numeric/bool keys are averaged over all steps and over each trajectory's last step, e.g.
            ALFWorld `is_action_valid` (per-step valid-action rate) and `success` (last-step success rate).

    Returns:
        dict[str, Any]: Scalar metrics under the `agent/` prefix; empty if the batch has no trajectory info.
    """
    non_tensor = batch.non_tensor_batch
    if "trajectory_uids" not in non_tensor or "step_indices" not in non_tensor:
        return {}

    response_mask = batch.batch["response_mask"].bool()  # (bs, response_length)
    num_rows = response_mask.shape[0]
    if num_rows == 0:
        return {}
    device = response_mask.device

    # Per-row (= per-step) scalars, all of shape (bs,). Float64 keeps the per-token mean of a broadcast
    # step-level advantage equal across steps, so step-level estimators show an exact zero spread below.
    num_action_tokens = response_mask.sum(-1).clamp(min=1).double()
    step_scores = (batch.batch["token_level_scores"].double() * response_mask).sum(-1)
    step_rewards = (batch.batch["token_level_rewards"].double() * response_mask).sum(-1)
    step_advantages = (batch.batch["advantages"].double() * response_mask).sum(-1) / num_action_tokens
    values = batch.batch.get("values", None)
    # V(s_t): critic output at the action boundary (last prompt token), the value step-level GAE uses
    step_values = values[:, 0].double() if values is not None else None

    trajectory_uids = non_tensor["trajectory_uids"]
    step_indices = np.asarray(non_tensor["step_indices"], dtype=np.int64)  # (bs,)
    _, trajectory_ids = np.unique(trajectory_uids, return_inverse=True)  # (bs,) contiguous trajectory id per row
    trajectory_ids = trajectory_ids.reshape(-1)
    num_trajectories = int(trajectory_ids.max()) + 1
    steps_per_trajectory = np.bincount(trajectory_ids, minlength=num_trajectories)  # (num_trajectories,)
    last_step_index = np.full(num_trajectories, -1, dtype=np.int64)
    np.maximum.at(last_step_index, trajectory_ids, step_indices)
    is_last_step = step_indices == last_step_index[trajectory_ids]  # (bs,) True on each trajectory's final step

    metrics: dict[str, Any] = {
        "agent/batch/num_step_rows": num_rows,
        "agent/batch/num_trajectories": num_trajectories,
    }

    # 1) Credit assignment by step index: what each step earned and what the estimator credited it with.
    def _add_step_group(prefix: str, rows: np.ndarray) -> None:
        rows_t = torch.as_tensor(rows, device=device)
        metrics[f"{prefix}/num_rows"] = int(rows.sum())
        metrics[f"{prefix}/score_mean"] = step_scores[rows_t].mean().item()
        metrics[f"{prefix}/adv_mean"] = step_advantages[rows_t].mean().item()
        metrics[f"{prefix}/adv_std"] = step_advantages[rows_t].std(correction=0).item()
        if step_values is not None:
            metrics[f"{prefix}/value_mean"] = step_values[rows_t].mean().item()

    for k in range(max_step_index):
        rows = step_indices == k
        if not rows.any():
            # step indices are contiguous from 0, so no trajectory reached any later step either
            break
        _add_step_group(f"agent/step_{k}", rows)
    _add_step_group("agent/step_last", is_last_step)

    # Critic fit at the action boundary. Step-level GAE trains the value only at the first response token, while
    # verl's critic/vf_explained_var averages over every response token, including positions never trained.
    returns = batch.batch.get("returns", None)
    if step_values is not None and returns is not None:
        has_action = response_mask[:, 0]  # (bs,) rows with at least one generated token
        if int(has_action.sum()) > 1:
            target = returns[:, 0].double()[has_action]  # value target at the action boundary
            residual_var = torch.var(target - step_values[has_action])
            metrics["agent/critic/step_vf_explained_var"] = (1.0 - residual_var / (torch.var(target) + 1e-5)).item()

    # 2) Spread of advantages inside a trajectory: 0 when all of its steps share one advantage (grpo/rloo),
    #    > 0 when the estimator assigns step-level credit (gae/token_gae/reinforce/gigpo).
    multi_step = steps_per_trajectory >= 2
    if multi_step.any():
        adv = step_advantages.cpu().numpy()
        traj_mean = np.bincount(trajectory_ids, weights=adv, minlength=num_trajectories) / steps_per_trajectory
        deviation = adv - traj_mean[trajectory_ids]
        traj_var = np.bincount(trajectory_ids, weights=deviation**2, minlength=num_trajectories) / steps_per_trajectory
        metrics["agent/credit/within_traj_adv_std"] = float(np.sqrt(traj_var[multi_step]).mean())
    metrics["agent/credit/multi_step_traj_ratio"] = float(multi_step.mean())

    # 3) Prompt groups (rollout.n > 1): a group whose trajectory returns are all equal gets zero advantage
    #    from grpo/rloo, i.e. contributes no policy gradient. Uses token_level_rewards, as the estimators do.
    if "uid" in non_tensor:
        traj_returns = np.bincount(trajectory_ids, weights=step_rewards.cpu().numpy(), minlength=num_trajectories)
        traj_uid = np.empty(num_trajectories, dtype=object)
        traj_uid[trajectory_ids] = np.asarray(non_tensor["uid"])  # all steps of a trajectory share its prompt uid
        _, group_ids = np.unique(traj_uid, return_inverse=True)
        group_ids = group_ids.reshape(-1)  # (num_trajectories,)
        num_groups = int(group_ids.max()) + 1
        group_size = np.bincount(group_ids, minlength=num_groups)
        multi_traj_group = group_size >= 2
        if multi_traj_group.any():
            group_max = np.full(num_groups, -np.inf)
            group_min = np.full(num_groups, np.inf)
            np.maximum.at(group_max, group_ids, traj_returns)
            np.minimum.at(group_min, group_ids, traj_returns)
            zero_std = multi_traj_group & (group_max - group_min <= 1e-6)
            num_multi = multi_traj_group.sum()
            metrics["agent/group/zero_std_ratio"] = float(zero_std.sum() / num_multi)
            metrics["agent/group/zero_std_solved_ratio"] = float((zero_std & (group_max > 0)).sum() / num_multi)
            metrics["agent/group/zero_std_failed_ratio"] = float((zero_std & (group_max <= 0)).sum() / num_multi)
            group_mean = np.bincount(group_ids, weights=traj_returns, minlength=num_groups) / group_size
            sq_dev = np.bincount(group_ids, weights=(traj_returns - group_mean[group_ids]) ** 2, minlength=num_groups)
            group_std = np.sqrt(sq_dev[multi_traj_group] / (group_size[multi_traj_group] - 1))  # unbiased, as GRPO
            metrics["agent/group/return_std_mean"] = float(group_std.mean())

    # 4) Why trajectories ended, recorded on the last step by flows that set `stop_reason` (e.g. AgentEnvLoop).
    stop_reason = non_tensor.get("stop_reason", None)
    if stop_reason is not None:
        reasons = [reason for reason in np.asarray(stop_reason)[is_last_step] if isinstance(reason, str)]
        if reasons:
            for name in STOP_REASONS:
                metrics[f"agent/stop/{name}_ratio"] = sum(reason == name for reason in reasons) / len(reasons)

    # 5) Recipe-specific reward diagnostics, one value per step row (bs,): `mean` over all steps (per-step rates,
    #    e.g. valid actions) and `last_step_mean` over each trajectory's final step (outcomes, e.g. success).
    #    Index/id fields and string metadata are skipped.
    for key in reward_extra_keys:
        if key in _SKIPPED_REWARD_INFO_KEYS or key.endswith("index") or key.endswith("_id") or key not in non_tensor:
            continue
        values = _numeric_or_nan(np.asarray(non_tensor[key], dtype=object).reshape(-1))
        if values is None or len(values) != num_rows:
            continue
        has_value = ~np.isnan(values)
        if has_value.any():
            metrics[f"agent/reward_info/{key}/mean"] = float(values[has_value].mean())
        last_with_value = has_value & is_last_step
        if last_with_value.any():
            metrics[f"agent/reward_info/{key}/last_step_mean"] = float(values[last_with_value].mean())

    return metrics
