"""Draw the README figures from results/*.csv, each in a light and a dark variant.

    python -m tools.plot_results [--results results] [--out assets]

Needs pandas and matplotlib. Colors follow one entity per run across all figures (step-level advantage = blue,
GRPO = orange, SFT -> RL = aqua, 4 updates per step = yellow, dynamic sampling = magenta; HotpotQA EM reward = blue,
shaped reward = aqua); the pairs that share a chart were checked for color-vision-deficiency separation.
Image sizes assume display at about 880 px wide (GitHub README), so line widths and fonts are scaled up.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

DPI = 150
LINE_W = 1.5  # pt, about 2 px when the image is shown at 65 %
DOT = 5.9  # marker diameter in pt, about 8 px on screen
RING = 1.4  # surface-colored ring around end dots

THEMES = {
    "light": {
        "bg": "#ffffff",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e9e8e4",
        "spine": "#c9c8c2",
        "wash": "#52514e",
        "series": {
            "gigpo": "#2a78d6",
            "grpo": "#eb6834",
            "sft_gigpo": "#1baf7a",
            "gigpo_4upd": "#eda100",
            "gigpo_dynsamp": "#e87ba4",
            "em": "#2a78d6",
            "shaped": "#1baf7a",
        },
        "ramp": ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
    },
    "dark": {
        "bg": "#0d1117",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#262a31",
        "spine": "#3d424b",
        "wash": "#c3c2b7",
        "series": {
            "gigpo": "#3987e5",
            "grpo": "#d95926",
            "sft_gigpo": "#199e70",
            "gigpo_4upd": "#c98500",
            "gigpo_dynsamp": "#d55181",
            "em": "#3987e5",
            "shaped": "#199e70",
        },
        "ramp": ["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
    },
}

LABELS = {
    "gigpo": "Step-level advantage",
    "grpo": "GRPO",
    "sft_gigpo": "Expert SFT, then RL",
    "gigpo_4upd": "4 updates per step",
    "gigpo_dynsamp": "Dynamic sampling",
    "em": "EM reward",
    "shaped": "Shaped reward",
}


def setup(theme: dict, ncols: int, width: float = 9.0, height: float = 3.5):
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 9.2,
            "text.color": theme["text"],
            "axes.labelcolor": theme["muted"],
            "xtick.color": theme["muted"],
            "ytick.color": theme["muted"],
        }
    )
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=DPI, squeeze=False)
    fig.patch.set_facecolor(theme["bg"])
    for ax in axes[0]:
        ax.set_facecolor(theme["bg"])
        ax.grid(True, axis="y", color=theme["grid"], linewidth=0.7, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(theme["spine"])
            ax.spines[side].set_linewidth(0.7)
        ax.tick_params(length=0, pad=4)
    return fig, axes[0]


def titles(fig, theme: dict, title: str, subtitle: str) -> None:
    fig.text(0.01, 0.985, title, ha="left", va="top", fontsize=11.5, fontweight="bold", color=theme["text"])
    fig.text(0.01, 0.915, subtitle, ha="left", va="top", fontsize=9.2, color=theme["muted"])


def legend(fig, theme: dict, runs: list[str], y: float = 0.86, names: dict[str, str] | None = None) -> None:
    handles = [
        Line2D([0], [0], color=theme["series"][r], linewidth=LINE_W * 1.6, solid_capstyle="round") for r in runs
    ]
    names = {**LABELS, **(names or {})}
    leg = fig.legend(
        handles,
        [names[r] for r in runs],
        loc="upper left",
        bbox_to_anchor=(0.005, y),
        ncol=len(runs),
        frameon=False,
        handlelength=1.6,
        columnspacing=1.6,
    )
    for text in leg.get_texts():
        text.set_color(theme["text"])


def line(ax, theme: dict, run: str, x, y) -> None:
    color = theme["series"][run]
    ax.plot(x, y, color=color, linewidth=LINE_W, solid_joinstyle="round", solid_capstyle="round", zorder=3)
    ax.plot(
        [x.iloc[-1]],
        [y.iloc[-1]],
        marker="o",
        markersize=DOT,
        color=color,
        markeredgecolor=theme["bg"],
        markeredgewidth=RING,
        zorder=4,
    )


def end_labels(ax, theme: dict, points: list[tuple[float, float]], fmt: str = "{:.1f}", min_gap: float = 6.0) -> None:
    """Value labels at line ends; skipped when two ends are too close to label without stacking."""
    ys = sorted(y for _, y in points)
    if any(b - a < min_gap for a, b in zip(ys, ys[1:])):
        return
    for x, y in points:
        ax.annotate(
            fmt.format(y),
            (x, y),
            xytext=(6, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.8,
            color=theme["text"],
        )


def save(fig, out: Path, name: str, theme_name: str) -> None:
    suffix = "" if theme_name == "light" else "_dark"
    fig.savefig(out / f"{name}{suffix}.png", facecolor=fig.get_facecolor(), dpi=DPI)
    plt.close(fig)


def fig_steplevel_vs_grpo(val: pd.DataFrame, theme: dict) -> plt.Figure:
    fig, axes = setup(theme, 2)
    runs = ["gigpo", "grpo"]
    for ax, split, n in zip(axes, ["seen", "unseen"], [140, 134]):
        ax.axvspan(160, 200, color=theme["wash"], alpha=0.07, linewidth=0, zorder=1)
        ax.text(180, 3, "steps 160-200", ha="center", va="bottom", fontsize=8.2, color=theme["muted"])
        ends = []
        for run in runs:
            d = val[val.run == run]
            line(ax, theme, run, d.step, d[split])
            ends.append((d.step.iloc[-1], d[split].iloc[-1]))
        end_labels(ax, theme, ends)
        ax.set_title(f"valid_{split} ({n} tasks)", loc="left", color=theme["text"])
        ax.set_xlim(0, 214)
        ax.set_ylim(0, 100)
        ax.set_xlabel("training step")
    axes[0].set_ylabel("success rate (%)")
    titles(fig, theme, "ALFWorld: step-level advantage vs GRPO", "Qwen3-4B-Instruct-2507, single seed, greedy decoding")
    legend(fig, theme, runs)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.72, wspace=0.18)
    return fig


def fig_sft_rl(val: pd.DataFrame, theme: dict, base: tuple[float, float]) -> plt.Figure:
    fig, axes = setup(theme, 2)
    d = val[val.run == "sft_gigpo"].set_index("step")
    stages = ["Base\nmodel", "Expert\nSFT", "+ RL\n10 steps", "+ RL\n20 steps", "+ RL\n100 steps"]
    for ax, split, n, base_value in zip(axes, ["seen", "unseen"], [140, 134], base):
        values = [base_value, d.loc[0, split], d.loc[10, split], d.loc[20, split], d.loc[100, split]]
        xs = range(len(stages))
        ax.bar(xs, values, width=0.32, color=theme["ramp"], linewidth=0, zorder=3)
        for x, v in zip(xs, values):
            ax.text(x, v + 2, f"{v:.1f}", ha="center", va="bottom", fontsize=8.8, color=theme["text"])
        ax.set_xticks(list(xs), stages)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(f"valid_{split} ({n} tasks)", loc="left", color=theme["text"])
    axes[0].set_ylabel("success rate (%)")
    titles(
        fig,
        theme,
        "ALFWorld: expert SFT, then agentic RL",
        "23k step-level expert demonstrations for SFT, then step-level-advantage RL from the SFT model; single seed",
    )
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.17, top=0.78, wspace=0.14)
    return fig


def fig_multi_update(val: pd.DataFrame, theme: dict) -> plt.Figure:
    fig, axes = setup(theme, 2)
    runs = ["gigpo_4upd", "gigpo"]
    for ax, split, n in zip(axes, ["seen", "unseen"], [140, 134]):
        ax.axvline(100, color=theme["spine"], linewidth=0.7, zorder=1)
        ends = []
        for run in runs:
            dd = val[val.run == run]
            line(ax, theme, run, dd.step, dd[split])
            ends.append((dd.step.iloc[-1], dd[split].iloc[-1]))
        end_labels(ax, theme, ends)
        ax.set_title(f"valid_{split} ({n} tasks)", loc="left", color=theme["text"])
        ax.set_xlim(0, 214)
        ax.set_ylim(0, 100)
        ax.set_xlabel("training step (same rollouts per step)")
    axes[0].set_ylabel("success rate (%)")
    titles(
        fig,
        theme,
        "ALFWorld: 4 mini-batch updates per step vs 1",
        "same rollouts and data order; the 4-update run also widens the PPO clip to 0.2; single seed",
    )
    legend(fig, theme, runs, names={"gigpo": "1 update per step"})
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.72, wspace=0.18)
    return fig


def fig_hotpotqa(hp: pd.DataFrame, theme: dict) -> plt.Figure:
    fig, axes = setup(theme, 2)
    runs = ["em", "shaped"]
    ends_em, ends_rq = [], []
    for run in runs:
        d = hp[hp.run == run]
        line(axes[0], theme, run, d.step, d.em)
        line(axes[1], theme, run, d.step, d.repeat_queries)
        ends_em.append((d.step.iloc[-1], d.em.iloc[-1]))
        ends_rq.append((d.step.iloc[-1], d.repeat_queries.iloc[-1]))
    end_labels(axes[0], theme, ends_em, min_gap=1.0)
    end_labels(axes[1], theme, ends_rq, fmt="{:.3f}", min_gap=0.02)
    axes[0].set_title("exact match on 1,000 dev questions (%)", loc="left", color=theme["text"])
    axes[0].set_ylim(44, 57)
    axes[1].set_title("repeated search queries per question", loc="left", color=theme["text"])
    axes[1].set_ylim(0, 0.2)
    for ax in axes:
        ax.set_xlim(0, 218)
        ax.set_xlabel("training step")
    titles(
        fig,
        theme,
        "HotpotQA search agent: exact-match reward vs shaped reward",
        "shaped = F1 partial credit + evidence credit + repeated-query penalty; evaluation is always exact match",
    )
    legend(fig, theme, runs)
    fig.subplots_adjust(left=0.06, right=0.97, bottom=0.15, top=0.72, wspace=0.2)
    return fig


def fig_training_signal(train: pd.DataFrame, theme: dict) -> plt.Figure:
    fig, axes = setup(theme, 1, width=9.0, height=3.4)
    ax = axes[0]
    runs = ["gigpo", "grpo", "sft_gigpo"]
    for run in runs:
        d = train[train.run == run].sort_values("step")
        smooth = d.no_signal_groups.rolling(10, min_periods=1).mean() * 100
        line(ax, theme, run, d.step, smooth)
    ax.set_xlim(0, 205)
    ax.set_ylim(0, 100)
    ax.set_xlabel("training step")
    ax.set_ylabel("groups without signal (%)")
    ax.set_title(
        "share of prompt groups whose 8 rollouts all succeed or all fail (10-step mean)",
        loc="left",
        color=theme["text"],
    )
    titles(fig, theme, "ALFWorld: how sparse the learning signal is", "such groups get zero advantage; single seed")
    legend(fig, theme, runs)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.15, top=0.7)
    return fig


def fig_dynamic_sampling(val: pd.DataFrame, theme: dict) -> plt.Figure:
    fig, axes = setup(theme, 3, width=9.6, height=3.3)
    runs = ["gigpo_dynsamp", "gigpo"]
    views = [("step", "training step", None), ("hours", "wall-clock hours (2 GPUs)", None)]
    views.append(("trajectories", "trajectories rolled out (thousands)", 1000))
    for ax, (col, xlabel, scale) in zip(axes, views):
        for run in runs:
            d = val[val.run == run]
            x = d[col] / scale if scale else d[col]
            line(ax, theme, run, x, d.seen)
        ax.set_ylim(0, 100)
        ax.set_xlabel(xlabel)
    axes[0].set_xlim(0, 205)
    axes[0].set_ylabel("valid_seen success (%)")
    titles(
        fig,
        theme,
        "ALFWorld: dynamic sampling (experimental)",
        "up to 3 rollout batches per step, train only on groups with signal; run stopped at step 109 (host OOM)",
    )
    legend(fig, theme, runs, names={"gigpo": "Without dynamic sampling"})
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.17, top=0.7, wspace=0.22)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="assets")
    parser.add_argument(
        "--base", nargs=2, type=float, default=(7.1, 2.2), help="base-model seen / unseen success for the SFT figure"
    )
    args = parser.parse_args(argv)
    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(results / "alfworld_val.csv")
    train = pd.read_csv(results / "alfworld_train.csv")
    hp = pd.read_csv(results / "hotpotqa_val.csv")
    for theme_name, theme in THEMES.items():
        save(fig_steplevel_vs_grpo(val, theme), out, "alfworld_steplevel_vs_grpo", theme_name)
        save(fig_sft_rl(val, theme, tuple(args.base)), out, "alfworld_sft_rl", theme_name)
        save(fig_multi_update(val, theme), out, "alfworld_multi_update", theme_name)
        save(fig_hotpotqa(hp, theme), out, "hotpotqa_reward", theme_name)
        save(fig_training_signal(train, theme), out, "alfworld_training_signal", theme_name)
        save(fig_dynamic_sampling(val, theme), out, "alfworld_dynamic_sampling", theme_name)
    print(f"wrote {len(list(out.glob('*.png')))} images to {out}")


if __name__ == "__main__":
    main()
