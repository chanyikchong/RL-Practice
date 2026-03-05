"""Multi-algorithm comparison dashboard."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.trainer import TrainingHistory


def _smooth(data: list[float], window: int = 50) -> np.ndarray:
    series = pd.Series(data)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def create_dashboard(
    histories: dict[str, TrainingHistory],
    snapshot_frames: dict[str, np.ndarray] | None = None,
    save_path: str | None = None,
    window: int = 50,
) -> None:
    """Create a 2-row dashboard comparing algorithms.

    Top row: Training curves per algorithm.
    Bottom row: Policy behavior snapshots (if provided) or summary stats.

    Args:
        histories: {"algo_name": TrainingHistory} mapping.
        snapshot_frames: {"algo_name": RGB frame array} for visual snapshots.
        save_path: If provided, save the dashboard.
        window: Smoothing window for curves.
    """
    names = list(histories.keys())
    n = len(names)
    cols = min(n, 4)
    rows_top = (n + cols - 1) // cols

    fig = plt.figure(figsize=(5 * cols, 4 * (rows_top + 1) + 2))

    # Top rows: training curves
    for i, name in enumerate(names):
        ax = fig.add_subplot(rows_top + 1, cols, i + 1)
        hist = histories[name]
        episodes = np.arange(1, len(hist.episode_rewards) + 1)
        smoothed = _smooth(hist.episode_rewards, window)

        ax.plot(episodes, smoothed, linewidth=2)
        ax.axhline(y=200, color="green", linestyle="--", alpha=0.5)
        ax.set_title(name, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.grid(True, alpha=0.3)

    # Bottom row: snapshots or summary table
    if snapshot_frames and len(snapshot_frames) > 0:
        for i, name in enumerate(names):
            if name in snapshot_frames:
                ax = fig.add_subplot(rows_top + 1, cols, rows_top * cols + i + 1)
                ax.imshow(snapshot_frames[name])
                ax.set_title(f"{name} (snapshot)")
                ax.axis("off")
    else:
        # Summary table
        ax = fig.add_subplot(rows_top + 1, 1, rows_top + 1)
        ax.axis("off")

        table_data = []
        for name in names:
            hist = histories[name]
            rewards = hist.episode_rewards
            smoothed = _smooth(rewards, 100)
            reached = np.where(smoothed >= 200)[0]
            conv_ep = int(reached[0]) + 1 if len(reached) > 0 else "N/A"
            table_data.append([
                name,
                f"{np.mean(rewards[-100:]):.1f}",
                f"{np.std(rewards[-100:]):.1f}",
                str(conv_ep),
                f"{np.max(rewards):.1f}",
            ])

        table = ax.table(
            cellText=table_data,
            colLabels=["Algorithm", "Mean (last 100)", "Std", "Eps to 200", "Max Reward"],
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        ax.set_title("Summary", fontweight="bold", pad=20)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    plt.close(fig)
