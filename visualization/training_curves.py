"""Plotting utilities for training histories."""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.trainer import TrainingHistory


def _smooth(data: list[float], window: int = 50) -> np.ndarray:
    """Compute rolling mean with given window."""
    series = pd.Series(data)
    return series.rolling(window=window, min_periods=1).mean().to_numpy()


def _smooth_std(data: list[float], window: int = 50) -> np.ndarray:
    """Compute rolling std with given window."""
    series = pd.Series(data)
    return series.rolling(window=window, min_periods=1).std().to_numpy()


def plot_training_history(
    history: TrainingHistory,
    title: str = "Training Progress",
    save_path: str | None = None,
    window: int = 50,
) -> None:
    """Plot smoothed reward curve with std band.

    Args:
        history: TrainingHistory from trainer.
        title: Plot title.
        save_path: If provided, save figure to this path.
        window: Smoothing window for rolling mean/std.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    episodes = np.arange(1, len(history.episode_rewards) + 1)
    raw = np.array(history.episode_rewards)
    smoothed = _smooth(history.episode_rewards, window)
    std = _smooth_std(history.episode_rewards, window)

    ax.plot(episodes, raw, alpha=0.2, color="blue", label="Raw")
    ax.plot(episodes, smoothed, color="blue", linewidth=2, label=f"Mean ({window}-ep)")
    ax.fill_between(episodes, smoothed - std, smoothed + std, alpha=0.15, color="blue")

    ax.axhline(y=200, color="green", linestyle="--", alpha=0.5, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def compare_training_curves(
    histories: dict[str, TrainingHistory],
    save_path: str | None = None,
    window: int = 50,
) -> None:
    """Overlay smoothed training curves for multiple algorithms.

    Args:
        histories: {"algo_name": TrainingHistory} mapping.
        save_path: If provided, save figure.
        window: Smoothing window.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories)))

    for (name, hist), color in zip(histories.items(), colors):
        episodes = np.arange(1, len(hist.episode_rewards) + 1)
        smoothed = _smooth(hist.episode_rewards, window)
        std = _smooth_std(hist.episode_rewards, window)

        ax.plot(episodes, smoothed, label=name, color=color, linewidth=2)
        ax.fill_between(episodes, smoothed - std, smoothed + std, alpha=0.1, color=color)

    ax.axhline(y=200, color="green", linestyle="--", alpha=0.5, label="Solved (200)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Algorithm Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)


def plot_convergence_metrics(
    histories: dict[str, TrainingHistory],
    save_path: str | None = None,
    threshold: float = 200.0,
    window: int = 100,
) -> None:
    """Plot convergence analysis: episodes to threshold, AUC, final performance.

    Args:
        histories: {"algo_name": TrainingHistory} mapping.
        save_path: If provided, save figure.
        threshold: Reward threshold for "solved".
        window: Window for computing rolling mean.
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    names = list(histories.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))

    # 1. Episodes to reach threshold
    convergence_eps = []
    for name in names:
        smoothed = _smooth(histories[name].episode_rewards, window)
        reached = np.where(smoothed >= threshold)[0]
        convergence_eps.append(int(reached[0]) + 1 if len(reached) > 0 else len(smoothed))

    axes[0].barh(names, convergence_eps, color=colors)
    axes[0].set_xlabel("Episodes to Reach 200")
    axes[0].set_title("Convergence Speed")
    axes[0].invert_yaxis()

    # 2. Area under curve (sample efficiency)
    aucs = []
    for name in names:
        rewards = np.array(histories[name].episode_rewards)
        aucs.append(float(np.sum(rewards)))

    axes[1].barh(names, aucs, color=colors)
    axes[1].set_xlabel("Cumulative Reward (AUC)")
    axes[1].set_title("Sample Efficiency")
    axes[1].invert_yaxis()

    # 3. Final performance box plots
    final_data = []
    for name in names:
        final_data.append(histories[name].episode_rewards[-100:])

    bp = axes[2].boxplot(final_data, labels=names, vert=True, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Final Performance (last 100 eps)")
    axes[2].axhline(y=200, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150)
    plt.show()
    plt.close(fig)
