"""CLI for comparing training histories of multiple algorithms.

Usage:
    uv run python scripts/compare.py --algos q_learning dqn reinforce a2c ppo
"""

from __future__ import annotations

import os
import pickle
import sys

import typer
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.trainer import TrainingHistory
from visualization.training_curves import compare_training_curves, plot_convergence_metrics
from visualization.dashboard import create_dashboard

console = Console()
app = typer.Typer(help="Compare training histories of multiple algorithms")


@app.command()
def compare(
    algos: list[str] = typer.Option(..., "--algos", "-a", help="Algorithms to compare"),
    history_dir: str = typer.Option("results/plots", "--history-dir", help="Directory with saved histories"),
    save_dir: str = typer.Option("results/plots", "--save-dir", help="Directory to save comparison plots"),
    dashboard: bool = typer.Option(True, "--dashboard/--no-dashboard", help="Create dashboard"),
) -> None:
    """Compare training curves and convergence metrics."""
    histories: dict[str, TrainingHistory] = {}

    for algo in algos:
        path = os.path.join(history_dir, f"{algo}_history.pkl")
        if not os.path.exists(path):
            console.print(f"[yellow]Warning: No history found for {algo} at {path}[/yellow]")
            continue

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, TrainingHistory):
            histories[algo] = data
        elif isinstance(data, dict) and "episode_rewards" in data:
            # Handle A3C or other formats that save plain dicts
            hist = TrainingHistory()
            hist.episode_rewards = data["episode_rewards"]
            hist.episode_lengths = data.get("episode_lengths", [0] * len(hist.episode_rewards))
            histories[algo] = hist
        else:
            console.print(f"[yellow]Warning: Unrecognized format for {algo}[/yellow]")

    if not histories:
        console.print("[red]No valid histories found. Train some algorithms first![/red]")
        raise typer.Exit(1)

    console.print(f"[green]Comparing {len(histories)} algorithms: {', '.join(histories.keys())}[/green]")

    os.makedirs(save_dir, exist_ok=True)

    # Training curves comparison
    curves_path = os.path.join(save_dir, "comparison_curves.png")
    compare_training_curves(histories, save_path=curves_path)
    console.print(f"  Training curves saved to {curves_path}")

    # Convergence metrics
    convergence_path = os.path.join(save_dir, "convergence_metrics.png")
    plot_convergence_metrics(histories, save_path=convergence_path)
    console.print(f"  Convergence metrics saved to {convergence_path}")

    # Dashboard
    if dashboard:
        dash_path = os.path.join(save_dir, "dashboard.png")
        create_dashboard(histories, save_path=dash_path)
        console.print(f"  Dashboard saved to {dash_path}")


if __name__ == "__main__":
    app()
