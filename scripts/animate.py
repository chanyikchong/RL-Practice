"""CLI for recording agent behavior as video.

Usage:
    uv run python scripts/animate.py --algo ppo --episodes 5
    uv run python scripts/animate.py --algo ppo --checkpoint results/checkpoints/ppo_final.pt
"""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import LunarLanderEnv
from visualization.policy_animator import animate_agent

from utils import HYPER_PARAM, SOLUTION_HYPER_PARAM


console = Console()
app = typer.Typer(help="Record agent behavior as MP4 video")

ALGO_REGISTRY = {
    "q_learning": ("algorithms.q_learning.agent", "QLearningAgent"),
    "dqn": ("algorithms.dqn.agent", "DQNAgent"),
    "double_dqn": ("algorithms.double_dqn.agent", "DoubleDQNAgent"),
    "reinforce": ("algorithms.reinforce.agent", "REINFORCEAgent"),
    "actor_critic": ("algorithms.actor_critic.agent", "ActorCriticAgent"),
    "a2c": ("algorithms.a2c.agent", "A2CAgent"),
    "a3c": ("algorithms.a3c.agent", "A3CAgent"),
    "ppo": ("algorithms.ppo.agent", "PPOAgent"),
}

SOLUTION_REGISTRY = {
    "q_learning": ("solutions.q_learning.agent", "QLearningAgent"),
    "dqn": ("solutions.dqn.agent", "DQNAgent"),
    "reinforce": ("solutions.reinforce.agent", "REINFORCEAgent"),
    "actor_critic": ("solutions.actor_critic.agent", "ActorCriticAgent"),
    "a2c": ("solutions.a2c.agent", "A2CAgent"),
    "a3c": ("solutions.a3c.agent", "A3CAgent"),
    "ppo": ("solutions.ppo.agent", "PPOAgent"),
}


@app.command()
def animate(
    algo: str = typer.Option(..., "--algo", "-a", help="Algorithm name"),
    checkpoint: str = typer.Option(None, "--checkpoint", "-c", help="Checkpoint path (default: latest)"),
    episodes: int = typer.Option(3, "--episodes", "-e", help="Number of episodes to record"),
    use_solution: bool = typer.Option(False, "--use-solution", "-s", help="Use reference solution"),
    output_dir: str = typer.Option("results/videos", "--output-dir", "-o", help="Output directory"),
) -> None:
    """Record agent playing LunarLander as MP4 video."""
    import importlib

    registry = SOLUTION_REGISTRY if use_solution else ALGO_REGISTRY
    hyper_params = SOLUTION_HYPER_PARAM if use_solution else HYPER_PARAM

    if algo not in registry:
        console.print(f"[red]Unknown algorithm: {algo}[/red]")
        raise typer.Exit(1)

    module_path, class_name = registry[algo]
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)
    agent = agent_class(**hyper_params[algo])

    if checkpoint is None:
        checkpoint = f"results/checkpoints/{algo}_final.pt"

    if os.path.exists(checkpoint):
        console.print(f"Loading checkpoint: {checkpoint}")
        agent.load(checkpoint)
    else:
        console.print(f"[yellow]No checkpoint found at {checkpoint}, using untrained agent[/yellow]")

    env = LunarLanderEnv(render_mode="rgb_array")
    console.print(f"Recording {episodes} episodes...")

    path = animate_agent(
        agent, env,
        n_episodes=episodes,
        output_path=output_dir,
        filename=algo,
    )

    console.print(f"  Saved: {path}")

    env.close()


if __name__ == "__main__":
    app()
