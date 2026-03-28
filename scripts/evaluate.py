"""CLI for evaluating a trained RL agent.

Usage:
    uv run python scripts/evaluate.py --algo dqn --checkpoint results/checkpoints/dqn_final.pt
"""

from __future__ import annotations

import os
import sys

import typer
from rich.console import Console

from .utils import HYPER_PARAM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import LunarLanderEnv
from core.evaluator import Evaluator

console = Console()
app = typer.Typer(help="Evaluate a trained RL agent")

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
def evaluate(
    algo: str = typer.Option(..., "--algo", "-a", help="Algorithm name"),
    checkpoint: str = typer.Option(..., "--checkpoint", "-c", help="Path to checkpoint"),
    episodes: int = typer.Option(100, "--episodes", "-e", help="Number of eval episodes"),
    use_solution: bool = typer.Option(False, "--use-solution", "-s", help="Use reference solution"),
) -> None:
    """Evaluate a trained agent over multiple episodes."""
    import importlib

    registry = SOLUTION_REGISTRY if use_solution else ALGO_REGISTRY
    if algo not in registry:
        console.print(f"[red]Unknown algorithm: {algo}[/red]")
        raise typer.Exit(1)

    module_path, class_name = registry[algo]
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)
    agent = agent_class(**HYPER_PARAM[algo])

    console.print(f"Loading checkpoint: {checkpoint}")
    agent.load(checkpoint)

    env = LunarLanderEnv()
    evaluator = Evaluator(agent, env)
    result = evaluator.evaluate(episodes)
    evaluator.print_results(result, algo)

    env.close()


if __name__ == "__main__":
    app()
