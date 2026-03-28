"""CLI for training an RL agent on LunarLander-v2.

Usage:
    uv run python scripts/train.py --algo dqn --episodes 1000
    uv run python scripts/train.py --algo ppo --episodes 500 --use-solution
"""

from __future__ import annotations

import os
import sys
import pickle

import typer
from rich.console import Console

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment import LunarLanderEnv
from core.trainer import Trainer
from utils import HYPER_PARAM, SOLUTION_HYPER_PARAM


console = Console()
app = typer.Typer(help="Train an RL agent on LunarLander-v2")

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


def _load_agent(algo: str, use_solution: bool):
    """Dynamically import and instantiate an agent."""
    registry = SOLUTION_REGISTRY if use_solution else ALGO_REGISTRY
    hyper_params = SOLUTION_HYPER_PARAM if use_solution else HYPER_PARAM
    if algo not in registry:
        console.print(f"[red]Unknown algorithm: {algo}[/red]")
        console.print(f"Available: {', '.join(registry.keys())}")
        raise typer.Exit(1)

    module_path, class_name = registry[algo]
    import importlib
    module = importlib.import_module(module_path)
    agent_class = getattr(module, class_name)
    return agent_class(**hyper_params[algo])


@app.command()
def train(
    algo: str = typer.Option(..., "--algo", "-a", help="Algorithm name"),
    episodes: int = typer.Option(1000, "--episodes", "-e", help="Number of training episodes"),
    max_step: int = typer.Option(1000, "--max-steps", "-m", help="Maximum number of steps per episode"),
    use_solution: bool = typer.Option(False, "--use-solution", "-s", help="Use reference solution"),
    eval_every: int = typer.Option(50, "--eval-every", help="Evaluate every N episodes"),
    save_every: int = typer.Option(100, "--save-every", help="Save checkpoint every N episodes"),
    tensorboard: bool = typer.Option(True, "--tensorboard/--no-tensorboard", help="Log to TensorBoard"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    save_path: str = typer.Option("results/checkpoints", "--save-path", help="Checkpoint file"),
) -> None:
    """Train an RL agent on LunarLander-v2."""
    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)

    source = "solution" if use_solution else "your implementation"
    console.print(f"[bold]Training {algo} ({source}) for {episodes} episodes[/bold]")

    # Special case for A3C (uses its own parallel training)
    if algo == "a3c":
        agent = _load_agent(algo, use_solution)
        console.print("[yellow]A3C uses parallel training — launching workers...[/yellow]")
        rewards = agent.train_parallel(max_episodes=episodes)
        console.print(f"[green]Done! Collected {len(rewards)} episode rewards.[/green]")
        if rewards:
            import numpy as np
            console.print(f"  Mean reward (last 100): {np.mean(rewards[-100:]):.1f}")
        # Save
        os.makedirs(f"{save_path}/checkpoints", exist_ok=True)
        agent.save(f"{save_path}/checkpoints/{algo}_final.pt")
        # Save history
        os.makedirs(f"{save_path}/plots", exist_ok=True)
        history_path = f"{save_path}/plots/{algo}_history.pkl"
        with open(history_path, "wb") as f:
            pickle.dump({"episode_rewards": rewards}, f)
        return

    agent = _load_agent(algo, use_solution)
    env = LunarLanderEnv()

    config = {
        "algo_name": algo,
        "max_steps_per_episode": max_step,
        "eval_every_n_episodes": eval_every,
        "save_checkpoint_every": save_every,
        "log_to_tensorboard": tensorboard,
        "checkpoint_dir": save_path
    }

    trainer = Trainer(agent, env, config)
    history = trainer.train(episodes)

    # Save training history
    os.makedirs("results/plots", exist_ok=True)
    history_path = f"results/plots/{algo}_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history, f)
    console.print(f"History saved to {history_path}")

    env.close()


if __name__ == "__main__":
    app()
