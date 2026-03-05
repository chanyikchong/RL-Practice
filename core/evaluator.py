"""Evaluation loop — runs an agent's greedy policy and reports metrics."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table

from core.base_agent import BaseAgent
from environment.lunar_lander import LunarLanderEnv

console = Console()


@dataclass
class EvalResult:
    """Aggregated evaluation results."""
    rewards: list[float] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if self.rewards else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    @property
    def solved(self) -> bool:
        """LunarLander is solved if mean reward >= 200."""
        return self.mean_reward >= 200.0


class Evaluator:
    """Runs greedy evaluation of a trained agent.

    Args:
        agent: A trained BaseAgent.
        env: A LunarLanderEnv instance.
        max_steps: Maximum steps per episode.
    """

    def __init__(self, agent: BaseAgent, env: LunarLanderEnv, max_steps: int = 1000):
        self.agent = agent
        self.env = env
        self.max_steps = max_steps

    def evaluate(self, n_episodes: int = 100) -> EvalResult:
        """Run ``n_episodes`` with greedy policy and return results."""
        self.agent.set_eval_mode()
        result = EvalResult()

        for ep in range(n_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            for step in range(self.max_steps):
                action = self.agent.select_action(obs)
                obs, reward, terminated, truncated, _, _ = self.env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            result.rewards.append(total_reward)
            result.lengths.append(step + 1)

        self.agent.set_train_mode()
        return result

    def print_results(self, result: EvalResult, algo_name: str = "Agent") -> None:
        """Pretty-print evaluation results to terminal."""
        table = Table(title=f"Evaluation Results: {algo_name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Episodes", str(len(result.rewards)))
        table.add_row("Mean Reward", f"{result.mean_reward:.1f}")
        table.add_row("Std Reward", f"{result.std_reward:.1f}")
        table.add_row("Mean Length", f"{result.mean_length:.1f}")
        table.add_row("Min Reward", f"{min(result.rewards):.1f}")
        table.add_row("Max Reward", f"{max(result.rewards):.1f}")
        status = "[bold green]SOLVED[/bold green]" if result.solved else "[bold red]NOT SOLVED[/bold red]"
        table.add_row("Solved (>=200)?", status)

        console.print(table)
