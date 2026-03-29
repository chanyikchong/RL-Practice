"""Reusable training loop for any BaseAgent on LunarLanderEnv."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
from torch.utils.tensorboard import SummaryWriter

from core.base_agent import BaseAgent
from environment.lunar_lander import LunarLanderEnv

console = Console()


@dataclass
class EpisodeResult:
    """Result of a single training episode."""
    total_reward: float
    length: int
    agent_metrics: dict = field(default_factory=dict)


@dataclass
class TrainingHistory:
    """Stores the full training history for later analysis / plotting."""
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    agent_metrics: list[dict] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    eval_rewards: list[tuple[int, float]] = field(default_factory=list)  # (episode, avg_reward)

    def mean_reward(self, last_n: int = 100) -> float:
        if not self.episode_rewards:
            return 0.0
        return float(np.mean(self.episode_rewards[-last_n:]))


DEFAULT_CONFIG = {
    "max_steps_per_episode": 1000,
    "eval_every_n_episodes": 50,
    "eval_episodes": 10,
    "save_checkpoint_every": 100,
    "log_to_tensorboard": True,
    "render_every_n_episodes": 0,
    "checkpoint_dir": "results/checkpoints",
    "tensorboard_dir": "results/tensorboard",
    "algo_name": "agent",
}


class Trainer:
    """Generic training loop. Handles rollouts, logging, and checkpointing.

    The learner never needs to write a training loop — just implement a
    BaseAgent and pass it here.

    Args:
        agent: Any agent implementing BaseAgent.
        env: A LunarLanderEnv instance.
        config: Training configuration dict. See DEFAULT_CONFIG for keys.
    """

    def __init__(self, agent: BaseAgent, env: LunarLanderEnv, config: dict | None = None):
        self.agent = agent
        self.env = env
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.history = TrainingHistory()
        self._writer: SummaryWriter | None = None

    def train(self, num_episodes: int) -> TrainingHistory:
        """Run the full training loop for ``num_episodes``.

        Returns:
            TrainingHistory with all recorded metrics.
        """
        cfg = self.config
        algo = cfg["algo_name"]

        # Setup directories
        os.makedirs(cfg["checkpoint_dir"], exist_ok=True)

        # Tensorboard
        if cfg["log_to_tensorboard"]:
            tb_dir = os.path.join(cfg["tensorboard_dir"], algo)
            os.makedirs(tb_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=tb_dir)

        console.print(f"[bold green]Training {algo} for {num_episodes} episodes[/bold green]")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(f"Training {algo}", total=num_episodes)

            for ep in range(1, num_episodes + 1):
                result = self._run_episode()
                self.history.episode_rewards.append(result.total_reward)
                self.history.episode_lengths.append(result.length)
                self.history.agent_metrics.append(result.agent_metrics)
                self.history.timestamps.append(time.time())

                # Tensorboard logging
                if self._writer:
                    self._writer.add_scalar("reward/episode", result.total_reward, ep)
                    self._writer.add_scalar("reward/mean_100", self.history.mean_reward(100), ep)
                    self._writer.add_scalar("episode/length", result.length, ep)
                    for k, v in result.agent_metrics.items():
                        if isinstance(v, (int, float)):
                            self._writer.add_scalar(f"agent/{k}", v, ep)

                # Periodic evaluation
                if cfg["eval_every_n_episodes"] > 0 and ep % cfg["eval_every_n_episodes"] == 0:
                    avg = self._evaluate(cfg["eval_episodes"])
                    self.history.eval_rewards.append((ep, avg))
                    console.print(
                        f"  [cyan]Ep {ep}[/cyan] | "
                        f"Mean reward (100): {self.history.mean_reward(100):.1f} | "
                        f"Eval ({cfg['eval_episodes']} eps): {avg:.1f}"
                    )

                # Checkpointing
                if cfg["save_checkpoint_every"] > 0 and ep % cfg["save_checkpoint_every"] == 0:
                    ckpt_path = os.path.join(cfg["checkpoint_dir"], f"{algo}_ep{ep}.pt")
                    self.agent.save(ckpt_path)

                progress.advance(task)

        # Final checkpoint
        final_path = os.path.join(cfg["checkpoint_dir"], f"{algo}_final.pt")
        self.agent.save(final_path)
        console.print(f"[bold green]Training complete![/bold green] Final checkpoint: {final_path}")
        console.print(f"  Final mean reward (100 eps): {self.history.mean_reward(100):.1f}")

        if self._writer:
            self._writer.close()

        return self.history

    def _run_episode(self) -> EpisodeResult:
        """Run one episode, calling agent.update() after each step."""
        obs, state_info = self.env.reset()
        total_reward = 0.0
        all_metrics: dict = {}
        max_steps = self.config["max_steps_per_episode"]

        for step in range(max_steps):
            action = self.agent.select_action(obs)
            next_obs, reward, terminated, truncated, state_info, reward_signals = self.env.step(action)
            done = terminated or truncated

            metrics = self.agent.update(
                state=obs,
                action=action,
                reward=reward,
                next_state=next_obs,
                terminated=terminated,
                truncated=truncated,
                done=done,
            )
            if metrics:
                all_metrics = metrics  # keep latest metrics

            total_reward += reward
            obs = next_obs

            if done:
                break

        return EpisodeResult(
            total_reward=total_reward,
            length=step + 1,
            agent_metrics=all_metrics,
        )

    def _evaluate(self, n_episodes: int) -> float:
        """Run n greedy episodes and return average reward."""
        self.agent.set_eval_mode()
        rewards = []
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            total = 0.0
            for _ in range(self.config["max_steps_per_episode"]):
                action = self.agent.select_action(obs)
                obs, reward, terminated, truncated, _, _ = self.env.step(action)
                total += reward
                if terminated or truncated:
                    break
            rewards.append(total)
        self.agent.set_train_mode()
        return float(np.mean(rewards))
