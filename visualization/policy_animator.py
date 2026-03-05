"""Record and render agent behavior as MP4/GIF with stat overlays."""

from __future__ import annotations

import os
from pathlib import Path

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from core.base_agent import BaseAgent
from environment.lunar_lander import LunarLanderEnv


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn text lines onto the top-left corner of an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        # Draw shadow then text for readability
        draw.text((11, y + 1), line, fill=(0, 0, 0))
        draw.text((10, y), line, fill=(255, 255, 255))
        y += 18
    return np.array(img)


def animate_agent(
    agent: BaseAgent,
    env: LunarLanderEnv | None = None,
    n_episodes: int = 3,
    output_path: str = "results/videos/",
    filename: str = "policy",
    fps: int = 30,
    max_steps: int = 1000,
) -> list[str]:
    """Record episodes as MP4 files with real-time stat overlays.

    Args:
        agent: Trained agent (will be set to eval mode).
        env: Environment instance. If None, creates one with rgb_array render.
        n_episodes: Number of episodes to record.
        output_path: Directory to save videos.
        filename: Base filename (episode number appended).
        fps: Frames per second.
        max_steps: Max steps per episode.

    Returns:
        List of saved file paths.
    """
    own_env = env is None
    if own_env:
        env = LunarLanderEnv(render_mode="rgb_array")

    os.makedirs(output_path, exist_ok=True)
    agent.set_eval_mode()
    saved_paths = []

    for ep in range(n_episodes):
        obs, state_info = env.reset()
        frames = []
        total_reward = 0.0

        for step in range(max_steps):
            frame = env.render_frame()
            action = agent.select_action(obs)
            action_name = LunarLanderEnv.ACTION_NAMES.get(action, "?")

            obs, reward, terminated, truncated, state_info, _ = env.step(action)
            total_reward += reward

            overlay_lines = [
                f"Step: {step + 1}",
                f"Reward: {total_reward:.1f}",
                f"Action: {action_name}",
                f"Speed: {state_info.speed:.2f}",
                f"Dist: {state_info.distance_to_target:.2f}",
            ]
            frame = _overlay_text(frame, overlay_lines)
            frames.append(frame)

            if terminated or truncated:
                break

        out_file = os.path.join(output_path, f"{filename}_ep{ep + 1}.mp4")
        imageio.mimwrite(out_file, frames, fps=fps)
        saved_paths.append(out_file)

    agent.set_train_mode()
    if own_env:
        env.close()

    return saved_paths


def compare_agents_side_by_side(
    agents_dict: dict[str, BaseAgent],
    env: LunarLanderEnv | None = None,
    output_path: str = "results/videos/comparison.mp4",
    fps: int = 30,
    max_steps: int = 1000,
) -> str:
    """Render multiple agents side-by-side in a single video.

    Args:
        agents_dict: {"algo_name": agent} mapping.
        env: Shared environment (will be re-created per agent).
        output_path: Where to save the combined video.
        fps: Frames per second.
        max_steps: Max steps per episode.

    Returns:
        Path to saved video.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_runs: dict[str, list[np.ndarray]] = {}

    for name, agent in agents_dict.items():
        render_env = LunarLanderEnv(render_mode="rgb_array")
        agent.set_eval_mode()
        obs, _ = render_env.reset(seed=42)
        frames = []

        for step in range(max_steps):
            frame = render_env.render_frame()
            frame = _overlay_text(frame, [name, f"Step {step + 1}"])
            frames.append(frame)
            action = agent.select_action(obs)
            obs, _, terminated, truncated, _, _ = render_env.step(action)
            if terminated or truncated:
                break

        agent.set_train_mode()
        render_env.close()
        all_runs[name] = frames

    # Pad all to same length
    max_len = max(len(f) for f in all_runs.values())
    for name in all_runs:
        while len(all_runs[name]) < max_len:
            all_runs[name].append(all_runs[name][-1])

    # Stack side-by-side (horizontal)
    names = list(all_runs.keys())
    combined_frames = []
    for i in range(max_len):
        row_frames = [all_runs[n][i] for n in names]
        # Ensure same height
        h = min(f.shape[0] for f in row_frames)
        w_total = sum(f.shape[1] for f in row_frames)
        row_frames = [f[:h] for f in row_frames]
        combined = np.concatenate(row_frames, axis=1)
        combined_frames.append(combined)

    imageio.mimwrite(output_path, combined_frames, fps=fps)
    return output_path
