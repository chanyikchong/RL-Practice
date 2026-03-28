"""Record and render agent behavior as MP4/GIF with stat overlays."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile

import imageio
import numpy as np
from PIL import Image, ImageDraw

from core.base_agent import BaseAgent
from environment.lunar_lander import LunarLanderEnv


def _overlay_text(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    """Burn text lines onto the top-left corner of an RGB frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    y = 10
    for line in lines:
        draw.text((11, y + 1), line, fill=(0, 0, 0))
        draw.text((10, y), line, fill=(255, 255, 255))
        y += 18
    return np.array(img)


def _write_mp4(path: str, frames: list[np.ndarray], fps: int = 30) -> None:
    """Write frames to a widely-compatible MP4 via the system ffmpeg.

    Pipes raw RGB frames into ffmpeg which encodes with libx264, yuv420p
    pixel format, and faststart flag for immediate playback.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    h, w = frames[0].shape[:2]
    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "23",
        "-movflags", "+faststart",
        path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    for frame in frames:
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    proc.wait()


def animate_agent(
    agent: BaseAgent,
    env: LunarLanderEnv | None = None,
    n_episodes: int = 3,
    output_path: str = "results/videos/",
    filename: str = "policy",
    fps: int = 30,
    max_steps: int = 1000,
) -> str:
    """Record multiple episodes into a single MP4 with stat overlays.

    All episodes are concatenated into one video. A title frame is shown
    between episodes indicating the episode number and final reward.

    Args:
        agent: Trained agent (will be set to eval mode).
        env: Environment instance. If None, creates one with rgb_array render.
        n_episodes: Number of episodes to record.
        output_path: Directory to save videos.
        filename: Base filename.
        fps: Frames per second.
        max_steps: Max steps per episode.

    Returns:
        Path to the saved video file.
    """
    own_env = env is None
    if own_env:
        env = LunarLanderEnv(render_mode="rgb_array")

    os.makedirs(output_path, exist_ok=True)
    agent.set_eval_mode()

    all_frames: list[np.ndarray] = []

    for ep in range(n_episodes):
        obs, state_info = env.reset()
        ep_frames: list[np.ndarray] = []
        total_reward = 0.0

        for step in range(max_steps):
            frame = env.render_frame()
            action = agent.select_action(obs)
            action_name = LunarLanderEnv.ACTION_NAMES.get(action, "?")

            obs, reward, terminated, truncated, state_info, _ = env.step(action)
            total_reward += reward

            overlay_lines = [
                f"Episode {ep + 1}/{n_episodes}",
                f"Step: {step + 1}",
                f"Reward: {total_reward:.1f}",
                f"Action: {action_name}",
                f"Speed: {state_info.speed:.2f}",
                f"Dist: {state_info.distance_to_target:.2f}",
            ]
            frame = _overlay_text(frame, overlay_lines)
            ep_frames.append(frame)

            if terminated or truncated:
                break

        # Add a brief separator between episodes (1 second of the last frame
        # with the final reward displayed)
        if ep_frames:
            last = ep_frames[-1].copy()
            h, w = last.shape[:2]
            img = Image.fromarray(last)
            draw = ImageDraw.Draw(img)
            label = f"Episode {ep + 1} done  |  Reward: {total_reward:.1f}"
            draw.rectangle([(0, h // 2 - 15), (w, h // 2 + 15)], fill=(0, 0, 0))
            draw.text((w // 2 - len(label) * 3, h // 2 - 7), label, fill=(255, 255, 255))
            separator = np.array(img)
            ep_frames.extend([separator] * fps)  # 1 second pause

        all_frames.extend(ep_frames)

    out_file = os.path.join(output_path, f"{filename}.mp4")
    _write_mp4(out_file, all_frames, fps=fps)

    agent.set_train_mode()
    if own_env:
        env.close()

    return out_file


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
        h = min(f.shape[0] for f in row_frames)
        row_frames = [f[:h] for f in row_frames]
        combined = np.concatenate(row_frames, axis=1)
        combined_frames.append(combined)

    _write_mp4(output_path, combined_frames, fps=fps)
    return output_path
