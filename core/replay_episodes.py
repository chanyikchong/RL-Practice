from collections import deque
from dataclasses import dataclass
import random
from typing import List, Dict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Episode:
    """A single episode"""
    states: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    gts: torch.Tensor


class ReplayEpisodes:
    def __init__(self, capacity: int = 100_000, seed: int | None = None):
        self._buffer: deque = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._buffer)

    def push(self, states: List[np.ndarray], actions: List[int], rewards: List[float], gts: List[float]) -> None:
        self._buffer.append(
            Episode(
                torch.tensor(np.stack(states, axis=0)),  # [T, dim_state]
                torch.tensor(actions),     # [T]
                torch.tensor(rewards),     # [T]
                torch.tensor(gts)          # [T]
            )
        )

    def sample(self, batch_size: int, use_pad: bool = False) -> Dict[str, torch.Tensor]:
        episodes = self._rng.sample(list(self._buffer), batch_size - 1)
        episodes.append(self._buffer[-1])
        if use_pad:
            return self.pad_trajectory(episodes)

        states = torch.cat([ep.states for ep in episodes], dim=0)
        actions = torch.cat([ep.actions for ep in episodes], dim=0)
        gts = torch.cat([ep.gts for ep in episodes], dim=0)
        lengths = torch.tensor([len(ep.actions) for ep in episodes])
        samples = {
            "states": states,
            "actions": actions,
            "gts": gts,
            "lengths": lengths,
        }
        return samples

    @staticmethod
    def pad_trajectory(episodes: List[Episode]) -> Dict[str, torch.Tensor]:
        states_list = [ep.states for ep in episodes]
        actions_list = [ep.actions for ep in episodes]
        gts_list = [ep.gts for ep in episodes]

        lengths = torch.tensor([len(x) for x in states_list])
        states_padded = pad_sequence(states_list, batch_first=True)
        actions_padded = pad_sequence(actions_list, batch_first=True, padding_value=0)
        gt_padded = pad_sequence(gts_list, batch_first=True, padding_value=0)
        B, T = actions_padded.shape
        mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)
        return {
            "states": states_padded,
            "actions": actions_padded,
            "gts": gt_padded,
            "lengths": lengths,
            "mask": mask,
        }
