from __future__ import annotations
from dataclasses import dataclass
from typing import List
import random
import numpy as np

@dataclass
class Transition:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Transition] = []
        self.idx = 0
    def __len__(self):
        return len(self.storage)
    def add(self, tr: Transition):
        if len(self.storage) < self.capacity:
            self.storage.append(tr)
        else:
            self.storage[self.idx] = tr
        self.idx = (self.idx + 1) % self.capacity
    def sample(self, batch_size: int) -> Transition:
        batch = random.sample(self.storage, batch_size)
        import numpy as np
        obs = np.stack([b.obs for b in batch])
        act = np.array([b.action for b in batch], dtype=np.int64)
        rew = np.array([b.reward for b in batch], dtype=np.float32)
        nxt = np.stack([b.next_obs for b in batch])
        done = np.array([b.done for b in batch], dtype=np.float32)
        return Transition(obs, act, rew, nxt, done)