from __future__ import annotations
from typing import List
import numpy as np

def action_entropy(action_history: List[int], n_actions: int) -> float:
    if len(action_history) == 0:
        return 0.0
    counts = np.bincount(np.array(action_history), minlength=n_actions)
    p = counts / counts.sum() if counts.sum() > 0 else np.ones(n_actions) / n_actions
    nz = p[p>0]
    return float(-(nz * np.log(nz)).sum())

def coordination_index(actions_a: List[int], actions_b: List[int]) -> float:
    n = min(len(actions_a), len(actions_b))
    if n == 0:
        return 0.0
    agree = sum(1 for i in range(n) if actions_a[i] == actions_b[i])
    return agree / n
