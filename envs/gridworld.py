from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import random
import numpy as np

Action = int  # 0: stay, 1: up, 2: right, 3: down, 4: left

@dataclass
class GridConfig:
    width: int = 7
    height: int = 7
    n_predators: int = 2
    max_steps: int = 100
    wall_prob: float = 0.0
    partial_obs_radius: int | None = None  # None => full obs; else local window radius


class MultiAgentPredatorPrey:
    """
    Synchronous predator–prey gridworld.
    Observations: flattened planes (predators, prey, walls), full or egocentric crop.
    Rewards: individual + team via outer experiment mixing.
    Capture: any predator on prey cell ends episode with team reward 1.
    """
    def __init__(self, cfg: GridConfig):
        self.prey_captured = False
        self.cfg = cfg
        self.n_agents = cfg.n_predators
        self.action_space_n = 5
        self.step_count = 0
        self.grid_walls = None
        self.pred_pos: List[Tuple[int,int]] = []
        self.prey_pos: Tuple[int,int] | None = None

    def _empty_grid(self):
        w, h = self.cfg.width, self.cfg.height
        walls = np.zeros((h, w), dtype=np.int32)
        if self.cfg.wall_prob > 0:
            for y in range(h):
                for x in range(w):
                    if (x,y) not in [(0,0), (w-1,h-1)] and random.random() < self.cfg.wall_prob:
                        walls[y,x] = 1
        return walls

    def _random_empty_cell(self):
        h, w = self.cfg.height, self.cfg.width
        while True:
            x, y = random.randrange(w), random.randrange(h)
            if self.grid_walls[y,x] == 0 and (x,y) not in self.pred_pos and (self.prey_pos is None or (x,y) != self.prey_pos):
                return (x,y)

    def reset(self) -> Dict[str, np.ndarray]:
        self.step_count = 0
        self.grid_walls = self._empty_grid()
        self.pred_pos = [self._random_empty_cell() for _ in range(self.n_agents)]
        self.prey_pos = self._random_empty_cell()
        self.prey_captured = False
        return self._get_obs()

    def step(self, actions: Dict[str, Action]):
        self.step_count += 1
        # move predators
        new_pred_pos = []
        for i in range(self.n_agents):
            x,y = self.pred_pos[i]
            a = actions[f"agent_{i}"]
            nx, ny = x, y
            if a == 1: ny = max(0, y-1)
            elif a == 2: nx = min(self.cfg.width-1, x+1)
            elif a == 3: ny = min(self.cfg.height-1, y+1)
            elif a == 4: nx = max(0, x-1)
            if self.grid_walls[ny, nx] == 0:
                new_pred_pos.append((nx, ny))
            else:
                new_pred_pos.append((x,y))

        # prey: move away from nearest predator (greedy Chebyshev/Mahattan)
        px, py = self.prey_pos
        def dist(a,b):
            return abs(a[0]-b[0]) + abs(a[1]-b[1])
        nearest = min(new_pred_pos, key=lambda p: dist(p, (px,py)))
        best = (px,py)
        best_d = dist(nearest, (px,py))
        for dx,dy in [(0,0),(0,-1),(1,0),(0,1),(-1,0)]:
            nx, ny = px+dx, py+dy
            if 0 <= nx < self.cfg.width and 0 <= ny < self.cfg.height and self.grid_walls[ny,nx]==0:
                d = dist(nearest, (nx,ny))
                if d > best_d:
                    best = (nx,ny); best_d = d
        new_prey_pos = best

        self.pred_pos = new_pred_pos
        self.prey_pos = new_prey_pos

        # rewards and termination
        done = False
        individual_rewards = [0.0 for _ in range(self.n_agents)]
        team_reward = 0.0
        for i,(x,y) in enumerate(self.pred_pos):
            if (x,y) == self.prey_pos:
                done = True
                team_reward = 1.0
                self.prey_captured = True
                individual_rewards[i] = 1.0
                break
        if self.step_count >= self.cfg.max_steps:
            done = True

        obs = self._get_obs()
        info = {"team_reward": team_reward}
        return obs, {f"agent_{i}": individual_rewards[i] for i in range(self.n_agents)}, done, info

    def _get_obs(self) -> Dict[str, np.ndarray]:
        h, w = self.cfg.height, self.cfg.width
        planes = np.zeros((3, h, w), dtype=np.float32)
        for (x,y) in self.pred_pos:
            planes[0, y, x] = 1.0
        if self.prey_pos:
            x,y = self.prey_pos
            planes[1, y, x] = 1.0
        planes[2,:,:] = self.grid_walls

        obs: Dict[str, np.ndarray] = {}
        for i,(x,y) in enumerate(self.pred_pos):
            if self.cfg.partial_obs_radius is None:
                o = planes.copy()
            else:
                r = self.cfg.partial_obs_radius
                x0, x1 = max(0, x-r), min(w, x+r+1)
                y0, y1 = max(0, y-r), min(h, y+r+1)
                crop = planes[:, y0:y1, x0:x1]
                pad_h = 2*r+1 - crop.shape[1]
                pad_w = 2*r+1 - crop.shape[2]
                o = np.pad(crop, ((0,0),(0,pad_h),(0,pad_w)))
            obs[f"agent_{i}"] = o.reshape(-1)
        return obs

    def obs_dim(self) -> int:
        if self.cfg.partial_obs_radius is None:
            return 3 * self.cfg.height * self.cfg.width
        side = 2*self.cfg.partial_obs_radius + 1
        return 3 * side * side
