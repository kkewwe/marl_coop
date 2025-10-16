from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, lr: float = 1e-3, gamma: float = 0.99,
                 eps_start: float = 1.0, eps_end: float = 0.05, eps_decay: int = 20000, device: str = "cpu"):
        self.q = MLP(obs_dim, n_actions).to(device)
        self.target = MLP(obs_dim, n_actions).to(device)
        self.target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=lr)
        self.gamma = gamma
        self.n_actions = n_actions
        self.device = device
        self.steps = 0
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

    def epsilon(self):
        frac = min(1.0, self.steps / float(self.eps_decay))
        return self.eps_start + (self.eps_end - self.eps_start) * frac

    def act(self, obs):
        import random
        import numpy as np
        self.steps += 1
        if random.random() < self.epsilon():
            return random.randrange(self.n_actions)
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q(x)
            return int(q.argmax(dim=1).item())

    def update(self, batch):
        obs = torch.tensor(batch.obs, dtype=torch.float32, device=self.device)
        act = torch.tensor(batch.action, dtype=torch.long, device=self.device)
        rew = torch.tensor(batch.reward, dtype=torch.float32, device=self.device)
        nxt = torch.tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        done = torch.tensor(batch.done, dtype=torch.float32, device=self.device)
        q = self.q(obs).gather(1, act.view(-1,1)).squeeze(1)
        with torch.no_grad():
            nxt_q = self.target(nxt).max(dim=1)[0]
            tgt = rew + (1.0 - done) * self.gamma * nxt_q
        loss = F.smooth_l1_loss(q, tgt)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), 5.0)
        self.optim.step()
        return float(loss.item())

    def hard_update(self):
        self.target.load_state_dict(self.q.state_dict())