from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict
import numpy as np

from .envs.gridworld import MultiAgentPredatorPrey, GridConfig
from .agents.dqn import DQNAgent
from .core.replay import ReplayBuffer, Transition
from .metrics import coordination_index
from .utils.seed import set_seed


@dataclass
class TrainConfig:
    episodes: int = 2000
    max_steps: int = 100
    buffer_size: int = 50000
    batch_size: int = 64
    start_training_after: int = 1000
    train_every: int = 4
    target_update_every: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    device: str = "cpu"
    seed: int = 0
    alpha_sweep: bool = False


def train(cfg: TrainConfig, env_cfg: GridConfig):
    set_seed(cfg.seed)
    env = MultiAgentPredatorPrey(env_cfg)
    obs_dim = env.obs_dim()
    n_actions = env.action_space_n

    agents: List[DQNAgent] = [
        DQNAgent(obs_dim, n_actions, lr=cfg.lr, gamma=cfg.gamma, device=cfg.device)
        for _ in range(env.n_agents)
    ]
    bufs: List[ReplayBuffer] = [ReplayBuffer(cfg.buffer_size) for _ in range(env.n_agents)]

    # alpha schedule: 0.00 to 0.99 (100 total values)
    n_alphas = 100
    episodes_per_alpha = max(1, cfg.episodes // n_alphas)
    alpha_values = [i / 100 for i in range(n_alphas)]

    total_steps = 0
    episode_returns = []
    capture_history = []
    coord_scores = []

    # per-alpha tracking
    per_alpha_caps = np.zeros(n_alphas)
    per_alpha_counts = np.zeros(n_alphas)
    per_alpha_returns = np.zeros(n_alphas)
    per_alpha_coord = np.zeros(n_alphas)

    for ep in range(cfg.episodes):
        alpha_idx = min(ep // episodes_per_alpha, n_alphas - 1)
        current_alpha = alpha_values[alpha_idx]

        obs = env.reset()
        ep_team_return = 0.0
        actions_trace = [[] for _ in range(env.n_agents)]

        for _ in range(cfg.max_steps):
            acts = {}
            for i in range(env.n_agents):
                a_i = agents[i].act(obs[f"agent_{i}"])
                acts[f"agent_{i}"] = a_i
                actions_trace[i].append(a_i)

            next_obs, indiv_rewards, done, info = env.step(acts)
            R_team = info.get("team_reward", 0.0)

            for i in range(env.n_agents):
                r_i = indiv_rewards[f"agent_{i}"]
                mixed = current_alpha * r_i + (1.0 - current_alpha) * R_team
                bufs[i].add(Transition(
                    obs[f"agent_{i}"],
                    acts[f"agent_{i}"],
                    mixed,
                    next_obs[f"agent_{i}"],
                    done
                ))

            obs = next_obs
            ep_team_return += R_team
            total_steps += 1

            if total_steps > cfg.start_training_after and total_steps % cfg.train_every == 0:
                for i in range(env.n_agents):
                    if len(bufs[i]) >= cfg.batch_size:
                        batch = bufs[i].sample(cfg.batch_size)
                        agents[i].update(batch)
            if total_steps % cfg.target_update_every == 0:
                for i in range(env.n_agents):
                    agents[i].hard_update()

            if done:
                break

        episode_returns.append(ep_team_return)
        capture = 1 if ep_team_return > 0 else 0
        capture_history.append(capture)
        coord = coordination_index(actions_trace[0], actions_trace[1])

        per_alpha_caps[alpha_idx] += capture
        per_alpha_returns[alpha_idx] += ep_team_return
        per_alpha_coord[alpha_idx] += coord
        per_alpha_counts[alpha_idx] += 1
        coord_scores.append(coord)

    # averages per alpha
    avg_cap = per_alpha_caps / np.maximum(1, per_alpha_counts)
    avg_ret = per_alpha_returns / np.maximum(1, per_alpha_counts)
    avg_coord = per_alpha_coord / np.maximum(1, per_alpha_counts)

    results = {
        "alpha_values": np.array(alpha_values),
        "per_alpha_capture_rate": avg_cap,
        "per_alpha_mean_return": avg_ret,
        "per_alpha_mean_coord": avg_coord,
        "episode_returns": np.array(episode_returns),
        "capture_history": np.array(capture_history),
        "coord_scores": np.array(coord_scores),
    }
    return results
