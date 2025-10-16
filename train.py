from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from .envs.gridworld import MultiAgentPredatorPrey, GridConfig
from .agents.dqn import DQNAgent
from .core.replay import ReplayBuffer, Transition
from .metrics import action_entropy, coordination_index
from .utils.seed import set_seed

@dataclass
class TrainConfig:
    episodes: int = 20000
    max_steps: int = 100
    buffer_size: int = 50000
    batch_size: int = 64
    start_training_after: int = 1000
    train_every: int = 4
    target_update_every: int = 1000
    gamma: float = 0.99
    lr: float = 1e-3
    alpha: float = 0.5
    device: str = "cpu"
    log_every: int = 500
    seed: int = 0


def train(cfg: TrainConfig, env_cfg: GridConfig):
    set_seed(cfg.seed)
    env = MultiAgentPredatorPrey(env_cfg)
    obs_dim = env.obs_dim()
    n_actions = env.action_space_n

    agents: List[DQNAgent] = [DQNAgent(obs_dim, n_actions, lr=cfg.lr, gamma=cfg.gamma, device=cfg.device) for _ in range(env.n_agents)]
    bufs: List[ReplayBuffer] = [ReplayBuffer(cfg.buffer_size) for _ in range(env.n_agents)]

    total_steps = 0
    episode_returns: List[float] = []
    reward_variances: List[float] = []
    entropies: List[List[int]] = [[] for _ in range(env.n_agents)]
    coord_scores: List[float] = []

    for ep in range(1, cfg.episodes+1):
        obs = env.reset()
        ep_team_return = 0.0
        actions_trace = [[] for _ in range(env.n_agents)]
        indiv_rewards_trace = [[] for _ in range(env.n_agents)]

        for t in range(env_cfg.max_steps):
            acts = {}
            for i in range(env.n_agents):
                a_i = agents[i].act(obs[f"agent_{i}"])
                acts[f"agent_{i}"] = a_i
                actions_trace[i].append(a_i)

            next_obs, indiv_rewards, done, info = env.step(acts)
            R_team = info.get("team_reward", 0.0)

            mixed_rewards = []
            for i in range(env.n_agents):
                r_i = indiv_rewards[f"agent_{i}"]
                mixed = cfg.alpha * r_i + (1.0 - cfg.alpha) * R_team
                mixed_rewards.append(mixed)
                indiv_rewards_trace[i].append(mixed)

            for i in range(env.n_agents):
                bufs[i].add(Transition(obs[f"agent_{i}"], acts[f"agent_{i}"], mixed_rewards[i], next_obs[f"agent_{i}"], done))

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
        final_rewards = [sum(indiv_rewards_trace[i]) for i in range(env.n_agents)]
        reward_variances.append(float(np.var(final_rewards)))
        for i in range(env.n_agents):
            entropies[i].extend(actions_trace[i])
        coord_scores.append(coordination_index(actions_trace[0], actions_trace[1] if env.n_agents>1 else actions_trace[0]))

        if ep % cfg.log_every == 0:
            import numpy as _np
            ent_vals = [action_entropy(entropies[i], n_actions) for i in range(env.n_agents)]
            mean_entropy = float(_np.mean(ent_vals))
            mean_return = float(_np.mean(episode_returns[-cfg.log_every:]))
            mean_var = float(_np.mean(reward_variances[-cfg.log_every:]))
            mean_coord = float(_np.mean(coord_scores[-cfg.log_every:]))
            print(f"Ep {ep:5d} | team_return(mean last {cfg.log_every}): {mean_return:.3f} | reward_var: {mean_var:.4f} | entropy: {mean_entropy:.3f} | coord: {mean_coord:.3f}")
            entropies = [[] for _ in range(env.n_agents)]

    results = {
        "episode_returns": np.array(episode_returns),
        "reward_variances": np.array(reward_variances),
        "coord_scores": np.array(coord_scores),
    }
    return results