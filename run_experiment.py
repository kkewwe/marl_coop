from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from .envs.gridworld import GridConfig
from .train import TrainConfig, train

def plot_results(results, alpha: float, outdir: str):
    os.makedirs(outdir, exist_ok=True)

    # Team reward
    fig1 = plt.figure()
    plt.plot(results["episode_returns"], linewidth=1.0)
    plt.title(f"Team Reward per Episode (alpha={alpha})")
    plt.xlabel("Episode"); plt.ylabel("Team Reward")
    fig1.savefig(os.path.join(outdir, f"team_reward_alpha{alpha}.png"), dpi=150)
    plt.close(fig1)

    # Reward variance
    fig2 = plt.figure()
    plt.plot(results["reward_variances"], linewidth=1.0)
    plt.title(f"Reward Variance (alpha={alpha})")
    plt.xlabel("Episode"); plt.ylabel("Variance")
    fig2.savefig(os.path.join(outdir, f"reward_var_alpha{alpha}.png"), dpi=150)
    plt.close(fig2)

    # Coordination index
    fig3 = plt.figure()
    plt.plot(results["coord_scores"], linewidth=1.0)
    plt.title(f"Coordination Index (alpha={alpha})")
    plt.xlabel("Episode"); plt.ylabel("Index [0,1]")
    fig3.savefig(os.path.join(outdir, f"coord_alpha{alpha}.png"), dpi=150)
    plt.close(fig3)

    if "capture_rate_ma" in results and results["capture_rate_ma"].size > 0:
        fig4 = plt.figure()
        window = len(results["capture_history"]) - len(results["capture_rate_ma"]) + 1
        x = np.arange(len(results["capture_rate_ma"])) + window - 1
        plt.plot(x, results["capture_rate_ma"] * 100.0, linewidth=1.0)
        plt.title(f"Capture Rate (Moving Avg) (alpha={alpha})")
        plt.xlabel("Episode"); plt.ylabel("Capture Rate (%)")
        fig4.savefig(os.path.join(outdir, f"capture_rate_alpha{alpha}.png"), dpi=150)
        plt.close(fig4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--partial_obs", type=int, default=0, help="radius; 0 = full obs")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    env_cfg = GridConfig(
        width=args.width, height=args.height, n_predators=2, max_steps=args.max_steps,
        wall_prob=0.0,
        partial_obs_radius=None if args.partial_obs == 0 else args.partial_obs,
    )
    tcfg = TrainConfig(
        episodes=args.episodes, max_steps=args.max_steps, alpha=args.alpha,
        device=args.device, seed=args.seed,
    )

    print(f"Starting training: episodes={tcfg.episodes}, alpha={tcfg.alpha}, seed={tcfg.seed}")
    results = train(tcfg, env_cfg)

    if "capture_history" in results and results["capture_history"].size > 0:
        overall_capture = float(np.mean(results["capture_history"]) * 100.0)
        print(f"Overall capture rate: {overall_capture:.2f}% "
              f"({int(results['capture_history'].sum())}/{len(results['capture_history'])} episodes)")

    if args.plot:
        tag = f"alpha{args.alpha}_seed{args.seed}"
        out = os.path.join(args.outdir, tag)
        plot_results(results, args.alpha, out)
        print(f"Saved plots to: {os.path.abspath(out)}")

if __name__ == "__main__":
    main()
