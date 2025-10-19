from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

from .envs.gridworld import GridConfig
from .train import TrainConfig, train


def plot_vs_alpha(results, outdir: str):
    os.makedirs(outdir, exist_ok=True)
    a = results["alpha_values"]
    y_cap = results["per_alpha_capture_rate"] * 100.0
    y_ret = results["per_alpha_mean_return"]
    y_coord = results["per_alpha_mean_coord"]

    fig = plt.figure()
    plt.plot(a, y_cap, linewidth=1.2)
    plt.title("Capture Rate vs α")
    plt.xlabel("α"); plt.ylabel("Capture Rate (%)")
    fig.savefig(os.path.join(outdir, "capture_rate_vs_alpha.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(a, y_ret, linewidth=1.2)
    plt.title("Mean Team Return vs α")
    plt.xlabel("α"); plt.ylabel("Mean Team Return")
    fig.savefig(os.path.join(outdir, "team_return_vs_alpha.png"), dpi=150)
    plt.close(fig)

    fig = plt.figure()
    plt.plot(a, y_coord, linewidth=1.2)
    plt.title("Coordination Index vs α")
    plt.xlabel("α"); plt.ylabel("Coordination Index")
    fig.savefig(os.path.join(outdir, "coord_vs_alpha.png"), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20000)
    parser.add_argument("--alpha", type=float, default=0.5, help="Fixed alpha if not sweeping")
    parser.add_argument("--alpha_sweep", action="store_true", help="Sweep α from 0.00→0.99 automatically")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=7)
    parser.add_argument("--height", type=int, default=7)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--partial_obs", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--outdir", type=str, default="runs")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    env_cfg = GridConfig(
        width=args.width,
        height=args.height,
        n_predators=2,
        max_steps=args.max_steps,
        wall_prob=0.0,
        partial_obs_radius=None if args.partial_obs == 0 else args.partial_obs,
    )

    tcfg = TrainConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        device=args.device,
        seed=args.seed,
        alpha_sweep=args.alpha_sweep,
    )

    mode = "α-sweep (0→0.99)" if args.alpha_sweep else f"fixed α={args.alpha}"
    print(f"Starting training: episodes={tcfg.episodes}, mode={mode}, seed={tcfg.seed}")

    results = train(tcfg, env_cfg)

    if args.plot:
        tag = "alpha_sweep" if args.alpha_sweep else f"alpha{args.alpha}_seed{args.seed}"
        outdir = os.path.join(args.outdir, tag)
        os.makedirs(outdir, exist_ok=True)

        if args.alpha_sweep:
            plot_vs_alpha(results, outdir)
        else:
            # simple per-episode plotting
            fig = plt.figure()
            plt.plot(results["episode_returns"], linewidth=1.0)
            plt.title(f"Team Reward per Episode (α={args.alpha})")
            plt.xlabel("Episode")
            plt.ylabel("Team Reward")
            fig.savefig(os.path.join(outdir, "team_reward.png"), dpi=150)
            plt.close(fig)

        print(f"Saved plots to: {os.path.abspath(outdir)}")

    if "capture_history" in results and results["capture_history"].size > 0:
        overall = float(np.mean(results["capture_history"]) * 100.0)
        print(f"Overall capture rate: {overall:.2f}% "
              f"({int(results['capture_history'].sum())}/{len(results['capture_history'])} episodes)")


if __name__ == "__main__":
    main()
