"""Play the container-management RL simulation in the terminal.

Examples:
  python play_game.py --mode manual
  python play_game.py --mode ppo --checkpoint curriculum_ppo_seed7.pt
  python play_game.py --mode random
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np

from waste_sorting_env import WasteSortingEnv
from ppo_numpy import LinearPPO


def run_game(mode="manual", seed=7, stage=3, horizon=60, checkpoint=None):
    env = WasteSortingEnv(stage=stage, seed=seed, horizon=horizon)
    agent = None
    if mode == "ppo":
        checkpoint_path = Path(checkpoint or "curriculum_ppo_seed7.pt")
        agent = LinearPPO(seed=seed)
        agent.load(checkpoint_path)

    obs = env.reset()
    done = False
    total_reward = 0.0
    print("\n=== RL Container Management Simulation ===")
    print(f"Mode: {mode} | Curriculum phase: {stage} | Containers: {env.n}")
    print("Goal: empty containers near target levels without overflowing them.")
    print("Actions: 0 = wait/no-op; 1..N = empty container 1..N.\n")

    while not done:
        print(f"Step {env.t:02d}/{env.horizon} | PU: {'BUSY' if env.busy else 'IDLE'}")
        for i, level in enumerate(env.levels, start=1):
            bar_len = max(0, min(20, int(round(float(level) * 20))))
            bar = "#" * bar_len + "." * (20 - bar_len)
            print(f"  Container {i}: [{bar}] {level:.3f}/{env.CAPACITY:.1f}")
        if mode == "manual":
            while True:
                raw = input(f"Choose action [0..{env.n}] (q to quit): ").strip().lower()
                if raw == "q":
                    print("Game stopped by player.")
                    done = True
                    break
                try:
                    action = int(raw)
                    if 0 <= action <= env.n:
                        break
                except ValueError:
                    pass
                print(f"Enter an integer from 0 to {env.n}, or q to quit.")
            if done:
                break
        elif mode == "random":
            action = int(env.rng.integers(env.n + 1))
            print(f"Random agent chooses action {action}.")
        elif mode == "ppo":
            action, _, _ = agent.act(obs, deterministic=True, valid_actions=env.n + 1)
            print(f"PPO agent chooses action {action}.")
        else:
            raise ValueError("mode must be manual, ppo, or random")

        obs, reward, done, stats = env.step(action)
        total_reward += reward
        print(f"Reward: {reward:+.3f} | Throughput: {stats['throughput']:.3f} | "
              f"Overflow events: {stats['violations']} | Collisions: {stats['collisions']}\n")

    print("=== Episode summary ===")
    print(f"Total reward: {total_reward:.3f}")
    print(f"Throughput: {env.stats['throughput']:.3f}")
    print(f"Successful empties: {env.stats['successful_empties']}")
    print(f"Overflow violations: {env.stats['violations']}")
    print(f"Collision timesteps: {env.stats['collisions']}")
    print(f"Processing-unit idle fraction: {env.stats['idle_steps'] / max(env.t, 1):.3f}")
    return env.stats


def main():
    parser = argparse.ArgumentParser(description="Play the RL container-management simulation.")
    parser.add_argument("--mode", choices=("manual", "ppo", "random"), default="manual")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stage", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--horizon", type=int, default=60)
    parser.add_argument("--checkpoint", default=None, help="PPO checkpoint path for --mode ppo")
    args = parser.parse_args()
    run_game(args.mode, args.seed, args.stage, args.horizon, args.checkpoint)


if __name__ == "__main__":
    main()
