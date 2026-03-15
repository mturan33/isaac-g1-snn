"""
Train G1 Locomotion with SNN-PPO in Isaac Lab
===============================================
Drop-in training script that uses SNN policy for G1 flat terrain walking.

Usage:
    # SNN policy (default):
    ./isaaclab.bat -p train_snn_g1.py --num_envs 2048 --max_iterations 3000
    
    # MLP baseline for comparison:
    ./isaaclab.bat -p train_snn_g1.py --num_envs 2048 --max_iterations 3000 --use_mlp
    
    # Adjust SNN hyperparameters:
    ./isaaclab.bat -p train_snn_g1.py --num_steps 12 --beta_init 0.9 --num_envs 2048

Author: Turan Yardimci
"""

import argparse
import os
import time
from datetime import datetime

# ---- Isaac Lab boilerplate ----
parser = argparse.ArgumentParser(description="SNN-PPO G1 Locomotion Training")

# Isaac Lab args
parser.add_argument("--num_envs", type=int, default=2048)
parser.add_argument("--max_iterations", type=int, default=3000)

# Policy selection
parser.add_argument("--use_mlp", action="store_true", help="Use MLP baseline instead of SNN")

# SNN hyperparameters
parser.add_argument("--num_steps", type=int, default=8, 
                    help="Internal SNN timesteps per env step")
parser.add_argument("--beta_init", type=float, default=0.85,
                    help="Initial LIF decay factor (0-1)")
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128],
                    help="Hidden layer dimensions")

# PPO hyperparameters
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--rollout_steps", type=int, default=24)
parser.add_argument("--ppo_epochs", type=int, default=5)
parser.add_argument("--minibatches", type=int, default=4)
parser.add_argument("--clip_param", type=float, default=0.2)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--entropy_coef", type=float, default=0.01)

# Logging
parser.add_argument("--log_interval", type=int, default=10)
parser.add_argument("--save_interval", type=int, default=500)
parser.add_argument("--log_dir", type=str, default="logs/snn_g1")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ---- Imports after Isaac Lab init ----
import torch
from torch.utils.tensorboard import SummaryWriter

from snn_actor_critic import SNNActorCritic, MLPActorCritic
from snn_ppo import SNNPPO

# =====================================================
# G1 Locomotion Environment
# =====================================================
# Isaac Lab's built-in flat-terrain velocity tracking for Unitree G1
# obs_dim=123: lin_vel(3) + ang_vel(3) + gravity(3) + vel_cmd(3)
#              + joint_pos(37) + joint_vel(37) + prev_actions(37)
# act_dim=37:  all joints (12 leg + 1 torso + 10 arm + 14 finger)
import gymnasium as gym
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks  # registers all tasks

ENV_ID = "Isaac-Velocity-Flat-G1-v0"
# =====================================================


def main():
    device = "cuda:0"

    # ---- Timestamp for logging ----
    policy_type = "mlp" if args.use_mlp else "snn"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"g1_{policy_type}_{timestamp}"
    log_path = os.path.join(args.log_dir, run_name)
    os.makedirs(log_path, exist_ok=True)

    # ---- Create environment ----
    env_cfg = parse_env_cfg(ENV_ID, device=device, num_envs=args.num_envs)

    # Force forward walking — prevent "stand still" exploit
    # Default (0.0, 1.0) allows zero velocity → robot learns to just stand
    env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
    env_cfg.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
    env_cfg.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

    env = gym.make(ENV_ID, cfg=env_cfg)

    # Get dimensions from env
    obs_dim = env.observation_space["policy"].shape[-1]
    act_dim = env.action_space.shape[-1]
    
    print(f"\n{'='*60}")
    print(f"  SNN-PPO G1 Locomotion Training")
    print(f"  Policy: {'MLP (baseline)' if args.use_mlp else 'SNN (LIF neurons)'}")
    print(f"  Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"  Num envs: {args.num_envs}")
    if not args.use_mlp:
        print(f"  SNN timesteps: {args.num_steps}")
        print(f"  Beta init: {args.beta_init}")
        print(f"  Hidden dims: {args.hidden_dims}")
    print(f"  Log dir: {log_path}")
    print(f"{'='*60}\n")
    
    # ---- Create model ----
    if args.use_mlp:
        model = MLPActorCritic(obs_dim, act_dim, args.hidden_dims).to(device)
    else:
        model = SNNActorCritic(
            obs_dim, act_dim,
            actor_hidden=args.hidden_dims,
            critic_hidden=args.hidden_dims,
            num_steps=args.num_steps,
            beta_init=args.beta_init,
        ).to(device)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model parameters: {param_count:,}")
    
    # ---- Create trainer ----
    trainer = SNNPPO(
        actor_critic=model,
        num_envs=args.num_envs,
        obs_dim=obs_dim,
        act_dim=act_dim,
        device=device,
        lr=args.lr,
        num_steps_per_update=args.rollout_steps,
        num_epochs=args.ppo_epochs,
        num_minibatches=args.minibatches,
        clip_param=args.clip_param,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
    )
    
    # ---- TensorBoard ----
    writer = SummaryWriter(log_path)
    
    # ---- Training loop ----
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    
    # Initialize SNN membranes
    model.reset_membranes(args.num_envs, device)
    
    best_reward = -float("inf")
    start_time = time.time()
    
    for iteration in range(1, args.max_iterations + 1):
        # Collect rollout
        obs = trainer.collect_rollout(env, obs)
        
        # PPO update
        metrics = trainer.update()
        
        # Logging
        if iteration % args.log_interval == 0:
            elapsed = time.time() - start_time
            fps = (iteration * args.rollout_steps * args.num_envs) / elapsed
            
            reward = metrics["misc/mean_reward"]
            print(
                f"[Iter {iteration:5d}/{args.max_iterations}] "
                f"Reward: {reward:+8.3f} | "
                f"Policy Loss: {metrics['loss/policy']:.4f} | "
                f"Value Loss: {metrics['loss/value']:.4f} | "
                f"Entropy: {metrics['loss/entropy']:.4f} | "
                f"FPS: {fps:.0f}"
            )
            
            # SNN-specific logging
            if not args.use_mlp:
                for key, val in metrics.items():
                    if key.startswith("snn/"):
                        print(f"  {key}: {val:.4f}")
            
            # TensorBoard
            for key, val in metrics.items():
                writer.add_scalar(key, val, iteration)
            writer.add_scalar("misc/fps", fps, iteration)
        
        # Save checkpoints
        if iteration % args.save_interval == 0:
            save_path = os.path.join(log_path, f"model_{iteration}.pt")
            trainer.save(save_path)
        
        # Save best
        reward = metrics["misc/mean_reward"]
        if reward > best_reward:
            best_reward = reward
            trainer.save(os.path.join(log_path, "model_best.pt"))
    
    # ---- Final save ----
    trainer.save(os.path.join(log_path, "model_final.pt"))
    
    # ---- Summary ----
    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  Training Complete!")
    print(f"  Policy: {'MLP' if args.use_mlp else 'SNN'}")
    print(f"  Best reward: {best_reward:+.3f}")
    print(f"  Total time: {total_time/60:.1f} minutes")
    print(f"  Checkpoints: {log_path}")
    print(f"{'='*60}")
    
    writer.close()
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
