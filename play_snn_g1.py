"""
Visualize trained SNN policy on G1.
Records video for LinkedIn demo.

Usage:
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt --use_mlp
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt --video

Author: Turan Yardimci
"""

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--use_mlp", action="store_true")
parser.add_argument("--num_steps", type=int, default=8)
parser.add_argument("--beta_init", type=float, default=0.85)
parser.add_argument("--hidden_dims", type=int, nargs="+", default=[256, 128])
parser.add_argument("--max_play_steps", type=int, default=2000)
parser.add_argument("--video", action="store_true", help="Record video")
parser.add_argument("--video_length", type=int, default=500, help="Video length in steps")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import torch
import gymnasium as gym
from snn_actor_critic import SNNActorCritic, MLPActorCritic
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks

ENV_ID = "Isaac-Velocity-Flat-G1-v0"


def main():
    device = "cuda:0"

    env_cfg = parse_env_cfg(ENV_ID, device=device, num_envs=args.num_envs)
    render_mode = "rgb_array" if args.video else None
    env = gym.make(ENV_ID, cfg=env_cfg, render_mode=render_mode)

    # Wrap for video recording
    if args.video:
        policy_type = "mlp" if args.use_mlp else "snn"
        video_dir = os.path.join(os.path.dirname(args.checkpoint), f"videos_{policy_type}")
        env = gym.wrappers.RecordVideo(
            env, video_dir,
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[VIDEO] Recording to {video_dir}")

    obs_dim = env.observation_space["policy"].shape[-1]
    act_dim = env.action_space.shape[-1]
    
    # Load model
    if args.use_mlp:
        model = MLPActorCritic(obs_dim, act_dim, args.hidden_dims).to(device)
    else:
        model = SNNActorCritic(
            obs_dim, act_dim,
            actor_hidden=args.hidden_dims,
            num_steps=args.num_steps,
            beta_init=args.beta_init,
        ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    policy_type = "MLP" if args.use_mlp else "SNN"
    print(f"\n[PLAY] {policy_type} policy loaded from {args.checkpoint}")
    print(f"[PLAY] Iteration: {checkpoint.get('iteration', '?')}")
    
    # Run
    obs_dict, _ = env.reset()
    obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
    model.reset_membranes(args.num_envs, device)
    
    total_reward = 0
    for step in range(args.max_play_steps):
        with torch.no_grad():
            action = model.act(obs, deterministic=True)
            if isinstance(action, tuple):
                action = action[0]
        
        obs_dict, reward, _, _, _ = env.step(action)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        total_reward += reward.mean().item()
        
        if step % 200 == 0:
            print(f"[Step {step:5d}] Avg Reward: {total_reward/(step+1):+.3f}")
    
    print(f"\n[DONE] Total avg reward: {total_reward/args.max_play_steps:+.3f}")
    
    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
