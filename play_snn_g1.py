"""
Visualize trained SNN policy on G1.
Records video for LinkedIn demo.

Usage:
    # GUI mode (watch live):
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt --use_mlp

    # Record video (GUI mode required — do NOT use --headless):
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt --record
    ./isaaclab.bat -p play_snn_g1.py --checkpoint logs/snn_g1/.../model_best.pt --use_mlp --record

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
parser.add_argument("--max_play_steps", type=int, default=1000)
parser.add_argument("--record", action="store_true", help="Record video (GUI mode required)")
parser.add_argument("--record_fps", type=int, default=25, help="Video FPS")
parser.add_argument("--record_dir", type=str, default=None, help="Video output dir")

from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import os
import math
import torch
import gymnasium as gym
from snn_actor_critic import SNNActorCritic, MLPActorCritic
from isaaclab_tasks.utils import parse_env_cfg
import isaaclab_tasks

ENV_ID = "Isaac-Velocity-Flat-G1-v0"


# ============================================================
# Video Recording (viewport capture + ffmpeg)
# ============================================================

def _find_ffmpeg():
    """Find ffmpeg binary."""
    import shutil
    import glob
    # WinGet Links shortcut
    winget_link = os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Links\ffmpeg.exe"
    )
    if os.path.isfile(winget_link):
        return winget_link
    # WinGet Packages (full build)
    winget_pkg = glob.glob(os.path.expandvars(
        r"%LOCALAPPDATA%\Microsoft\WinGet\Packages\*FFmpeg*\*\bin\ffmpeg.exe"
    ))
    if winget_pkg:
        return winget_pkg[0]
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    return "ffmpeg"


class VideoRecorder:
    """Captures viewport frames at target FPS, then merges with ffmpeg."""

    def __init__(self, output_dir, fps=25, control_dt=0.02):
        self.output_dir = output_dir
        self.fps = fps
        self.frame_dir = os.path.join(output_dir, "frames")
        import shutil as _sh
        if os.path.exists(self.frame_dir):
            _sh.rmtree(self.frame_dir)
        os.makedirs(self.frame_dir, exist_ok=True)
        self.frame_count = 0
        self._sim_step = 0
        self._capture_interval = max(1, int(1.0 / (control_dt * fps)))

        from omni.kit.viewport.utility import get_active_viewport
        self.viewport = get_active_viewport()
        print(f"[VIDEO] Recorder: {fps} FPS, capture every {self._capture_interval} steps")

    def set_camera_pose(self, eye, target):
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=list(eye), target=list(target))
        except Exception:
            pass

    def on_step(self):
        self._sim_step += 1
        if self._sim_step % self._capture_interval == 0:
            from omni.kit.viewport.utility import capture_viewport_to_file
            frame_path = os.path.join(
                self.frame_dir, f"frame_{self.frame_count:06d}.png"
            )
            capture_viewport_to_file(self.viewport, frame_path)
            self.frame_count += 1

    def finalize(self, output_name="demo.mp4"):
        import subprocess
        if self.frame_count == 0:
            print("[VIDEO] No frames captured!")
            return None

        output_path = os.path.join(self.output_dir, output_name)
        frame_pattern = os.path.join(self.frame_dir, "frame_%06d.png")
        ffmpeg_bin = _find_ffmpeg()

        # Wait for async frame writes
        import time as _t
        last_frame = os.path.join(
            self.frame_dir, f"frame_{self.frame_count - 1:06d}.png"
        )
        print(f"[VIDEO] Waiting for {self.frame_count} frames to flush...")
        for _ in range(60):
            if os.path.exists(last_frame) and os.path.getsize(last_frame) > 0:
                break
            _t.sleep(0.5)
        _t.sleep(2.0)

        actual = len([f for f in os.listdir(self.frame_dir) if f.endswith('.png')])
        print(f"[VIDEO] Frames on disk: {actual}/{self.frame_count}")

        cmd = [
            ffmpeg_bin, "-y",
            "-framerate", str(self.fps),
            "-i", frame_pattern,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ]
        print(f"[VIDEO] Converting to MP4...")
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                print(f"[VIDEO] Saved: {output_path}")
                import shutil
                shutil.rmtree(self.frame_dir)
                return output_path
            else:
                print(f"[VIDEO] ffmpeg failed: {result.stderr.decode(errors='replace')[:300]}")
                print(f"[VIDEO] Frames saved in: {self.frame_dir}")
                return None
        except Exception as e:
            print(f"[VIDEO] ffmpeg error: {e}")
            print(f"[VIDEO] Frames saved in: {self.frame_dir}")
            return None


class CameraTracker:
    """EMA-smoothed camera following robot 0 at 45-degree angle."""

    EYE_RADIUS = 4.0
    EYE_ANGLE = -0.785  # -45deg front-right
    EYE_Z = 1.5
    TARGET_Z = 0.7

    def __init__(self, recorder):
        self._recorder = recorder
        self._sx = self._sy = self._syaw = 0.0
        self._alpha_pos = 0.12
        self._alpha_yaw = 0.06
        self._init = False

    def update(self, robot_pos):
        """robot_pos: [num_envs, 3] tensor — tracks env 0."""
        rx = robot_pos[0, 0].item()
        ry = robot_pos[0, 1].item()

        if not self._init:
            self._sx, self._sy = rx, ry
            self._init = True
        else:
            self._sx += self._alpha_pos * (rx - self._sx)
            self._sy += self._alpha_pos * (ry - self._sy)

        eye = (
            self._sx + self.EYE_RADIUS * math.cos(self.EYE_ANGLE),
            self._sy + self.EYE_RADIUS * math.sin(self.EYE_ANGLE),
            self.EYE_Z,
        )
        target = (self._sx, self._sy, self.TARGET_Z)

        if self._recorder:
            self._recorder.set_camera_pose(eye, target)


# ============================================================
# Main
# ============================================================

def main():
    device = "cuda:0"

    env_cfg = parse_env_cfg(ENV_ID, device=device, num_envs=args.num_envs)
    env = gym.make(ENV_ID, cfg=env_cfg)

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

    # Setup recorder
    recorder = None
    tracker = None
    if args.record:
        if args.record_dir:
            vid_dir = args.record_dir
        else:
            vid_dir = os.path.join(os.path.dirname(args.checkpoint),
                                   f"videos_{policy_type.lower()}")
        os.makedirs(vid_dir, exist_ok=True)
        recorder = VideoRecorder(vid_dir, fps=args.record_fps,
                                 control_dt=env_cfg.sim.dt * env_cfg.decimation)
        tracker = CameraTracker(recorder)
        print(f"[VIDEO] Output dir: {vid_dir}")

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

        obs_dict, reward, terminated, truncated, infos = env.step(action)
        obs = obs_dict["policy"] if isinstance(obs_dict, dict) else obs_dict
        total_reward += reward.mean().item()

        # Camera tracking + frame capture
        if recorder:
            # Get robot position from env
            try:
                robot_pos = env.unwrapped.scene["robot"].data.root_pos_w
                tracker.update(robot_pos)
            except Exception:
                pass
            recorder.on_step()

        if step % 200 == 0:
            print(f"[Step {step:5d}] Avg Reward: {total_reward/(step+1):+.3f}")

    print(f"\n[DONE] Total avg reward: {total_reward/args.max_play_steps:+.3f}")

    # Finalize video
    if recorder:
        vid_name = f"g1_{policy_type.lower()}_demo.mp4"
        recorder.finalize(vid_name)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
