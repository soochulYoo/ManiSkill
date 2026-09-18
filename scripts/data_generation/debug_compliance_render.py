"""One-off diagnostic: render a couple of pd_ee_pose_compliance eval episodes to video so we can
see what the robot is actually doing, instead of just a success_rate=0.0 number. Run from the
mani_skill repo root; must run on a GPU node (physx_cuda).
"""
import os
import sys
from functools import partial

import numpy as np
import torch

os.chdir("examples/baselines/act")  # checkpoints/runs are all saved relative to this dir
sys.path.insert(0, ".")

import train_rgbd
from act.make_env import make_eval_envs
from eval_friction_sweep import Args, build_agent, evaluate_with_ttt
from train_rgbd import FlattenRGBDObservationWrapper

sim_backend = os.environ.get("SIM_BACKEND", "physx_cuda")
device = torch.device("cuda" if sim_backend == "physx_cuda" else "cpu")

args = Args(
    env_id="PushCube-v1",
    control_mode="pd_ee_pose_compliance",
    sim_backend=sim_backend,
    num_eval_episodes=2,
    # physx_cpu's multi-env path has a pre-existing, unrelated bug with depth+float16 +
    # multiprocessing shared memory (see eval_friction_comparison.sbatch's comment) -- use 1 env
    # to sidestep it for this diagnostic.
    num_eval_envs=1 if sim_backend == "physx_cpu" else 2,
    max_episode_steps=200,  # match training -- the sweep scripts were missing this (default is
    # PushCube-v1's registered 50), so eval was silently running a 4x shorter horizon than the
    # policy was trained/evaluated-during-training on.
)

density, static_f, dynamic_f = (1000.0, 3.3, 2.3)  # baseline condition
env_kwargs = dict(
    control_mode=args.control_mode,
    reward_mode="normalized_dense",
    obs_mode="rgbd" if args.include_depth else "rgb",
    render_mode="rgb_array",
    obj_density=density,
    static_friction=static_f,
    dynamic_friction=dynamic_f,
    max_episode_steps=args.max_episode_steps,
)

wrappers = [partial(FlattenRGBDObservationWrapper, depth=args.include_depth, force=False)]
video_dir = f"runs/debug_compliance_render_{sim_backend}"
envs = make_eval_envs(
    args.env_id, args.num_eval_envs, args.sim_backend, env_kwargs, None,
    video_dir=video_dir, wrappers=wrappers,
)

train_rgbd.args = args  # Agent.get_action() reads this module-global (set by eval_friction_sweep.py's
# __main__ normally; this diagnostic script bypasses that entrypoint, so set it explicitly)
agent = build_agent(envs, args, include_force=False, use_force_magnitude_head=False, device=device)
ckpt = torch.load("runs/act_no_force_n600_i40000/checkpoints/best_eval_success_once.pt", map_location=device)
agent.load_state_dict(ckpt["ema_agent"])
norm_stats = ckpt["norm_stats"]

metrics = evaluate_with_ttt(args.num_eval_episodes, agent, envs, norm_stats, args, device, use_ttt=False)
envs.close()

success_key = "success_at_end" if "success_at_end" in metrics else "success_once"
print(f"success_rate={float(np.mean(metrics[success_key])):.4f}")
print(f"Video(s) saved under {video_dir}")
