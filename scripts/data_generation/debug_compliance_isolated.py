"""Isolated diagnostic: drive PDEEPoseComplianceController with a deliberate, hand-crafted target
displacement (no trained policy, no rendering) and print its state once per control step, to see
whether the controller converges toward the target or diverges/oscillates -- independent of
whatever the ACT policy outputs and without the heavy RGBD+CNN overhead of the full eval pipeline.

Set SIM_BACKEND=physx_cpu or physx_cuda (default physx_cuda) to compare backends.
"""
import os

import torch

import gymnasium as gym
import mani_skill.envs  # noqa: F401

sim_backend = os.environ.get("SIM_BACKEND", "physx_cuda")

env = gym.make(
    "PushCube-v1",
    num_envs=1,
    sim_backend=sim_backend,
    control_mode="pd_ee_pose_compliance",
    obs_mode="state",
    render_mode="rgb_array",
    max_episode_steps=200,
)
obs, _ = env.reset(seed=0)

controller = env.unwrapped.agent.controller.controllers["arm"]
cur = controller.ee_pose_at_base
print(f"sim_backend={sim_backend}", flush=True)
print(f"initial ee pose p={cur.p.tolist()} q={cur.q.tolist()}", flush=True)
print(f"pos_stiffness={controller._pos_stiffness.tolist()} pos_damping={controller._pos_damping.tolist()}",
      flush=True)

target_p = cur.p.clone()
target_p[:, 0] += 0.2
target_p[:, 2] += 0.1
target_euler = torch.zeros((1, 3), device=cur.p.device)

action_dim = env.action_space.shape[-1]
action = torch.zeros((1, action_dim), device=cur.p.device)
action[:, 0:3] = target_p
action[:, 3:6] = target_euler
print(f"target.p={target_p.tolist()}", flush=True)

for i in range(30):
    obs, rew, term, trunc, info = env.step(action)
    ee = controller.ee_pose_at_base
    qvel_norm = controller.qvel.norm().item()
    print(f"[control step {i}] ee.p={ee.p.tolist()} ee.q={ee.q.tolist()} |qvel|={qvel_norm:.4f}", flush=True)

print("Final qpos:", controller.qpos.tolist())
print("Final ee pose:", controller.ee_pose_at_base.p.tolist())
