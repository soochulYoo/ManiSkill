#!/usr/bin/env bash
# Renders side-by-side comparison videos for the 122 episodes that exist in both
# demos_friction/.../trajectory.h5 (pd_joint_pos, 600 episodes) and
# demos_friction/.../trajectory.rgbd.pd_ee_pose.physx_cpu.h5 (pd_ee_pose conversion, 122
# episodes), so the two control modes can be eyeballed against each other per-episode --
# including the episodes with the inflated elapsed_steps noted in train_friction_comparison.sbatch.
#
# Run standalone:
#   bash scripts/data_generation/compare_pd_joint_vs_ee_pose_videos.sh
# Or via the SLURM wrapper:
#   sbatch scripts/data_generation/compare_pd_joint_vs_ee_pose_videos.sbatch
set -euo pipefail

python scripts/data_generation/compare_pd_joint_vs_ee_pose_videos.py "$@"
