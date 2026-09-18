#!/usr/bin/env bash
# Converts raw pd_joint_pos motion-planning demos into pd_ee_pose actions, keeping only episodes
# that still succeed after conversion (no --allow-failure). Control-mode conversion is lossy --
# expect roughly a 20% survival rate for PushCube-v1 (measured on a 20-episode sample), so collect
# enough raw demos up front that ~100+ survive (e.g. 600 raw -> ~100-120 successful).
#
# Run standalone:
#   bash scripts/data_generation/replay_pd_ee_pose.sh
# Or via the SLURM wrapper:
#   sbatch scripts/data_generation/replay_pd_ee_pose.sbatch
set -euo pipefail

RAW_TRAJ="${RAW_TRAJ:-/scratch2/soochul/ManiSkill/demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.h5}"
TARGET_CONTROL_MODE="${TARGET_CONTROL_MODE:-pd_ee_pose}"

echo "[converting $RAW_TRAJ -> control_mode=$TARGET_CONTROL_MODE]"

python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$RAW_TRAJ" \
    -o rgbd \
    -c "$TARGET_CONTROL_MODE" \
    --use-first-env-state \
    --save-traj
