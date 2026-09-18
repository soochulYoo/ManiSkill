#!/usr/bin/env bash
# Replays raw pd_joint_pos motion-planning demos to attach rgbd observations, WITHOUT changing
# control mode (unlike replay_pd_ee_pose.sh, which re-solves actions via IK into pd_ee_pose).
#
# --allow-failure is REQUIRED here, not optional. mani_skill.utils.wrappers.record.RecordEpisode's
# trajectory buffer only advances its flush watermark (env_episode_ptr) inside flush_trajectory()
# (see record.py's flush_trajectory, ~line 711); a plain reset() never advances it. replay_trajectory
# only calls flush_trajectory() for episodes that reproduce their recorded "success" flag -- so
# without --allow-failure, any episode whose replay doesn't hit success (common even with an
# UNCHANGED control mode, since physx_cpu isn't bit-identical run-to-run and this task's success
# check has a tight tolerance) is silently never flushed, and its steps sit in the buffer and get
# swept into the NEXT successful flush -- i.e. the next "successful" episode's saved trajectory is
# actually the failed episode(s) before it concatenated with itself (multiple terminated=True
# events spliced into one saved "episode"). This is the same bug that produced replay_pd_ee_pose.sh's
# inflated elapsed_steps; it's a property of replay_trajectory.py's CPU path, not of IK conversion.
# Since we aren't re-solving actions here (same control mode in and out), the recorded action
# sequence for every episode already IS a valid successful demo -- a replay "failing" its success
# check just means physx settled the cube fractionally outside the tolerance this run, not that the
# motion was bad. --allow-failure flushes every episode regardless, which also fixes the
# contamination above as a side effect (every episode's flush advances the watermark), so we recover
# clean data for (close to) all 600 episodes instead of a contaminated ~25% subset.
#
# Run standalone:
#   bash scripts/data_generation/replay_pd_joint_pos.sh
# Or via the SLURM wrapper:
#   sbatch scripts/data_generation/replay_pd_joint_pos.sbatch
set -euo pipefail

RAW_TRAJ="${RAW_TRAJ:-/scratch2/soochul/ManiSkill/demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.h5}"
TARGET_CONTROL_MODE="${TARGET_CONTROL_MODE:-pd_joint_pos}"

echo "[replaying $RAW_TRAJ -> control_mode=$TARGET_CONTROL_MODE (rgbd obs only, no control-mode change)]"

python -m mani_skill.trajectory.replay_trajectory \
    --traj-path "$RAW_TRAJ" \
    -o rgbd \
    -c "$TARGET_CONTROL_MODE" \
    --use-first-env-state \
    --allow-failure \
    --save-traj
