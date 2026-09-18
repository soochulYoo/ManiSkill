#!/bin/bash
#SBATCH --job-name=act_train_friction_comparison
#SBATCH --partition=suma_a6000
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2%1
#SBATCH --time=24:00:00
#SBATCH --output=logs/act_train_%A_%a.log

# One array task per model variant (3 -> --array=0-2):
#   0: no_force    -- ACT without any force input
#   1: force       -- ACT with force (direction+magnitude), no auxiliary head
#   2: force_head  -- ACT with force + the force-magnitude prediction head (used for BOTH the
#                     force_head_no_ttt and force_head_ttt eval cases -- TTT is an eval-time-only
#                     behavior, not a separate training run)
# --time is a rough guess (not benchmarked on this cluster) -- check logs/act_train_<jobid>_0.log
# after it's running a while and adjust before resubmitting if it's way off.
#
# ENV_ID/EXP_TAG/DEMO_PATH/TOTAL_ITERS/MAX_EPISODE_STEPS/SAVE_FREQ are overridable via env vars so
# this same script works for other tasks, not just PushCube-v1 -- all default to today's exact
# PushCube values/behavior when unset. EXP_TAG is appended to each case's --exp-name (e.g.
# "act_no_force" vs "act_no_force_pullcube") so different tasks' runs/ dirs never collide. e.g.:
#   ENV_ID=PullCube-v1 EXP_TAG=pullcube TOTAL_ITERS=40000 MAX_EPISODE_STEPS=100 SAVE_FREQ=2000 \
#     DEMO_PATH=/path/to/trajectory.rgbd.pd_joint_pos.physx_cpu.h5 \
#     sbatch --array=0-2 train_friction_comparison.sbatch

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill/examples/baselines/act

# point this at your full (all-episodes) replayed trajectory file, matching --control-mode below
# (train_rgbd.py asserts the two match). e.g. the output of:
#   python -m mani_skill.trajectory.replay_trajectory --traj-path <raw>.h5 -o rgbd -c pd_joint_pos --use-first-env-state --allow-failure --save-traj
# (see scripts/data_generation/replay_pd_joint_pos.sh -- --allow-failure is required, not
# optional: without it, replay-vs-success mismatches leave RecordEpisode's flush watermark behind
# and the next successful flush splices in the unflushed episode(s) before it. Since this replay
# doesn't re-solve actions via IK -- same control mode the demos were collected under -- every
# recorded action sequence is already a valid demo regardless of whether its replay reproduces the
# success flag, so --allow-failure recovers clean data for all 600 episodes, no filtered subset
# needed).
DEMO_PATH="${DEMO_PATH:-/scratch2/soochul/ManiSkill/demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5}"

ENV_ID="${ENV_ID:-PushCube-v1}"
EXP_TAG="${EXP_TAG:-}"
EXP_SUFFIX="${EXP_TAG:+_$EXP_TAG}"

CASES=(no_force force force_head)
CASE="${CASES[$SLURM_ARRAY_TASK_ID]}"

case "$CASE" in
  no_force)
    EXTRA_ARGS=(--no-include-force)
    ;;
  force)
    EXTRA_ARGS=(--include-force --no-use-force-magnitude-head)
    ;;
  force_head)
    EXTRA_ARGS=(--include-force --use-force-magnitude-head)
    ;;
esac


# --max-episode-steps is required (this baseline does not auto-derive it from the env -- an
# unset value crashes at the first periodic eval with `NoneType + int`). Demos run 61-105 steps;
# 200 gives the policy some headroom beyond the longest demo.
#
# --total-iters overrides the 1,000,000 default: at the observed ~7-8k iters/hour on this
# cluster, 1M iters would take ~5+ days, blowing past --time=24:00:00 above -- the job gets
# killed before the LR ever drops (scheduled at 2/3 of total_iters) and long before it's done.
# 60k was chosen after watching no_force's 100k run: eval success had already plateaued well
# before the 60k checkpoint, so the remaining iters weren't buying anything. All 3 cases use the
# same --total-iters so they share the same LR-drop timing (2/3 of total_iters) for a fair
# comparison.
python train_rgbd.py \
  --env-id "$ENV_ID" \
  --demo-path "$DEMO_PATH" \
  --control-mode pd_joint_pos \
  --exp-name "act_${CASE}${EXP_SUFFIX}" \
  --save-freq "${SAVE_FREQ:-20000}" \
  --sim-backend physx_cuda \
  --max-episode-steps "${MAX_EPISODE_STEPS:-200}" \
  --total-iters "${TOTAL_ITERS:-60000}" \
  --track \
  --wandb-entity usam205_ \
  "${EXTRA_ARGS[@]}"
