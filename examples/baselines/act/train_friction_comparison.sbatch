#!/bin/bash
#SBATCH --job-name=act_train_friction_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2
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

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill/examples/baselines/act

# point this at your full (all-episodes) replayed trajectory file, e.g. the output of:
#   python -m mani_skill.trajectory.replay_trajectory --traj-path <raw>.h5 -o rgbd --use-env-states --save-traj
DEMO_PATH="${DEMO_PATH:-/scratch2/soochul/ManiSkill/demos_friction/density_1000_friction_3.3_2.3/PushCube-v1/motionplanning/trajectory.rgbd.pd_joint_pos.physx_cpu.h5}"

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

# --save-freq guarantees a checkpoint exists even if the policy never hits a new best success
# rate during training-time periodic eval (save_ckpt is otherwise only called on new bests).
python train_rgbd.py \
  --env-id PushCube-v1 \
  --demo-path "$DEMO_PATH" \
  --control-mode pd_joint_pos \
  --exp-name "act_${CASE}" \
  --save-freq 20000 \
  "${EXTRA_ARGS[@]}"
