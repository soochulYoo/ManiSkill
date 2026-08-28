#!/bin/bash
#SBATCH --job-name=act_eval_friction_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/act_eval_%j.log

# Runs all 4 comparison cases (no_force / force / force_head_no_ttt / force_head_ttt) across the
# full friction sweep in one job, then saves a comparison plot. Uses the GPU sim backend, both for
# speed (vectorized rollout across all --num-eval-envs at once) and because the CPU backend's
# multi-env path has a pre-existing bug in this baseline (depth observations are cast to float16,
# which Python's multiprocessing shared-memory arrays can't represent) -- unrelated to anything
# added here, but avoided by just using GPU eval.
#
# --time is a rough guess (not benchmarked on this cluster): 4 cases x 7 conditions x
# num-eval-episodes rollouts. Check logs/act_eval_<jobid>.log partway through and adjust before
# resubmitting if it's way off.
#
# Fill in the checkpoint paths below (or override via env vars, e.g.
#   CKPT_NO_FORCE=runs/act_no_force/checkpoints/500000.pt sbatch eval_friction_comparison.sbatch
# ) once you know which checkpoint from training you want to evaluate -- training's --save-freq
# checkpoints are numbered by iteration, and best_eval_success_once.pt/best_eval_success_at_end.pt
# only exist if that run ever beat its previous best success rate during training-time eval.

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill/examples/baselines/act

CKPT_NO_FORCE="${CKPT_NO_FORCE:-runs/act_no_force/checkpoints/best_eval_success_once.pt}"
CKPT_FORCE="${CKPT_FORCE:-runs/act_force/checkpoints/best_eval_success_once.pt}"
CKPT_FORCE_HEAD="${CKPT_FORCE_HEAD:-runs/act_force_head/checkpoints/best_eval_success_once.pt}"

python eval_friction_sweep.py \
  --env-id PushCube-v1 \
  --control-mode pd_joint_pos \
  --sim-backend physx_cuda \
  --num-eval-episodes 100 \
  --num-eval-envs 25 \
  --ckpt-no-force "$CKPT_NO_FORCE" \
  --ckpt-force "$CKPT_FORCE" \
  --ckpt-force-head "$CKPT_FORCE_HEAD" \
  --output-dir runs/friction_sweep_eval
