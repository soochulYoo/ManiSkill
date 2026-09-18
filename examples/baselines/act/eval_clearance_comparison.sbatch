#!/bin/bash
#SBATCH --job-name=act_eval_clearance_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node33
#SBATCH --nodelist=node38
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/act_eval_clearance_%j.log

# PegInsertionSide-v1 counterpart of eval_friction_comparison.sbatch: runs all 4 comparison cases
# (no_force / force / force_head_no_ttt / force_head_ttt) across the hole-clearance sweep in
# eval_clearance_sweep.py, then saves a comparison plot. See eval_friction_comparison.sbatch for
# the full rationale on GPU sim backend / CONTROL_MODE / CKPT_ITER -- identical here, just a
# different swept quantity (hole clearance instead of table friction).
#
# CONTROL_MODE selects which controller ALL 4 cases run under, e.g.:
#   CONTROL_MODE=pd_joint_pos sbatch eval_clearance_comparison.sbatch
#   CONTROL_MODE=pd_joint_pos_soft sbatch eval_clearance_comparison.sbatch
#
# EXP_TAG must match whatever train_friction_comparison.sbatch used for this task (e.g.
# EXP_TAG=peginsertion) so the default checkpoint paths resolve to the right runs/ dir.
ENV_ID="${ENV_ID:-PegInsertionSide-v1}"
EXP_TAG="${EXP_TAG:-peginsertion}"
EXP_SUFFIX="${EXP_TAG:+_$EXP_TAG}"

CKPT_TAG="${CKPT_ITER:-best}"
if [[ -n "${CKPT_ITER:-}" ]]; then
  DEFAULT_CKPT_NO_FORCE="runs/act_no_force${EXP_SUFFIX}/checkpoints/${CKPT_ITER}.pt"
  DEFAULT_CKPT_FORCE="runs/act_force${EXP_SUFFIX}/checkpoints/${CKPT_ITER}.pt"
  DEFAULT_CKPT_FORCE_HEAD="runs/act_force_head${EXP_SUFFIX}/checkpoints/${CKPT_ITER}.pt"
else
  DEFAULT_CKPT_NO_FORCE="runs/act_no_force${EXP_SUFFIX}/checkpoints/best_eval_success_once.pt"
  DEFAULT_CKPT_FORCE="runs/act_force${EXP_SUFFIX}/checkpoints/best_eval_success_once.pt"
  DEFAULT_CKPT_FORCE_HEAD="runs/act_force_head${EXP_SUFFIX}/checkpoints/best_eval_success_once.pt"
fi

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill/examples/baselines/act

CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
CKPT_NO_FORCE="${CKPT_NO_FORCE:-$DEFAULT_CKPT_NO_FORCE}"
CKPT_FORCE="${CKPT_FORCE:-$DEFAULT_CKPT_FORCE}"
CKPT_FORCE_HEAD="${CKPT_FORCE_HEAD:-$DEFAULT_CKPT_FORCE_HEAD}"

EXTRA_CKPT_ARGS=()
[[ -f "$CKPT_NO_FORCE" ]] && EXTRA_CKPT_ARGS+=(--ckpt-no-force "$CKPT_NO_FORCE")
[[ -f "$CKPT_FORCE" ]] && EXTRA_CKPT_ARGS+=(--ckpt-force "$CKPT_FORCE")
[[ -f "$CKPT_FORCE_HEAD" ]] && EXTRA_CKPT_ARGS+=(--ckpt-force-head "$CKPT_FORCE_HEAD")

python eval_clearance_sweep.py \
  --env-id "$ENV_ID" \
  --control-mode "$CONTROL_MODE" \
  --sim-backend physx_cuda \
  --num-eval-episodes 100 \
  --num-eval-envs 25 \
  --max-episode-steps "${MAX_EPISODE_STEPS:-200}" \
  "${EXTRA_CKPT_ARGS[@]}" \
  --output-dir "runs/clearance_sweep_eval${EXP_SUFFIX}_${CONTROL_MODE}_ckpt${CKPT_TAG}"
