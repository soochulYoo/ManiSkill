#!/bin/bash
#SBATCH --job-name=act_eval_friction_comparison
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01,node33
#SBATCH --nodelist=node38
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
# CONTROL_MODE selects which controller ALL 4 cases run under (uniform -- no per-case override),
# e.g.:
#   CONTROL_MODE=pd_joint_pos sbatch eval_friction_comparison.sbatch
#   CONTROL_MODE=pd_joint_pos_soft sbatch eval_friction_comparison.sbatch
# pd_joint_pos matches training exactly (rigid, same as what generated the demos). pd_joint_pos_soft
# is a lower-stiffness implicit-drive variant (see panda.py's arm_soft_stiffness/arm_soft_damping)
# that yields more under contact, giving a more friction-informative force signal -- useful for
# isolating whether TTT's online adaptation helps once the force signal actually carries something
# to adapt to, and for a same-controller comparison against the plain pd_joint_pos run.
# (This pipeline no longer uses any pd_ee_pose control mode, so the pd_ee_pose_compliance
# GPU-instability issue that motivated pd_ee_pose_soft doesn't apply here.)
#
# Output dir is namespaced by CONTROL_MODE and CKPT_TAG automatically so runs under different
# controllers/checkpoint-iterations never overwrite each other.
#
# --time is a rough guess (not benchmarked on this cluster): 4 cases x 7 conditions x
# num-eval-episodes rollouts. Check logs/act_eval_<jobid>.log partway through and adjust before
# resubmitting if it's way off.
#
# CKPT_ITER picks which --save-freq checkpoint (numbered by training iteration) to evaluate for
# ALL 3 cases at once, e.g. CKPT_ITER=20000 sbatch eval_friction_comparison.sbatch -- useful when
# a later iteration count (e.g. training's own best_eval_*.pt, picked once loss/eval plateaus)
# saturates success rate across all cases and makes the cross-case comparison less informative;
# an earlier, not-yet-saturated checkpoint shows more separation. Defaults to each case's
# best_eval_success_once.pt (the checkpoint that beat its own running-best success_once rate
# during training) if CKPT_ITER is unset. Individual --ckpt-* paths can still be overridden
# directly (e.g. CKPT_NO_FORCE=runs/act_no_force/checkpoints/500000.pt), which takes precedence
# over CKPT_ITER for that one case.
#
# ENV_ID/EXP_TAG mirror train_friction_comparison.sbatch's own overrides -- EXP_TAG must match
# whatever that training run used so the default checkpoint paths below resolve to the right
# runs/ dir, e.g. ENV_ID=PullCube-v1 EXP_TAG=pullcube sbatch eval_friction_comparison.sbatch.
ENV_ID="${ENV_ID:-PushCube-v1}"
EXP_TAG="${EXP_TAG:-}"
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

# Only pass a --ckpt-* flag if that checkpoint actually exists -- eval_friction_sweep.py treats
# an omitted flag as "skip this case" (Optional[str] = None), but a path to a missing file still
# crashes torch.load. This lets the job run against whichever cases have finished training instead
# of requiring all three every time.
EXTRA_CKPT_ARGS=()
[[ -f "$CKPT_NO_FORCE" ]] && EXTRA_CKPT_ARGS+=(--ckpt-no-force "$CKPT_NO_FORCE")
[[ -f "$CKPT_FORCE" ]] && EXTRA_CKPT_ARGS+=(--ckpt-force "$CKPT_FORCE")
[[ -f "$CKPT_FORCE_HEAD" ]] && EXTRA_CKPT_ARGS+=(--ckpt-force-head "$CKPT_FORCE_HEAD")

python eval_friction_sweep.py \
  --env-id "$ENV_ID" \
  --control-mode "$CONTROL_MODE" \
  --sim-backend physx_cuda \
  --num-eval-episodes 100 \
  --num-eval-envs 25 \
  --max-episode-steps "${MAX_EPISODE_STEPS:-200}" \
  "${EXTRA_CKPT_ARGS[@]}" \
  --output-dir "runs/friction_sweep_eval${EXP_SUFFIX}_${CONTROL_MODE}_ckpt${CKPT_TAG}"
