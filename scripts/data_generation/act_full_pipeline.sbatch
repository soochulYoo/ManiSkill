#!/bin/bash
#SBATCH --job-name=act_full_pipeline
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/act_full_pipeline_%j.log

# Runs the whole ACT pipeline sequentially in one job, in one process, so it can be submitted and
# left to run unattended:
#   1. Collect NUM_DEMOS motion-planning demos for PushCube-v1 (no --static-friction/
#      --dynamic-friction/--obj-density override -- letting the env fall back to its own
#      defaults, per mani_skill/examples/motionplanning/panda/run.py:44-46).
#   2. Replay them via replay_trajectory (same as
#      scripts/data_generation/replay_pd_joint_pos.sh) to attach rgbd observations, WITHOUT
#      changing control mode -- pd_joint_pos in, pd_joint_pos out, so this is just a
#      same-controller re-run (no IK re-solve). Uses --allow-failure: without it, episodes whose
#      replay doesn't reproduce the recorded success flag are silently dropped from
#      RecordEpisode's flush watermark, and their steps get spliced onto the NEXT successful
#      episode's saved trajectory (see replay_pd_joint_pos.sh's comment for the full mechanism).
#      --allow-failure flushes every episode on its own, avoiding that entirely.
#   3. Train the 3 ACT variants (no_force / force / force_head) sequentially on the replayed
#      demos, for TOTAL_ITERS iterations each.
#   4. Evaluate all 3 checkpoints across the friction sweep under pd_joint_pos (rigid), with the
#      force_head_ttt case additionally run under pd_joint_pos_soft (see panda.py's
#      arm_soft_stiffness/arm_soft_damping) to give TTT a more friction-informative force signal
#      to adapt to.
#
# NUM_DEMOS/TOTAL_ITERS are overridable via env vars so the same script can run a fast smoke test
# before committing to the full run, e.g.:
#   NUM_DEMOS=50 TOTAL_ITERS=10 sbatch scripts/data_generation/act_full_pipeline.sbatch
#   NUM_DEMOS=600 TOTAL_ITERS=40000 sbatch scripts/data_generation/act_full_pipeline.sbatch
# RECORD_DIR and the training exp-names are namespaced by NUM_DEMOS/TOTAL_ITERS automatically so
# a smoke test never overwrites or mixes with a full run's demos/checkpoints.
#
# --time is a rough guess (not benchmarked at NUM_DEMOS=600/TOTAL_ITERS=40000 on this cluster) --
# check logs/act_full_pipeline_<jobid>.log partway through and adjust before resubmitting if it's
# way off.

set -euo pipefail

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill

NUM_DEMOS="${NUM_DEMOS:-600}"
TOTAL_ITERS="${TOTAL_ITERS:-40000}"
RECORD_DIR="${RECORD_DIR:-demos_friction/default_n${NUM_DEMOS}}"
RUN_TAG="n${NUM_DEMOS}_i${TOTAL_ITERS}"

DEMO_DIR="$RECORD_DIR/PushCube-v1/motionplanning"
RAW_TRAJ="$DEMO_DIR/trajectory.h5"
RGBD_TRAJ="$DEMO_DIR/trajectory.rgbd.pd_joint_pos.physx_cpu.h5"

echo "############################################################"
echo "# [1/4] Collecting $NUM_DEMOS motion-planning demos (default friction/density) -> $RECORD_DIR"
echo "############################################################"
python -m mani_skill.examples.motionplanning.panda.run \
  --env-id PushCube-v1 \
  --traj-name trajectory \
  -n "$NUM_DEMOS" \
  --only-count-success \
  --record-dir "$RECORD_DIR"

echo "############################################################"
echo "# [2/4] Replaying $RAW_TRAJ -> rgbd (control mode unchanged: pd_joint_pos)"
echo "############################################################"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$RAW_TRAJ" \
  -o rgbd \
  -c pd_joint_pos \
  --use-first-env-state \
  --allow-failure \
  --save-traj

cd examples/baselines/act

CASES=(no_force force force_head)
for CASE in "${CASES[@]}"; do
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

  echo "############################################################"
  echo "# [3/4] Training act_${CASE}_${RUN_TAG} for $TOTAL_ITERS iterations"
  echo "############################################################"
  python train_rgbd.py \
    --env-id PushCube-v1 \
    --demo-path "/scratch2/soochul/ManiSkill/$RGBD_TRAJ" \
    --control-mode pd_joint_pos \
    --exp-name "act_${CASE}_${RUN_TAG}" \
    --save-freq 20000 \
    --sim-backend physx_cuda \
    --max-episode-steps 200 \
    --total-iters "$TOTAL_ITERS" \
    --track \
    --wandb-entity usam205_ \
    "${EXTRA_ARGS[@]}"
done

# Picks the best checkpoint available for a run: best_eval_success_once.pt is only written if
# training ever beat its running-best success_once rate during a periodic eval (0.0 > 0.0 never
# fires, so a run that never sees a single success -- e.g. a very short smoke-test run -- never
# creates it). 0.pt is the unconditional iteration-0 checkpoint and always exists as a fallback.
pick_ckpt() {
  local dir="runs/$1/checkpoints"
  if [[ -f "$dir/best_eval_success_once.pt" ]]; then
    echo "$dir/best_eval_success_once.pt"
  elif [[ -f "$dir/best_eval_success_at_end.pt" ]]; then
    echo "$dir/best_eval_success_at_end.pt"
  else
    echo "$dir/0.pt"
  fi
}

CKPT_NO_FORCE=$(pick_ckpt "act_no_force_${RUN_TAG}")
CKPT_FORCE=$(pick_ckpt "act_force_${RUN_TAG}")
CKPT_FORCE_HEAD=$(pick_ckpt "act_force_head_${RUN_TAG}")

echo "############################################################"
echo "# [4/4] Evaluating across friction sweep"
echo "############################################################"
echo "Using checkpoints: $CKPT_NO_FORCE | $CKPT_FORCE | $CKPT_FORCE_HEAD"
python eval_friction_sweep.py \
  --env-id PushCube-v1 \
  --control-mode pd_joint_pos \
  --control-mode-ttt pd_joint_pos_soft \
  --sim-backend physx_cuda \
  --num-eval-episodes 100 \
  --num-eval-envs 25 \
  --max-episode-steps 200 \
  --ckpt-no-force "$CKPT_NO_FORCE" \
  --ckpt-force "$CKPT_FORCE" \
  --ckpt-force-head "$CKPT_FORCE_HEAD" \
  --output-dir "runs/friction_sweep_eval_${RUN_TAG}"

echo "############################################################"
echo "# Pipeline complete: $RUN_TAG"
echo "############################################################"
