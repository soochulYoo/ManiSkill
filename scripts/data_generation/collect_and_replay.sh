#!/usr/bin/env bash
# Generalized version of the PushCube collection+replay steps used by act_full_pipeline.sbatch,
# parametrized by ENV_ID so it works for any task with a registered motion-planning solution
# (mani_skill.examples.motionplanning.panda.solutions.MP_SOLUTIONS) and a Panda-family robot.
# Collects NUM_DEMOS demos with the env's own defaults (no friction/density/clearance override --
# those are meant to be swept at EVAL time only, not collection time, same as PushCube's pipeline),
# then replays them in-place (same control mode in and out) purely to attach rgbd observations.
#
# --allow-failure on the replay is REQUIRED, not optional -- see
# scripts/data_generation/replay_pd_joint_pos.sh's comment for the full mechanism (RecordEpisode's
# flush watermark only advances on an actual flush, so without --allow-failure, episodes that fail
# to reproduce their recorded success flag get spliced onto the next successful episode's saved
# trajectory instead of being cleanly excluded).
#
# Run standalone:
#   ENV_ID=PullCube-v1 NUM_DEMOS=200 bash scripts/data_generation/collect_and_replay.sh
# Or via the SLURM wrapper:
#   ENV_ID=PullCube-v1 NUM_DEMOS=200 sbatch scripts/data_generation/collect_and_replay.sbatch
set -euo pipefail

ENV_ID="${ENV_ID:?must set ENV_ID, e.g. PullCube-v1 or PegInsertionSide-v1}"
NUM_DEMOS="${NUM_DEMOS:-200}"
CONTROL_MODE="${CONTROL_MODE:-pd_joint_pos}"
ENV_TAG="$(echo "${ENV_ID%-v*}" | tr '[:upper:]' '[:lower:]')"
RECORD_DIR="${RECORD_DIR:-demos_${ENV_TAG}/default_n${NUM_DEMOS}}"

DEMO_DIR="$RECORD_DIR/$ENV_ID/motionplanning"
RAW_TRAJ="$DEMO_DIR/trajectory.h5"

echo "############################################################"
echo "# Collecting $NUM_DEMOS motion-planning demos for $ENV_ID -> $RECORD_DIR"
echo "############################################################"
python -m mani_skill.examples.motionplanning.panda.run \
  --env-id "$ENV_ID" \
  --traj-name trajectory \
  -n "$NUM_DEMOS" \
  --only-count-success \
  --control-mode "$CONTROL_MODE" \
  --record-dir "$RECORD_DIR"

echo "############################################################"
echo "# Replaying $RAW_TRAJ -> rgbd (control mode unchanged: $CONTROL_MODE)"
echo "############################################################"
python -m mani_skill.trajectory.replay_trajectory \
  --traj-path "$RAW_TRAJ" \
  -o rgbd \
  -c "$CONTROL_MODE" \
  --use-first-env-state \
  --allow-failure \
  --save-traj

echo "Done. Replayed file: $DEMO_DIR/trajectory.rgbd.${CONTROL_MODE}.physx_cpu.h5"
