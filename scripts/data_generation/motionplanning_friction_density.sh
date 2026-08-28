#!/usr/bin/env bash
# Sweeps object density / table friction and collects motion-planning demonstrations for each
# condition via mani_skill.examples.motionplanning.panda.run. Requires PushCube-v1 (or any other
# env that accepts obj_density/static_friction/dynamic_friction kwargs) and must be run with bash
# (uses the `read ... <<<` here-string, which plain `sh`/dash does not support).
#
# Run standalone to sweep all conditions sequentially:
#   bash scripts/data_generation/motionplanning_friction_density.sh
# Run as a SLURM job array task (see motionplanning_friction_density.sbatch) to run just the one
# condition at index $SLURM_ARRAY_TASK_ID:
#   sbatch scripts/data_generation/motionplanning_friction_density.sbatch
set -euo pipefail

# --- Conditions to sweep ---
# Format: "density static_friction dynamic_friction"
CONDITIONS=(
    # "1000 0.5 0.35"
    # "1000 1.0 0.7"
    # "1000 2.0 1.4"
    "1000 3.3 2.3"
    # "1000 4.5 3.15"
    # "1000 6.0 4.2"
    # "1000 8.0 5.6"
)

# add more env ids here to sweep the same conditions across other supported tasks
ENV_IDS=(PushCube-v1)

# When run as a SLURM job array task, SLURM_ARRAY_TASK_ID selects a single condition to run
# (so the .sbatch file can parallelize one condition per job). Otherwise all conditions run
# sequentially in this one process.
if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    INDICES=("$SLURM_ARRAY_TASK_ID")
else
    INDICES=("${!CONDITIONS[@]}")
fi

for i in "${INDICES[@]}"; do
    read -r DENSITY STATIC DYNAMIC <<< "${CONDITIONS[$i]}"
    echo "[density=$DENSITY static=$STATIC dynamic=$DYNAMIC]"

    for env_id in "${ENV_IDS[@]}"; do
        python -m mani_skill.examples.motionplanning.panda.run \
            --env-id "$env_id" \
            --traj-name="trajectory" \
            -n 100 \
            --only-count-success \
            --obj-density "$DENSITY" \
            --static-friction "$STATIC" \
            --dynamic-friction "$DYNAMIC" \
            --record-dir "demos_friction/density_${DENSITY}_friction_${STATIC}_${DYNAMIC}"
    done
done
