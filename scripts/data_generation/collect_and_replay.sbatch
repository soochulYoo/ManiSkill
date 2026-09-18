#!/bin/bash
#SBATCH --job-name=collect_and_replay
#SBATCH --exclude=cs-gpu-01,node05
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=logs/collect_and_replay_%j.log

# node05 excluded: hit a Vulkan/GPU-driver failure there during the PushCube pd_joint_pos replay
# (see git history). --time is a rough guess -- 200 demos should be well under the 15min this took
# for 600 PushCube demos, but check logs/collect_and_replay_<jobid>.log partway through.
#
# Usage:
#   ENV_ID=PullCube-v1 NUM_DEMOS=200 sbatch scripts/data_generation/collect_and_replay.sbatch
#   ENV_ID=PegInsertionSide-v1 NUM_DEMOS=200 sbatch scripts/data_generation/collect_and_replay.sbatch

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill
bash scripts/data_generation/collect_and_replay.sh
