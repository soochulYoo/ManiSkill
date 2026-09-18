#!/bin/bash
#SBATCH --job-name=replay_pd_ee_pose
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/replay_pd_ee_pose_%j.log

# --time is a rough guess (not benchmarked at full scale) -- check logs/replay_pd_ee_pose_<jobid>.log
# partway through and adjust before resubmitting if it's way off.

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

bash /scratch2/soochul/ManiSkill/scripts/data_generation/replay_pd_ee_pose.sh
