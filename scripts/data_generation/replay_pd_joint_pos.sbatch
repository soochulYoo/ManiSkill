#!/bin/bash
#SBATCH --job-name=replay_pd_joint_pos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=02:00:00
#SBATCH --output=logs/replay_pd_joint_pos_%j.log

# --time is a rough guess -- this replay is much cheaper than replay_pd_ee_pose.sbatch's since
# there's no IK re-solve, just re-running the same pd_joint_pos actions to render rgbd obs. Check
# logs/replay_pd_joint_pos_<jobid>.log partway through and adjust before resubmitting if it's way
# off.

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

bash /scratch2/soochul/ManiSkill/scripts/data_generation/replay_pd_joint_pos.sh
