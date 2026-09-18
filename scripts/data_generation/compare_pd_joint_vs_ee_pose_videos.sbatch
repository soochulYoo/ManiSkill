#!/bin/bash
#SBATCH --job-name=compare_pd_joint_vs_ee_pose_videos
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=06:00:00
#SBATCH --output=logs/compare_pd_joint_vs_ee_pose_videos_%j.log

# --time is a rough guess (122 episodes x 2 control modes to replay+render, then combine) --
# check logs/compare_pd_joint_vs_ee_pose_videos_<jobid>.log partway through and adjust before
# resubmitting if it's way off.

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill
bash scripts/data_generation/compare_pd_joint_vs_ee_pose_videos.sh
