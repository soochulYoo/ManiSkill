#!/bin/bash
#SBATCH --job-name=pushcube_demo_collection
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/demo_collection_%A_%a.log


source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

bash /scratch2/soochul/ManiSkill/scripts/data_generation/motionplanning_friction_density.sh
