#!/bin/bash
#SBATCH --job-name=debug_compliance_render
#SBATCH --partition=suma_rtx4090
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:20:00
#SBATCH --output=logs/debug_compliance_render_%j.log

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill
python scripts/data_generation/debug_compliance_render.py
