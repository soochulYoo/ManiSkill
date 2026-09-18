#!/bin/bash
#SBATCH --job-name=quick_compliance_gain_check
#SBATCH --exclude=cs-gpu-01
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:30:00
#SBATCH --output=logs/quick_compliance_gain_check_%j.log

# One-off, cheap sanity check for whether raising arm_ee_pos_stiffness/damping
# (mani_skill/agents/robots/panda/panda.py) let pd_ee_pose_compliance actually complete PushCube
# at all -- only the no_force checkpoint, only 10 episodes per condition, not the full sweep.

source /scratch2/soochul/ManiSkill/.venv/bin/activate
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.x86_64.json

cd /scratch2/soochul/ManiSkill/examples/baselines/act
python eval_friction_sweep.py \
  --env-id PushCube-v1 \
  --control-mode pd_ee_pose_compliance \
  --sim-backend physx_cuda \
  --num-eval-episodes 10 \
  --num-eval-envs 10 \
  --ckpt-no-force runs/act_no_force_n600_i40000/checkpoints/best_eval_success_once.pt \
  --output-dir runs/quick_compliance_gain_check
