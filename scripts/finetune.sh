#!/bin/bash
#SBATCH --job-name=depth-v2-train
#SBATCH --output=logs/.depth_v2_train_%j.out
#SBATCH --error=logs/.depth_v2_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=24:00:00

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

uv sync

# Activate conda environment (adjust to your environment name)
source .venv/bin/activate
# OR: conda activate depth_env

# Run the training script
python train.py \
    --model vits \
    --dataset-path /scratch/soccernet/ \
    --sport-name basketball \
    --seed 42 \
    --train-batch-size 2 \
    --val-batch-size 16 \
    --epochs 15 \
    --backbone-lr 5e-6 \
    --head-lr 1e-4 \
    --weight-decay 1e-5 \
    --experiment-name "depth_v2_masking_70_percent_switch_3_${SLURM_JOB_ID}" \
    --use-masking \
    # --use-wandb \
    # --use-cutmix \
    # --use-registers

# Print end time
echo "End time: $(date)"
