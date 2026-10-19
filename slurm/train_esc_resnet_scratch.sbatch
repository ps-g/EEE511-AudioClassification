#!/bin/bash
# One ESC-50 fold per array task: ResNet-50 from scratch, 450 epochs,
# LR drops 10x at epochs 300 and 350 (see config/esc_resnet_scratch.json).
#SBATCH --job-name=esc-resnet-scratch
#SBATCH --partition=public
#SBATCH --array=1-5
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4-00:00:00
#SBATCH --output=slurm/logs/%x-%A_%a.out
set -euo pipefail

# ~/.local has a torch 2.7.0 that shadows the env's torch 2.3.1 and breaks torchaudio
export PYTHONNOUSERSITE=1
PYTHON=/home/scmcdowe/.conda/envs/pt/bin/python

cd /home/scmcdowe/projects/EEE511-AudioClassification
$PYTHON train.py --config_path config/esc_resnet_scratch.json --fold "$SLURM_ARRAY_TASK_ID"
