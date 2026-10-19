#!/bin/bash
#SBATCH --job-name=esc-prep
#SBATCH --partition=public
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x-%j.out
set -euo pipefail

# ~/.local has a torch 2.7.0 that shadows the env's torch 2.3.1 and breaks torchaudio
export PYTHONNOUSERSITE=1
PYTHON=/home/scmcdowe/.conda/envs/pt/bin/python

cd /home/scmcdowe/projects/EEE511-AudioClassification
$PYTHON preprocessing/preprocessingESC.py \
    --csv_file /scratch/scmcdowe/ESC-50/meta/esc50.csv \
    --data_dir /scratch/scmcdowe/ESC-50/audio/ \
    --store_dir /scratch/scmcdowe/esc50_spectrograms/ \
    --sampling_rate 44100
