#!/bin/bash
#SBATCH --job-name=opt2_draft
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
# Option 2: direct reward back-propagation (TrOCR NLL) for text rendering. Args forwarded to train_sd15_draft.py.
source ~/.bashrc
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
set -e
python scripts/train_sd15_draft.py "$@"
nvidia-smi --query-gpu=name,memory.used --format=csv
