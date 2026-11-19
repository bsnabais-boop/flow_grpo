#!/bin/bash
#SBATCH --job-name=opt2_proxy
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
# Option 2 step 1: does a differentiable text reward (TrOCR NLL / CLIP) agree with the PaddleOCR reward?
source ~/.bashrc
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
set -e
python scripts/opt2_proxy_check.py "$@"
