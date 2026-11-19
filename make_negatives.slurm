#!/bin/bash
#SBATCH --job-name=negatives
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00

source ~/.bashrc
conda activate flow_grpo 2>/dev/null
# conda activate does not work in batch shells here; put the env on PATH explicitly
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false

# Extra args are forwarded, e.g.:
#   sbatch make_negatives.slurm [--limit 64]
python scripts/phase1_make_negatives.py "$@"
