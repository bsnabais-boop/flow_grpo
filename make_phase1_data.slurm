#!/bin/bash
#SBATCH --job-name=phase1_data
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

source ~/.bashrc
conda activate flow_grpo 2>/dev/null
# conda activate does not work in batch shells here; put the env on PATH explicitly
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false

# Extra args are forwarded, e.g.:
#   for i in 0 1 2 3; do sbatch make_phase1_data.slurm --shard $i --num-shards 4; done
python scripts/phase1_make_data.py "$@"
