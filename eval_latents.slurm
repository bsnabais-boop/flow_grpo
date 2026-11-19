#!/bin/bash
#SBATCH --job-name=eval_latents
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
#   sbatch eval_latents.slurm --lora logs/ocr/sd1.5/checkpoints/checkpoint-240/lora
#   sbatch eval_latents.slurm --num-steps 40 --eta 0
python scripts/eval_intermediate_latents.py "$@"
