#!/bin/bash
#SBATCH --job-name=p3_ssched
#SBATCH --output=logs/slurm_%j.out   # Guarda o output num ficheiro com o ID do Job
#SBATCH --error=logs/slurm_%j.err    # Guarda os erros num ficheiro separado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Aumentar CPU (8 núcleos)
#SBATCH --mem=64G                    # Aumentar a RAM do sistema para 64GB
#SBATCH --gres=gpu:1                 # Pede 1 GPU EXCLUSIVO (Impede OOMs causados por outros)
#SBATCH --time=24:00:00


source ~/.bashrc
conda activate flow_grpo 2>/dev/null
# conda activate does not work in batch shells here; put the env on PATH explicitly
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH


cd /user/home/b.nabais/flow_grpo


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline


echo "Starting..."

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 1 --main_process_port 29532 scripts/train_sd15.py \
  --config config/grpo.py:mixed_sd15_prm_ssched_1gpu \
  --config.activation_checkpointing=True \
  --config.train.use_8bit_adam=True \
  --config.num_epochs=100