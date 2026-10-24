#!/bin/bash
#SBATCH --job-name=grpo_sd35
#SBATCH --output=logs/slurm_%j.out   # Guarda o output num ficheiro com o ID do Job
#SBATCH --error=logs/slurm_%j.err    # Guarda os erros num ficheiro separado
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # Aumentar CPU (8 núcleos)
#SBATCH --mem=64G                    # Aumentar a RAM do sistema para 64GB
#SBATCH --gres=gpu:1                 # Pede 1 GPU EXCLUSIVO (Impede OOMs causados por outros)
#SBATCH --time=24:00:00              # Tempo máximo do job (24 horas)


source ~/.bashrc
conda activate flow_grpo


cd /user/home/b.nabais/flow_grpo


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=offline


echo "Starting..."

export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 1 scripts/train_sd3_fast.py \
  --config config/grpo.py:general_ocr_sd3 \
  --config.activation_checkpointing=True \
  --config.train.use_8bit_adam=True