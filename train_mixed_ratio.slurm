#!/bin/bash
#SBATCH --job-name=mix_ratio
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=nsdl2
#SBATCH --time=40:00:00
# Weight study (2026-11-18): sbatch train_mixed_ratio.slurm <config: mixed_ratio_08_1gpu | _16_ | _32_> <port> [extra overrides]
# Efficient settings from logs/bench_efficiency (same 512 imgs + 1 optimizer step per epoch as before): sampling and
# training in batches of 32 (16 micro-batches), PaddleOCR on 8 worker processes.
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
cfg=$1; port=$2; shift 2
accelerate launch --num_processes 1 --main_process_port $port scripts/train_sd15.py \
  --config config/grpo.py:$cfg \
  --config.activation_checkpointing=True --config.train.use_8bit_adam=True \
  --config.sample.train_batch_size=32 --config.sample.num_batches_per_epoch=16 \
  --config.train.batch_size=32 --config.train.micro_batches=16 --config.train.gradient_accumulation_steps=16 \
  --config.sample.ocr_workers=8 "$@"
