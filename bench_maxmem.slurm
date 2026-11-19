#!/bin/bash
#SBATCH --job-name=bench_mem
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=nsdl2
#SBATCH --time=02:00:00
# Max-memory benchmark (2026-11-19): larger sampling/training batches and more images per epoch (1 optimizer step per
# epoch), bf16 autocast + OCR pool. Epoch 1 is the measurement.
set +e
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
OUT=logs/bench_maxmem
mkdir -p $OUT

run() {  # tag sample_bs n_batches train_bs micro_batches ocr_workers
  tag=$1; S=$2; N=$3; T=$4; M=$5; W=$6
  echo "=== $tag: sample_bs=$S batches=$N train_bs=$T micro=$M ocr_workers=$W ===" | tee -a $OUT/summary.txt
  accelerate launch --num_processes 1 --main_process_port 29573 scripts/train_sd15.py \
    --config config/grpo.py:mixed_long_win23_wsched_1gpu \
    --config.activation_checkpointing=True --config.train.use_8bit_adam=True \
    --config.num_epochs=2 --config.save_freq=1000 --config.save_dir=$OUT/$tag \
    --config.sample.train_batch_size=$S --config.sample.num_batches_per_epoch=$N \
    --config.train.batch_size=$T --config.train.micro_batches=$M --config.train.gradient_accumulation_steps=$M \
    --config.sample.ocr_workers=$W --config.lora_autocast=True > $OUT/$tag.log 2>&1
  echo "exit=$?" >> $OUT/summary.txt
  grep "\[TIMING\]" $OUT/$tag.log >> $OUT/summary.txt
  grep -m1 -E "OutOfMemory|CUDA out of memory|Error" $OUT/$tag.log >> $OUT/summary.txt
}

rm -f $OUT/summary.txt
nvidia-smi --query-gpu=name,memory.total --format=csv >> $OUT/summary.txt
# tag            S   N  T   M  W     (images/epoch = S*N; micro_batches M = S*N/T)
run i512_s64_t64   64  8  64  8  8   # 512 imgs, train batch 64 with bf16
run i1024_s128_t64 128 8  64 16 8    # 1024 imgs
run i1024_s128_t128 128 8 128  8 8
run i2048_s256_t128 256 8 128 16 16  # 2048 imgs
cat $OUT/summary.txt
