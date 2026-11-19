#!/bin/bash
#SBATCH --job-name=bench_eff
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=nsdl2
#SBATCH --time=02:00:00
# Efficiency benchmark (2026-11-18): time per epoch of the mixed long-run recipe (512 images, 1 optimizer step per
# epoch) for several sampling/training batch sizes and OCR worker counts. 2 epochs per setting; epoch 1 is the
# measurement (epoch 0 includes warm-up). Results: logs/bench_efficiency/summary.txt
set +e
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false
OUT=logs/bench_efficiency
mkdir -p $OUT

run() {  # tag sample_bs n_batches train_bs micro_batches ocr_workers
  tag=$1; S=$2; N=$3; T=$4; M=$5; W=$6
  echo "=== $tag: sample_bs=$S batches=$N train_bs=$T micro=$M ocr_workers=$W ===" | tee -a $OUT/summary.txt
  accelerate launch --num_processes 1 --main_process_port 29571 scripts/train_sd15.py \
    --config config/grpo.py:mixed_long_win23_wsched_1gpu \
    --config.activation_checkpointing=True --config.train.use_8bit_adam=True \
    --config.num_epochs=2 --config.save_freq=1000 --config.save_dir=$OUT/$tag \
    --config.sample.train_batch_size=$S --config.sample.num_batches_per_epoch=$N \
    --config.train.batch_size=$T --config.train.micro_batches=$M --config.train.gradient_accumulation_steps=$M \
    --config.sample.ocr_workers=$W > $OUT/$tag.log 2>&1
  echo "exit=$?" >> $OUT/summary.txt
  grep "\[TIMING\]" $OUT/$tag.log >> $OUT/summary.txt
  grep -m1 -E "OutOfMemory|CUDA out of memory|Error" $OUT/$tag.log >> $OUT/summary.txt
}

rm -f $OUT/summary.txt
nvidia-smi --query-gpu=name,memory.total --format=csv >> $OUT/summary.txt
run b8_w0   8  64  8 64  0    # current setting
run b8_w8   8  64  8 64  8
run b16_w8  16 32 16 32  8
run b32_w8  32 16 32 16  8
run b64_w8  64  8 64  8  8
run b64_t32_w8 64 8 32 16 8  # sample 64 at a time, train in micro-batches of 32
run b32_w16 32 16 32 16 16
cat $OUT/summary.txt
