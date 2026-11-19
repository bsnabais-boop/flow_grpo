#!/bin/bash
#SBATCH --job-name=ocr_easy
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
# Robustness check of the option-1 OCR gain: strict OCR metrics, repetition, second recogniser (TrOCR), 10/15/20 steps.
set -e
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export PYTHONPATH=/user/home/b.nabais/flow_grpo/third_party/easyocr_pkgs:$PYTHONPATH
export TOKENIZERS_PARALLELISM=false
L=logs/phase3/ocr_long_win23/checkpoints/checkpoint-500/lora
python scripts/ocr_robustness.py --out logs/ocr_robustness/v2_easyocr --easyocr third_party/easyocr_models --setting \
  base_10:none:10 long500_10:$L:10 base_15:none:15 long500_15:$L:15 base_20:none:20 long500_20:$L:20 base_50:none:50
