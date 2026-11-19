#!/bin/bash
#SBATCH --job-name=diag_variants
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00

# "Problem 2" (2026-11-13): per-image gradient signal/noise of different sampling schemes, paired on the same
# 64 text prompts x 8 images (seeded initial noise), base model, config of the current run (nc_wsched_fix).
#   v0 independent z_T, stochastic at all steps, last transition skipped  (= what training does now)
#   v1 shared z_T within a group, same steps
#   v2 independent z_T, SDE window steps 1-4   (window alone)
#   v3 shared z_T, SDE window steps 1-4
#   v4 shared z_T, SDE window steps 2-3
source ~/.bashrc
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
OUT=logs/diag_ocr_signal/v3_variants
C="--config mixed_sd15_nc_wsched_fix_1gpu --models base --text-prompts 64 --align-prompts 0 --out $OUT"
python scripts/diag_ocr_signal.py $C --skip-last 1 --tag _v0_indep_full
python scripts/diag_ocr_signal.py $C --skip-last 1 --tag _v1_shared_full --same-latent
python scripts/diag_ocr_signal.py $C --skip-last 0 --tag _v2_indep_win1-4 --sde-window-size 4 --sde-window-start 1
python scripts/diag_ocr_signal.py $C --skip-last 0 --tag _v3_shared_win1-4 --sde-window-size 4 --sde-window-start 1 --same-latent
python scripts/diag_ocr_signal.py $C --skip-last 0 --tag _v4_shared_win2-3 --sde-window-size 2 --sde-window-start 2 --same-latent
