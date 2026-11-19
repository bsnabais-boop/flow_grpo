#!/bin/bash
#SBATCH --job-name=phase3_eval_all
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
# Evaluates, one after another in this single job, every checkpoint of the given Phase 3 runs that has no eval yet.
# Usage: sbatch eval_phase3_all.slurm mixed_prm_both [more runs]
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
for run in "$@"; do
  for ck in $(ls -d logs/phase3/$run/checkpoints/checkpoint-* | sort -t- -k2 -n); do
    out=logs/phase3_eval/${run}_$(basename $ck)
    [ -f $out/summary.json ] && continue
    [ -f $ck/lora/adapter_model.safetensors ] || continue
    echo "$(date '+%F %T') eval $ck"
    python scripts/phase3_eval.py --num-steps 10 --lora $ck/lora --out $out
  done
done
