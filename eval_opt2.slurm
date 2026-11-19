#!/bin/bash
#SBATCH --job-name=opt2_eval
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
# Option 2 pilot eval: every checkpoint (seed 777, 4 imgs/prompt, vs logs/phase3_eval/base_10steps) and the final,
# pre-chosen checkpoint with new seeds (seed 2024, 8 imgs/prompt, vs logs/phase3_confirm/base_10steps). Run on nsdl2.
# Usage: sbatch eval_opt2.slurm <run name under logs/opt2> <final step>
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
run=$1; final=$2
for ck in $(ls -d logs/opt2/$run/checkpoints/checkpoint-* | sort -t- -k2 -n); do
  out=logs/phase3_eval/opt2_${run}_$(basename $ck)
  [ -f $out/summary.json ] && continue
  echo "$(date '+%F %T') eval $ck"
  python scripts/phase3_eval.py --num-steps 10 --lora $ck/lora --out $out
done
python scripts/phase3_eval.py --num-steps 10 --lora logs/opt2/$run/checkpoints/checkpoint-$final/lora \
  --images-per-prompt 8 --seed 2024 --out logs/phase3_confirm/opt2_${run}_checkpoint-$final
