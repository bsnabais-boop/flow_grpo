#!/bin/bash
# Watches the Phase 3 GRPO runs and submits scripts/phase3_eval.py (10 steps, paired with logs/phase3_eval/base_10steps)
# for every new checkpoint. Survives the Claude session: run with
#   nohup setsid bash scripts/phase3_autoeval.sh > logs/phase3_autoeval.log 2>&1 &
# Stop: pkill -f phase3_autoeval.sh
cd /user/home/b.nabais/flow_grpo
while true; do
  for run in mixed_orm mixed_prm; do
    for ck in logs/phase3/$run/checkpoints/checkpoint-*; do
      [ -f "$ck/lora/adapter_model.safetensors" ] || continue
      name=$(basename "$ck")
      out=logs/phase3_eval/${run}_${name}
      [ -e "$out" ] && continue
      mkdir -p "$out"
      sleep 30  # make sure the checkpoint is fully written
      job=$(sbatch --parsable --nodelist=nsdl2 eval_phase3.slurm --num-steps 10 --lora "$ck/lora" --out "$out")
      echo "$(date '+%F %T') submitted $job for $ck -> $out"
    done
  done
  # stop when both trainings are gone
  if [ -z "$(squeue -h -u $USER -n p3_orm,p3_prm)" ]; then echo "trainings finished"; break; fi
  sleep 300
done
