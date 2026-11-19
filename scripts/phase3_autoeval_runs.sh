#!/bin/bash
# Like phase3_autoeval.sh, for any list of Phase 3 runs. Usage (survives the Claude session):
#   nohup setsid bash scripts/phase3_autoeval_runs.sh "mixed_prm_wsched mixed_prm_ssched mixed_prm_both" "p3_wsched,p3_ssched,p3_both" > logs/phase3_autoeval2.log 2>&1 &
# Evaluates every new checkpoint at 10 steps on nsdl2 (paired with logs/phase3_eval/base_10steps).
cd /user/home/b.nabais/flow_grpo
RUNS="$1"; JOBNAMES="$2"
started=0
while true; do
  for run in $RUNS; do
    for ck in logs/phase3/$run/checkpoints/checkpoint-*; do
      [ -f "$ck/lora/adapter_model.safetensors" ] || continue
      out=logs/phase3_eval/${run}_$(basename "$ck")
      [ -e "$out" ] && continue
      mkdir -p "$out"; sleep 30
      job=$(sbatch --parsable --nodelist=nsdl2 eval_phase3.slurm --num-steps 10 --lora "$ck/lora" --out "$out")
      echo "$(date '+%F %T') submitted $job for $ck -> $out"
    done
  done
  if [ -n "$(squeue -h -u $USER -n $JOBNAMES)" ]; then started=1; fi
  if [ $started = 1 ] && [ -z "$(squeue -h -u $USER -n $JOBNAMES)" ]; then echo "trainings finished"; break; fi
  sleep 300
done
