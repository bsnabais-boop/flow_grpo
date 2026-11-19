#!/bin/bash
#SBATCH --job-name=stitchvm
#SBATCH --output=logs/slurm_%j.out
#SBATCH --error=logs/slurm_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=nsdl2
#SBATCH --time=08:00:00
# StitchVM for SD1.5 (2026-11-18): data -> stitch-point search -> training (A: NoisyCLIP tail, main; B: OpenAI CLIP tail,
# ablation) -> evaluation in the phase-1 harness (same settings as logs/phase1_noisyclip/eval10_noisyclip, rerun on
# the H200 so NoisyCLIP x0phi / TAESD / VAE are paired with the StitchVMs). Stages are skipped if their output exists.
set -e
export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH
cd /user/home/b.nabais/flow_grpo
export TOKENIZERS_PARALLELISM=false
O=logs/stitchvm
[ -f $O/data.npz ] || python scripts/stitchvm_build.py data
for C in asiimo/noisyclip openai/clip-vit-large-patch14; do
  T=${C//\//_}
  [ -f $O/search_$T.json ] || python scripts/stitchvm_build.py search --clip $C
done
python scripts/stitchvm_build.py train --clip asiimo/noisyclip --name A_noisyclip
python scripts/stitchvm_build.py train --clip openai/clip-vit-large-patch14 --name B_openai
python scripts/phase1_noisyclip.py --num-prompts 128 --images-per-prompt 4 --batch-size 8 --num-steps 10 --eta 1.0 \
  --decode-every 5 --seed 0 --noisyclip asiimo/noisyclip --phi logs/phase1_noisyclip/sd15_50steps/phi.npy \
  --stitchvm $O/A_noisyclip $O/B_openai --out logs/phase1_noisyclip/eval10_stitchvm
