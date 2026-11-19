# Project: cheaper text-in-image generation with Flow-GRPO (thesis project)

## Main goal (from the user, verbatim)

> **Steering the "Latent Spark" (Early Trajectory Optimization):** Focus Flow GRPO policy updates on the critical
> initial flow-matching timesteps ($t \in [1.0, 0.5]$)—where global layout, spatial composition, and character
> structures form—to generate legible text in significantly fewer total sampling steps.
>
> **Eliminating VAE Overhead via Noisy CLIP:** Build directly on your colleague's Noisy CLIP research to evaluate
> text-image semantic alignment directly on noisy latents ($z_t$), bypassing full image generation rollouts and
> eliminating $100\%$ of intermediate VAE decoding bottlenecks during RL reward evaluation.
>
> **Compute Cost & Step Reduction:** Reduce the total number of ODE denoiser steps required for high-quality
> text-in-image generation (e.g., cutting generation down from 40–50 steps to 10–15 steps), drastically lowering
> inference latency, VRAM consumption, and total energy usage.

## Project plan: "Steering the Latent Spark: Process-Oriented Alignment for Diffusion Models"

Source: supervisor's slides + proposal doc (Google Slides 1soa9_17N..., Google Doc 1Aouj6_MlxwzzRavaUyzzluFNrF2YUG6WIcFDaDONiOU).
Lab: João Magalhães (NOVA). The "colleague" is **Vasco Ramos**; NoisyCLIP = *Ramos, Cohen, Szpektor, Magalhães,
"Beyond the Noise: Aligning Prompts with Latent Representations in Diffusion Models", ECCV 2026*.
(`Thesis_Plan.pdf` in the repo root is a corrupted upload of Vasco's unrelated 2024 dissertation plan on
instructional video; ignore it.)

Idea: move from Outcome Reward Models (score only the final image) to a **Process Reward Model (PRM)** that scores
text-image alignment on intermediate noisy latents z_t, and use it inside GRPO for step-level credit assignment,
early detection of misalignment and pruning of doomed trajectories (compute savings).

- **Phase 1 - Noise-aware PRM (NoisyCLIP).** Score alignment of z_t with the prompt without a VAE decode.
  Option A (NoisyCLIP): project the 4-channel latent (via the predicted clean state x0_hat) to RGB with a fixed
  linear matrix Phi(z_t), then CLIP; reported ~90% correlation with the final score midway (t ~ 25 of 50). NoisyCLIP
  is trained on an LLM-generated dataset of aligned / subtly and majorly misaligned prompts. Option B: StitchVM
  (diffusion backbone layers stitched to a reward-model tail). Open question: start with A or B.
  Reward shape question: strict/binary step thresholds vs smooth margins (partial-credit rewards can stall GRPO).
  Evaluation: GenAI-Bench; NoisyCLIP error categories Count, Color, Background, Subject swap.
- **Phase 2 - Trajectory sampling & scoring / conformal early pruning.** Flow-GRPO-style SDE sampling, groups of
  G=4-8, train rollouts with ~10 steps. Query the PRM at intervals (e.g. every 5 steps); trajectory reward = sum of
  step rewards. Calibrated early pruning (Conformal Risk Control threshold lambda_hat with 1-alpha guarantee;
  Conf-OT for drift across noise levels): stop trajectories below lambda_hat midway (e.g. step 20/50, up to ~50%
  rollout compute saved). Open question: where is the ideal t_prune.
- **Phase 3 - Critic-free GRPO update** with group-normalised advantages of the cumulative PRM reward; KL penalty to
  the reference model against reward hacking (tuning it is an open question).
- Open question from the slides: should early steering target global layout (t ~ 20-30 of 50) or details (t < 15)?

Why SD1.5: NoisyCLIP's linear RGB projection is for **4-channel** latents (SD1.x family).
The user's own summary also targets step reduction (40-50 -> 10-15 inference steps) and mentions legible text.
**Open decision:** the slides' task is prompt alignment (GenAI-Bench, compositional prompts), while the work so far
used an OCR/text-rendering task. Confirm with the user/supervisor which task the PRM + GRPO should target.

## Constraints and context

- The supervisor asked to use **Stable Diffusion 1.5** (`runwayml/stable-diffusion-v1-5`), not SD3. SD1.5 is an
  epsilon-prediction DDPM model, not flow matching: "t in [1.0, 0.5]" maps to the high-noise first half of the DDIM
  schedule (steps 0-4 of 10, DDIM timesteps ~901-501). GRPO log-probs come from a stochastic DDIM step
  (`flow_grpo/diffusers_patch/ddim_with_logprob.py`), DDPO-style.
- Task: OCR reward (PaddleOCR, CPU) + CLIPScore, `reward = 80*ocr + clipscore`.
- The user writes in Portuguese or English; answer in the language of their message.

## Cluster / environment (important)

- SLURM cluster. `conda activate` does NOT work inside batch jobs: every `.slurm` script must contain
  `export PATH=/user/home/b.nabais/.conda/envs/flow_grpo/bin:$PATH`. From an interactive shell use
  `/user/home/b.nabais/.conda/envs/flow_grpo/bin/python` (the shell can lose the env).
- Nodes: `novasearchdl` (A100-SXM4-40GB, slow when shared), `nscluster` (A100-PCIe-40GB), `nsdl2` (H200, fast).
  **The GPU type changes the samples for the same seed** (base eval 0.113 on A100 vs 0.131 on H200). Paired
  comparisons must run on the same GPU type; the A100 reference runs need `sbatch --exclude=nsdl2`.
- W&B runs are offline (`WANDB_MODE=offline`, folders in `wandb/`); the user syncs them to
  `bs-nabais-nova-fct/flow_grpo`. `scripts/wandb_offline_metrics.py <wandb/offline-run-...>` reads them locally.
- Claude's background watchers stop when the session ends; on resume, check `squeue -u $USER` and the logs first.

## Code map (SD1.5 work)

| File | Purpose |
|---|---|
| `scripts/train_sd15.py` | GRPO training for SD1.5. `config.sample.reward_mode` = `final` (reward of finished image, used now) or `intermediate` (x0 previews, old). `config.sample.eval_sampler` = `train` (fixed-seed eval with the training sampler) or `standard` (40-step DDIM). |
| `flow_grpo/diffusers_patch/sd15_pipeline_with_logprob.py` | Sampling with an SDE window (stochastic DDIM inside, deterministic outside); optional `sde_generator` for seeded evals. |
| `config/grpo.py` | SD1.5 configs: `general_ocr_sd15_*` (see "Experiment log"). |
| `dataset/ocr_simple/` | Short 1-2 word OCR prompts (train 1088 / test 128, disjoint words); `make_prompts.py` regenerates. |
| `scripts/eval_intermediate_latents.py` | Samples N images/prompt, decodes the x0 preview at **every** step, scores it, reports per-step reward, ranking agreement with the final image and timings. Supports `--lora`, `--num-steps`, `--eta`, `--sde-window-size/range`. Launch with `sbatch [--exclude=nsdl2] eval_latents.slurm <args>`. |
| `scripts/compare_evals.py` | Paired comparison of several `eval_intermediate_latents` runs (same prompts/seeds) vs the first one. |
| `scripts/early_selection.py` | Offline simulation of early pruning / best-of-N vs compute from saved per-step rewards. |
| `scripts/eval_paired.py` | Paired comparison of the fixed-seed evals inside one training log. |
| `scripts/reward_progress.py` | Per-epoch training OCR from a training log (its "total" column is wrong when the reward tensor print is truncated, e.g. 4-step windows). |
| `flow_grpo/noisyclip_scorer.py` | **Placeholder, not usable** - see "Noisy CLIP" below. |

Evaluation rule learned the hard way: in-training evals (512 images) produced two false "gains" (z=2.5, z=1.8).
Only claim an improvement after an **independent paired check**: `eval_intermediate_latents.py` with 1024 images
(128 prompts x 8), `--seed 2024`, same sampler as the reference, same GPU type, compared with `compare_evals.py`
against `logs/intermediate_eval/win4_base` (A100, 4-step window in steps 2-7) or a matching base run.

## Status (2026-11-09)

### GRPO training on SD1.5: has not improved OCR yet
All runs: base OCR ~0.12, exact match ~2%. Independent checks vs base (OCR difference, z):

| Setup | Result |
|---|---|
| Rewards from x0 previews only (original) | nothing learned (previews before step 5 do not predict the final image) |
| Final reward, 2-step window steps 0-4 (= the "latent spark" region), 32 prompts/epoch | +0.011 (z=1.0) |
| 2-step window steps 2-7 | flat |
| 4-step window steps 2-7 | -0.002 / +0.002 |
| + clip_range 1e-3 | -0.011 (z=-1.4) |
| + 1 optimizer step/epoch | -0.021 (z=-2.9) at ep30, -0.002 at ep90 |
| + 1 step/epoch, lr 3e-4 | +0.0005 (z=0.1) |

Diagnostics (W&B): reward signal exists (no zero-variance groups, reward std ~22); policy moves very little per
update (approx_kl ~1e-7); CLIPScore rises slightly (+0.002..+0.009), so training changes the model, not towards
legible text. Likely issue: SD1.5 almost never renders the target text, so groups contain no good sample.

### Per-step findings (solid, base and trained models alike, 10-step sampler)
- Preview OCR reaches ~95% of final at step 5 and 100% at step 6; within-group ranking agreement with the final
  reward: 0.45 (step 5), 0.60 (6), 0.70 (7), 0.79 (8). Steps 1-3 carry no usable signal.
- Early exit (output the step-k x0 preview): step 6 keeps final OCR (0.127 vs 0.125), CLIP 0.947 vs 0.973.
- Scoring one preview (VAE decode + OCR on CPU) costs ~1.4 UNet steps -> the VAE/OCR overhead is real.
- Early pruning (`scripts/early_selection.py`, i.e. the Phase 2 question "where is t_prune"): with the OCR reward
  and decoded previews, pruning only pays at step 6-7 of 10 and saves <= ~20% even with free scoring, because
  previews before step 5 carry no ranking signal. To redo with the NoisyCLIP PRM and a conformal threshold.

### NoisyCLIP: the scorer in the repo is a placeholder, not NoisyCLIP
`flow_grpo/noisyclip_scorer.py` creates a **randomly initialised, untrained** `LatentCLIPAdapter` (no weights are
loaded), mean-pools the latent over space and expects 16-channel SD3 latents. Its scores are random, and it is not
the published method (linear RGB projection of 4-channel latents + CLIP). The `noisyclip_sd3_fast` config trained
against it. Needed: Vasco Ramos' NoisyCLIP code/weights (ECCV 2026).

### Phase 1 prototype results (2026-11-09, `scripts/phase1_noisyclip.py`, `logs/phase1_noisyclip/sd15_50steps`)
Base SD1.5, 128 GenAI-Bench prompts x 4 seeds, 50-step DDIM (eta 0), CLIP ViT-L/14, Phi = affine 4->3 fitted by
least squares on 32 held-out prompts (R^2 0.82-0.86). Untrained (no NoisyCLIP fine-tuning).
- Cost per image: Phi(x0_hat)+CLIP 6 ms = **0.10 UNet steps**; VAE decode+CLIP 103 ms = 1.62 UNet steps (~16x).
- Detecting a random wrong prompt (own > mismatched): Phi(x0_hat) 93% at step 6/50, 97% at 10, 100% from 14.
- Predicting the final image's CLIP score (Spearman): Phi(x0_hat) 0.16 (20%), 0.49 (40%), 0.62 (60%) and
  **plateaus at ~0.65 even at the last step**; within-prompt ranking plateaus at ~0.4. VAE(x0_hat): 0.63 (40%),
  0.83 (60%), 0.94 (80%); within-prompt 0.33 / 0.64 / 0.77.
- Phi(z_t) (no x0 prediction) is useless until late (negative correlation before ~55%).
- Interpretation: the untrained projection has a ceiling - CLIP on a 64x64 linear RGB projection is not the same
  signal as CLIP on the decoded image. The paper's ~90% at mid-trajectory presumably comes from NoisyCLIP's
  training. Next: train a noise-aware scorer (e.g. distil CLIP(VAE(x0)) / final-image CLIP into a model on
  Phi(x0_hat_t) or on latents, with aligned + misaligned prompts), compare with a tiny decoder (TAESD) baseline,
  and add subtle negatives (count/color/subject swaps). Ideally get Vasco's NoisyCLIP code/weights first.

### Phase 1: own noise-aware scorer (2026-11-09)
- Data: `scripts/phase1_make_data.py` (+ `make_phase1_data.slurm`, 4 shards): 1440 GenAI-Bench training prompts x 2
  seeds, 50-step DDIM, Phi(x0_hat_t) + x0_hat_t at 25 timesteps + CLIP embedding of the final image ->
  `logs/phase1_data/sd15_50steps/` (3.2 GB). The 32 calibration + 128 eval prompts are excluded (verified).
- Model: `scripts/phase1_train_scorer.py` (+ `train_phase1_scorer.slurm`): CLIP ViT-L/14 vision tower + LoRA r16
  (3.4M trainable) on Phi(x0_hat_t) + timestep MLP, distilled to the final image's CLIP embedding; score =
  cos(prediction, CLIP text embedding). 3 epochs = 8 min on H200; converges after 1 epoch.
- distill_v1 (`logs/phase1_scorer/distill_v1`), held-out training prompts, cos(prediction, final image embedding):
  untrained 0.53 (t=961) / 0.72 (t=601) / 0.74 (t=401); trained **0.80 / 0.91 / 0.93**.
- Evaluation on the 128 eval prompts: `phase1_noisyclip.py --scorer <dir> --phi <the scorer's Phi>`.
- **Result (`logs/phase1_noisyclip/sd15_50steps_trained_v1`, 512 images, never-seen prompts)**, Spearman with the
  final CLIP score / within-prompt Spearman, at 20% / 40% / 60% / 80% of 50 steps:
  trained 0.44/0.33, 0.77/0.48, 0.85/0.53, 0.86/0.61 | untrained Phi 0.17/0.06, 0.50/0.26, 0.62/0.34, 0.65/0.39 |
  TAESD 0.25/0.19, 0.65/0.37, 0.80/0.55, 0.91/0.74 | VAE 0.22/0.13, 0.64/0.35, 0.82/0.60, 0.94/0.75.
  Trained scorer is the best predictor up to ~60% (it predicts the outcome; decoders measure the current preview)
  and plateaus at 0.86 / 0.62 afterwards. Wrong-prompt detection: 94% at step 2, 100% from step 8 (earliest of all).
  Cost per image: trained 0.16 UNet steps, Phi 0.11, TAESD 0.31, VAE 1.78. (Paper: ~0.9 correlation at mid-run.)
- Limits: target is CLIPScore (weak for composition); negatives are random prompts only; 64x64 projection input
  caps the ceiling. Ideas: latent input (x0_lat saved in the data), higher-res projection, subtle negatives.

### Phase 1 follow-ups (2026-11-09): skills, robustness, subtle negatives, v2 scorer
- **Per-skill breakdown** (`scripts/phase1_skill_breakdown.py <run>`): the v1 scorer behaves similarly on all GenAI-Bench
  skills; weakest within-prompt ranking on spatial/action relations (0.34-0.42 at 40-60%). Random wrong-prompt test
  is 100% everywhere, so it cannot separate skills.
- **Robustness** (`logs/phase1_noisyclip/robust_*`, v1 trained on 50-step eta=0 only), Spearman / within-prompt at 40%:
  15 steps 0.72/0.46, 10 steps 0.64/0.38, 50 steps eta=1 0.80/0.40, 10 steps eta=1 (GRPO-like) 0.60/0.34; still the
  best predictor in the first half everywhere (decoders 0.45-0.66 / 0.25-0.37); ~0.05-0.1 below the 50-step result.
  Retrain on 10-step stochastic data before Phase 2/3.
- **Subtle negatives**: `scripts/phase1_make_negatives.py` (Qwen2.5-7B-Instruct, `make_negatives.slurm`) ->
  `dataset/genai_bench/negatives.json`: minimal edits per prompt for count/color/background/subject/relation (1600
  prompts; non-null 383/1004/1353/1592/1216). Some edits are near-synonyms (baker->chef): noisy labels.
  `phase1_noisyclip.py --negatives` scores them; `scripts/phase1_negatives_report.py <run>` reports per category.
- **Result (A100, paired, `logs/phase1_noisyclip/neg_eval_{v1,v2_neg,v2_lat,v2_lat_neg}`)**: CLIP on the FINAL image
  prefers the true prompt over the subtle negative only 81% of the time (count 0.76, color 0.79, background 0.88,
  subject 0.91, relation 0.66) - this is the ceiling of a CLIP-distilled PRM. The trained scorer reaches 0.77 at 20%
  and ~0.80 from 40% (agreement with CLIP's final verdict 0.85 at 20%, 0.91 at 40%); TAESD/VAE are at 0.71 at 20%.
  So the PRM already makes CLIP's final judgement early; to judge composition better the *teacher* must change
  (e.g. VQAScore / a VLM judge), not the student.
- **v2 scorers** (`scripts/phase1_train_scorer.py --latent-input` adds a conv decoder on x0_hat latents ->
  RGB residual at 256x256; `--negatives` adds a margin-distillation loss):
  `v2_lat` is the best: Spearman with final 0.82 (40%), 0.88 (60%), 0.90 (end) vs v1 0.78/0.85/0.85; within-prompt
  +0.02..+0.07 (significant at 50-60%). The negative-margin loss gives no gain (already at the CLIP ceiling) and
  slightly hurts ranking -> drop it. Default scorer from now on: `logs/phase1_scorer/v2_lat`.

### Scorer for the GRPO setting + Phase 2 pruning (2026-11-10)
- `logs/phase1_scorer/v2_lat_10steps_eta1`: v2_lat retrained on 10-step eta=1 trajectories (1440 prompts x 4 seeds,
  every step; `logs/phase1_data/sd15_10steps_eta1`). Eval at 10 steps eta=1 (H200, paired with v1 and the 50-step
  v2_lat), Spearman / within-prompt at step 4, 5, 6 of 10: **0.68/0.42, 0.80/0.50, 0.86/0.61** vs v1 0.60/0.34,
  0.70/0.41, 0.77/0.49 (within-prompt +0.07..+0.12, significant from step 5); end 0.90/0.68 vs TAESD 0.93/0.74.
  Use this scorer for Phase 2/3.
- Phase 2 offline pruning: `scripts/phase2_conformal_pruning.py <run>` (Conformal Risk Control on the gap to the
  group leader; risk = best-final trajectory pruned, or fraction of top half pruned). 10-step eta=1, G=4, new scorer:
  alpha=0.1 -> best at step 5: prunes 33%, saves **15%** of UNet compute (TAESD 7%, VAE negative: its cost exceeds the
  savings); alpha=0.2 -> step 4, prunes 41%, saves **23%** (TAESD 17%). Test risk ~= alpha (guarantee holds within
  noise). Regret is tiny (~0.001-0.003 CLIPScore), i.e. the pruned "best" is nearly tied with the kept one, so the
  "best" risk is strict. 50-step deterministic: 20-24% at step 20-25. Savings capped by G=4 and one pruning point.

### Phase 2 with G=8 and multi-point pruning (2026-11-10)
Data: `logs/phase1_noisyclip/eval10_G8_v2_lat_10steps_eta1` (128 held-out GenAI-Bench prompts x 8, 10 steps eta=1,
scorer v2_lat_10steps_eta1). `scripts/phase2_conformal_pruning.py --multi 3 4 5 6 3,5 4,6 ... --group-size G
--risk best|top|regret --delta d` (one lambda in units of the per-step within-group std, calibrated by CRC on half
the prompts, tested on the other half, 40 random splits). Within-group std of the final CLIP score: 0.021.
UNet compute saved at alpha=0.1 (test risk ~0.09-0.10 in all rows, guarantee holds):
| risk | G=4 | G=8 |
| best image of the group never lost (p<=0.1) | 14% (prune at step 6) | 17% (step 6, 47% pruned) |
| best kept within 0.005 of the best | - | **22%** (step 6, 58% pruned) |
| best kept within 0.01 of the best | - | 24% (step 6, 63% pruned) |
alpha=0.2, best: 23% (G=4) / 24% (G=8, steps 4+6). TAESD saves 10-18%, VAE <= 5% (often negative: decode cost).
Multi-point pruning (3+5, 4+6, 2+4+6) does not beat a single checkpoint at step 6 (= t_prune ~ 60% of the
trajectory) with a shared lambda. The ceiling: a trajectory pruned at step 6 of 10 saves only 40% of its steps, and
before step 4-5 the PRM is not reliable enough to prune hard. Larger G helps a little.

### What already maps onto the plan
- `eval_intermediate_latents.py` already computes x0_hat at every step and scores it -> the harness for Phase 1
  (swap the VAE decode for the linear projection, compare with decoded CLIP and with the final score per step).
- `early_selection.py` -> Phase 2 pruning analysis; the per-step ranking-agreement metrics answer "t_prune".
- `train_sd15.py` with `reward_mode="intermediate"` -> Phase 3 skeleton (currently uses deltas of step rewards; the
  plan uses the sum of step rewards).

### Step-reduction baseline (2026-11-10, OCR task, base SD1.5, DDIM eta=0, 1024 images, seed 2024, H200)
`logs/intermediate_eval/h200_eta0_{4,6,8,10,20,50}steps/`. Final OCR / CLIP / exact match:
4 steps 0.047/0.819/0.00 | 6 0.065/0.923/0.01 | 8 0.097/0.953/0.01 | 10 0.118/0.967/0.02 | 20 0.160/0.977/0.04 |
**50 0.193/0.975/0.04**. (A100 cross-check: 6 steps 0.067, 8 steps 0.095.) 10 steps keep 61% of the 50-step OCR,
20 steps 83%; CLIP saturates by ~10-20 steps. Target for the thesis: a 10-15-step model at the 50-step level.

### Phase 3: GRPO for prompt alignment + text rendering together (started 2026-11-10)
User decision (2026-11-10): focus on prompt alignment but keep great text rendering -> train both.
- Prompts `dataset/mixed_align_ocr/train.txt`: 1440 GenAI-Bench training prompts + 1088 OCR prompts (held-out
  eval prompts of both excluded). Reward (`flow_grpo/mixed_reward.py`, `reward_mode="mixed"` in train_sd15.py):
  100*alignment + 80*OCR (OCR only if the prompt has a quoted text). Per-prompt advantage normalisation.
- `mixed_sd15_prm_1gpu` (our method): alignment = noise-aware PRM v2_lat_10steps_eta1 on x0_hat at steps 4,6,8,10,
  averaged (no VAE); job 626354, `logs/phase3/mixed_prm`. `mixed_sd15_orm_1gpu` (baseline): alignment = CLIP-L on
  the final image; job 626360, `logs/phase3/mixed_orm`. Both: 10 steps, stochastic DDIM at all steps, all 9
  transitions trained, lr 3e-4, clip 1e-3, KL beta 0.04, 32 prompts x 8 images/epoch, ckpt every 10 epochs
  (checkpoint-N, N = 2 x epoch). approx_kl ~2e-4 (vs 1e-7 in the old OCR runs): the policy now moves.
  Launchers `train_phase3_{orm,prm}.slurm` (need distinct `--main_process_port`, two runs share nsdl2).
- Eval: `scripts/phase3_eval.py` (10-step deterministic DDIM, fixed per-image seeds -> paired): 128 held-out
  GenAI-Bench prompts x 4 (CLIP-L, PickScore = independent judge, subtle-negative accuracy) + 128 OCR test prompts
  x 4 (OCR, exact, CLIP). `scripts/phase3_compare.py base_10steps <runs>` -> diffs with 95% bootstrap CIs.
  `scripts/phase3_autoeval.sh` (nohup) evaluates every checkpoint automatically -> `logs/phase3_eval/<run>_checkpoint-N`.
- Base references (H200): 10 steps CLIP 0.2812, PickScore 0.812, neg-acc 0.819, OCR 0.094, exact 1.6% |
  20 steps 0.2829 / 0.822 / 0.827 / 0.142 / 2.3% | 50 steps 0.2825 / 0.826 / 0.824 / 0.138 / 2.3%.
  (OCR differs from the earlier step-reduction table: different seeds, 4 vs 8 images, 512 images.)

### Phase 3 confirmation (2026-11-10, `logs/phase3_confirm/`, seed 2024, 8 imgs/prompt, 1024+1024 images, H200)
**The PRM run's OCR gain did NOT reproduce**: prm checkpoint-120 OCR 0.107 vs base 0.103 (+0.005, CI -0.008..+0.017),
exact +0.6pp n.s.; in-training eval had said +0.026* -> selection effect (best of 12 checkpoints, 512 images), the
third false gain. orm checkpoint-120: OCR +0.004 n.s., align PickScore -0.002*. Alignment CLIP/neg-acc unchanged for both.
Base 20 / 50 steps vs 10: OCR +0.049* / +0.079*, PickScore +0.008* / +0.011*. Phase 3 has no confirmed gain yet.

### Phase 3 curricula (supervisor's suggestion, submitted 2026-11-10 ~14:00)
Two schedulers on top of `mixed_sd15_prm_1gpu` (one change each + both):
- `mixed_sd15_prm_wsched_1gpu` (job 626630): reward-weight curriculum `sample.weight_schedule=(60, 0.5)`: over 60
  epochs OCR weight 0 -> 80, alignment 100 -> 50 (alignment first, text later; never 0). Only the ratio matters
  (per-prompt normalisation), only on text prompts. `MixedReward.set_epoch`.
- `mixed_sd15_prm_ssched_1gpu` (626631): denoising-steps curriculum `sample.steps_schedule=[(0,4),(20,6),(40,8),
  (60,10)]` (supervisor said 2 -> 10; 2 steps give SD1.5 images with no usable content, base 4 steps OCR 0.047).
  PRM at `prm_fracs=(0.4,0.6,0.8,1.0)` of the current trajectory. Trainer uses manual gradient accumulation when
  steps_schedule is set (Accelerator gas=1, step when j is the last transition and (i+1) % gas == 0, loss scaled by
  1/(gas x transitions)) -> still 2 optimizer steps/epoch. `steps_at(config, epoch)` in train_sd15.py.
- `mixed_sd15_prm_both_1gpu` (626632): both.
- Scorer check at 4/6 steps (only trained on the 10-step grid): jobs 626633/626634 -> `logs/phase1_noisyclip/eval{4,6}_v2_lat_10steps_eta1`.
  Result: works. 4 steps: Spearman/within at step 2 0.67/0.35, step 3 0.83/0.63; 6 steps: step 3 0.71/0.44, 4 0.81/0.58,
  5 0.86/0.64 (similar to 10 steps at the same fraction).
- First epochs: 4-step runs have approx_kl ~0.02-0.04 after one update (10-step: ~2e-4; pre-update KL is 0, so log-probs
  match), clipfrac ~0.9 -> clip 1e-3 cuts most gradient in the 4-step phase. Not a bug.
- Checkpoints auto-evaluated at 10 steps by `scripts/phase3_autoeval_runs.sh` (nohup, log `logs/phase3_autoeval2.log`).
- Cluster full on 2026-11-10 (147 pending jobs): curriculum jobs got `nice=50` so the confirmation evals
  (626597-626601, `logs/phase3_confirm/`, seed 2024, 8 imgs/prompt, base 10/20/50 + prm/orm checkpoint-120) go first.
- Meeting report (artifact): https://claude.ai/artifact/VQq6aaQJ2tiJ3p6FbkyLCE, built by
  `scripts/report/build_report.py` + `template.html` (reads the result files, writes latent_spark_report.html next
  to itself; rebuild and republish to the same URL).

## Running experiments
- Step-reduction baseline (OCR task, base SD1.5, eta=0, 1024 images, seed 2024, A100): 4 steps done (job 626258);
  6/8/10/20/50 resubmitted 2026-11-10 (jobs 626338-626342, A100 too busy: only 6 (OCR 0.067, CLIP 0.923; 4 steps
  0.053/0.819) and 8 kept; all of 4/6/8/10/20/50 rerun on H200 in parallel, jobs 626346-626351 ->
  `logs/intermediate_eval/h200_eta0_{N}steps/`; `--score-every` 1/1/1/2/5 because OCR on CPU costs ~15 min
  per scored step) -> `logs/intermediate_eval/infer_eta0_{N}steps/`. The first 6-20 step jobs were cancelled on
  2026-11-09 when switching to Phase 1.
- 2026-11-10: v2_lat retrained for the GRPO setting: data 10-step eta=1, 4 seeds/prompt, every step
  (`logs/phase1_data/sd15_10steps_eta1`, jobs 626329-626332) -> training 626343 (`logs/phase1_scorer/v2_lat_10steps_eta1`)
  -> eval 626344 at 10 steps eta=1 on nsdl2 (`logs/phase1_noisyclip/eval10_v2_lat_10steps_eta1`), paired with the
  50-step-trained v2_lat under the same setting (626345, `eval10_v2_lat`).

## Next steps (following the 3 phases)
1. Decide the task (prompt alignment on GenAI-Bench as in the slides, or OCR/text) and get NoisyCLIP code/weights.
2. Phase 1: implement the NoisyCLIP Option-A scorer for SD1.5 (x0_hat -> linear 4->3 RGB projection -> CLIP) in
   the per-step harness; measure per timestep its correlation with the final score and its cost vs VAE decode.
3. Phase 2: per-step ranking agreement of the PRM -> t_prune; conformal threshold for early pruning.
4. Phase 3: GRPO with the cumulative PRM reward, KL penalty, window in the early "latent spark" steps.
5. Step-reduction baseline (running): quality vs number of steps for base SD1.5, plus a 40/50-step reference.

### 2026-11-10 15:07: all Phase 3 training jobs cancelled by root (admin)
626354 (prm, last ckpt checkpoint-200 = epoch 100), 626360 (orm, checkpoint-200), 626630 wsched (epoch 4, no ckpt),
626631 ssched / 626632 both (epoch 11, checkpoint-20), plus pending evals; the nohup auto-eval watchers died too.
Likely reason: 5 trainings + evals at once on the shared nsdl2. train_sd15.py has no resume (config.resume_from unused).
Ask the user before resubmitting.
Resubmitted 2026-11-10 (user: in order, prm/orm stay as they are): serial chain on nsdl2 via afterany dependencies,
wsched 626748 -> ssched 626749 -> both 626750, each 100 epochs (`--config.num_epochs=100` in the slurm files;
train_sd15.py now breaks at epoch >= num_epochs after saving the checkpoint). Old partial outputs moved to
`logs/phase3/mixed_prm_{ssched,both}_cancelled1110` and `logs/phase3_eval/cancelled1110/`.
Auto-eval: nohup `phase3_autoeval_runs.sh`, log `logs/phase3_autoeval3.log`.

### Curricula results (2026-11-11, in-training eval, 512 imgs/set, vs base 10 steps)
- wsched (626748, done 01:23, 4.5h): OCR +0.005..+0.022 at every checkpoint, significant at ep 30/40/90 (0.115 vs 0.094);
  exact n.s.; CLIP/neg-acc flat; PickScore -0.001..-0.002 mid-run (sig ep 20/30/60), ~0 at end. Most consistent run so far.
- ssched (626749, done 09:07, 3.5h): approx_kl/clipfrac per 20-epoch block 7.7e-3/0.64 (4 steps), 4.8e-4/0.26 (6),
  1.8e-4/0.17 (8), ~8e-5/0.13 (10). Its evals queued (626861-626874, temporarily held so confirmations go first;
  `logs/release_held.log` releases them).
- both (626750): running/pending. prm/orm checkpoint-200 evals: 626878/626879.
- New-seed confirmations: 626875 wsched ep30, 626876 wsched ep100 (final, pre-chosen), 626877 ssched ep100 ->
  `logs/phase3_confirm/{wsched,ssched}_checkpoint-N` (LoRAs copied to `logs/phase3/confirm_ckpts/`).
- Report: builder now has CONF_LAB, curricula chart, run-details table; free text in `scripts/report/curr_note.html`
  (confirmation interpretation) and `curr_glance.html` (glance bullet) if present.
- 2026-11-11: user wants **one job at a time** on the cluster. All pending jobs chained with afterany:
  626875 -> 626876 -> 626877 (confirmations) -> 626878/626879 (prm/orm ckpt-200 evals) -> 626861..626874 (ssched evals)
  -> 626750 (both training) -> 626880 (`eval_phase3_all.slurm mixed_prm_both`: evals all its checkpoints in one job).
  The auto-eval and release watchers were stopped. New jobs: chain after the last one, never in parallel.
- 2026-11-11 meeting material: guião (Claude Doc) https://claude.ai/artifact/1Cobu6AZ8goAMKqeFrAQkm (Portuguese, points 1-8,
  questions, glossary); visual companion https://claude.ai/artifact/UZ9m2MP3zPJr4SN9XJNdND built by
  `scripts/report/build_visual_guide.py` -> `latent_spark_visual.html` (reads results; rebuild + republish same path
  when confirmations / ssched evals arrive).

### 2026-11-11 afternoon: supervisor feedback + confirmations
- Confirmations (new seeds, 1024 imgs/set vs base 10): **wsched epoch 100 (pre-chosen final) OCR +0.0135
  [+0.0007, +0.0269]* (0.116 vs 0.103), exact +0.9pp (borderline), OCR-set PickScore +0.002*, alignment unchanged**
  = first OCR gain that passes the independent check (still far below base 20 steps +0.049). wsched ep30 +0.006 n.s.;
  ssched ep100 +0.000 n.s.; ssched in-training evals flat (one sig. checkpoint, ep70); prm/orm ep100 OCR +0.011/+0.017 n.s.
- Supervisor: liked the work; (1) investigate WHY OCR does not improve: hyperparameters, signal strength, in-group
  variance, gradients; (2) do NOT train our own CLIP scorer, use NoisyCLIP from HF; (3) curricula must not start OCR at 0.
- NoisyCLIP = HF `asiimo/noisyclip` (code github.com/novasearch/noisyclip): CLIP ViT-L/14, only the vision tower
  fine-tuned (text tower verified identical to openai/clip-vit-large-patch14; repo has no tokenizer -> fallback in
  `Clip`), trained on SDXL **noisy latents z_t** (UNet input) rendered with a fixed linear map (`nc_project` in
  phase1_noisyclip.py). Made for SDXL latents: must check it on SD1.5.
- Added: `phase1_noisyclip.py --noisyclip asiimo/noisyclip` (methods nc_zt / nc_x0 / nc_x0phi); `MixedReward(align=
  "noisyclip", noisyclip_input=zt|x0|x0phi)`; pipeline x0_callback now gets (i, t, x0, z_t); weight_schedule
  (ramp, align_end_frac, ocr_start_frac); configs `mixed_sd15_nc_1gpu`, `mixed_sd15_nc_wsched_1gpu` (OCR 40->80),
  `mixed_sd15_nc_both_1gpu`; launchers train_phase3_nc{,_wsched,_both}.slurm (100 epochs). Not submitted yet.
- `scripts/diag_ocr_signal.py` (+ diag_ocr.slurm): per-group OCR signal + per-sample policy-gradient analysis
  (SNR at batch 128, OCR vs alignment part, per-transition share, KL-gradient ratio). W&B already shows: beta*KL ~20%
  of |policy loss|; training OCR only 0.230 -> 0.243 over 100 epochs (PRM run); max_grad_norm = 1.0.
- Queue (serial): 626750 both (running) -> 626880 eval_all both -> 626995 diag (base, prm ep100, wsched ep100)
  -> 626996 phase1 NoisyCLIP eval (`logs/phase1_noisyclip/eval10_noisyclip`, paired with eval10_v2_lat_10steps_eta1).
- 2026-11-11 ~17:10: user chose to cancel 626750 (both, epoch 51; ckpts to ep50 kept) and 626880; diag 626995 now first.
- **OCR diagnostic results (2026-11-11, job 626995, A100, `logs/diag_ocr_signal/*.json`, 32 text + 16 align prompts x 8,
  training sampler; base / prm ep100 / wsched ep100):**
  - Signal is NOT missing: OCR within-group std 0.24-0.28, range 0.63-0.71; 72-78% of text groups have an image with
    OCR >= 0.5, 41-50% contain an exact match. OCR = 80-88% of within-group reward variance on text prompts; advantage
    correlates 0.97-0.99 with OCR. (The old "no good sample in the group" explanation was for the long prompts.)
  - **57% of the policy-gradient norm comes from the LAST transition (t=101->1)**, cos 0.96 with the total; every other
    transition 1.5-8% (cos ~0.1). Tiny sigma at the last DDIM step -> huge log-prob gradients; the update mostly
    tweaks the final step, not steps 3-5 where text forms.
  - Gradient SNR: the first estimate (0.58 at 128, "1.0 at 384") was WRONG - it is the pure-noise value (||mean|| is
    inflated by the noise; raw SNR at the measured N=384 was 0.99-1.02 for every model and every transition).
    Bias-corrected signal/noise: base ~0.09-0.11 at 128 samples (0.3 at 1024), trained models ~0 (not detectable).
    The policy-gradient signal per update is tiny; the per-transition "norm share" is mostly noise norm.
  - KL-penalty gradient = 0.72x (prm) / 1.16x (wsched) the policy gradient at the end of training (beta 0.04), nearly
    orthogonal to it -> KL holds the model back as much as the reward pushes it (also likely last-step dominated).
  - OCR and alignment gradient parts are orthogonal (cos 0.01-0.05) and of similar norm on the mixed batch.
  - Grad-norm clipping (1.0) never triggers (expected grad norm ~0.002-0.004).

### NoisyCLIP (HF) on SD1.5 + fixed GRPO run (2026-11-11 evening)
- **NoisyCLIP check** (626996, `logs/phase1_noisyclip/eval10_noisyclip`, 10 steps eta=1, 512 imgs). Spearman with the final
  CLIP score / within-prompt / wrong-prompt detection, at step 5 and step 10 of 10:
  nc_zt (paper's input: z_t + SDXL colour map) -0.09/-0.01/0.78 and 0.16/0.23/0.98 (useless);
  nc_x0 (x0_hat + SDXL map) 0.09/0.15/0.98 and 0.17/0.24/0.98; nc_x0phi (x0_hat + our SD1.5 Phi) 0.44/0.29/1.00 and
  0.55/0.36/1.00 (~= plain CLIP on Phi(x0), proj_x0 0.48/0.29); our trained v2_lat_10steps_eta1 0.81/0.53 and 0.91/0.69;
  TAESD 0.63/0.35. Off-the-shelf NoisyCLIP does not transfer to SD1.5 latents (made for SDXL). Cost 0.10 UNet steps.
- `train.skip_last = k` in train_sd15.py: leaves the last k transitions out of the update (forces manual accumulation).
- Config `mixed_sd15_nc_wsched_fix_1gpu` (NoisyCLIP x0phi, OCR 40->80, skip_last=1, beta 0.01, 1 optimizer step of
  256 samples/epoch), launcher `train_phase3_nc_wsched_fix.slurm`.
- diag_ocr_signal.py now also reports per-transition SNR, SNR without the last k transitions (`--skip-last`), per-
  transition KL share and the KL/policy ratio without the last transition.
- Serial chain: 627017 diag (627014 crashed on a bug in the new code, fixed) (base + wsched ep100, config nc_wsched_fix, `logs/diag_ocr_signal/v2_fix/`) -> 627015
  training (`logs/phase3/mixed_nc_wsched_fix`) -> 627016 eval of all checkpoints on nsdl2 (`logs/phase3_eval/mixed_nc_wsched_fix_checkpoint-N`).
  The config is read when 627015 starts: adjust it after the diag if needed.
- Diag v2 (627017, `logs/diag_ocr_signal/v2_fix/`, config nc_wsched_fix, beta 0.01): wsched ep100 raw SNR 0.98 (= noise);
  **92% of the KL gradient comes from the last transition**; KL/policy gradient norm 0.30 with all transitions, 0.05
  without the last one. Training 627015 started 2026-11-11 on nscluster (A100); eval 627016 follows on nsdl2.
- 627015 (nc_wsched_fix) COMPLETED 2026-11-12 (8h15 on A100 nscluster), checkpoint-N = epoch N (1 optimizer step/epoch);
  W&B wandb/offline-run-20261111_210253-fu9u7p94. approx_kl/clipfrac log as 0 (all PPO ratios computed before the single
  update - expected). Training OCR per 20 epochs 0.230/0.252/0.259/0.274/0.257 vs wsched (mwqrmaxd) 0.237/0.251/0.260/0.254/0.243
  (noisy, ~128 text images per epoch). Eval 627016 pending on nsdl2.

### English report (2026-11-12)
- `scripts/report/build_report_en.py` -> `latent_spark_report_en.html`, published to the meeting-report URL
  https://claude.ai/artifact/VQq6aaQJ2tiJ3p6FbkyLCE (version 6; replaces the old build_report.py content). In-house
  trained scorer left out on the user's request (runs that used it are labelled "earlier in-house scorer"). Points 1-10:
  SD1.5, OCR GRPO, per-step, fewer steps, NoisyCLIP on SD1.5 (views in `scripts/report/assets/`, rendered from
  `logs/phase1_data/sd15_10steps_eta1` x0_lat), pruning, Phase 3, curricula + confirmations, diagnostic, new run.
  Section 10 picks up `logs/phase3_eval/mixed_nc_wsched_fix_checkpoint-N` automatically: rebuild + republish when 627016 is done.
- Phase 2 pruning redone with NoisyCLIP (eval10_noisyclip, G=4, risk best): alpha 0.1 -> 7% saved (random pruning
  at step 1 gives ~9% at the same risk), TAESD 9% (step 6), VAE -7%; alpha 0.2 -> NC 14-16%, TAESD 17%, random ~18%.
- 627016 eval done (2026-11-12, in-training eval, 512 imgs/set vs base 10): nc_wsched_fix OCR +0.000..+0.026 (sig at ep
  10/30/70/80; ep100 +0.017 n.s.), exact n.s.; unlike every earlier run, alignment also rises steadily: align CLIP
  +0.003..+0.004* from ep40, align PickScore +0.003* at ep100, OCR-set CLIP +0.011* / PickScore +0.005* at ep100
  (CLIP at the base-20-step level, PickScore still well below it). neg-acc n.s. Needs a new-seed confirmation of ep100.
- 2026-11-13: new-seed confirmation of nc_wsched_fix ep100 submitted: job 627172 (pinned to nsdl2, LoRA copied to
  `logs/phase3/confirm_ckpts/nc_wsched_fix_checkpoint-100`) -> `logs/phase3_confirm/nc_wsched_fix_checkpoint-100`.
  Report v7 published (section 10 now has the epoch-100 table; the builder picks up the confirmation automatically:
  rebuild + republish when 627172 is done).
- 2026-11-13 "problem 2" (gradient mostly noise): hypothesis = within a group every image has its own z_T and z_T has no
  log-prob, so reward variance caused by z_T is pure noise in A*grad(logp). `diag_ocr_signal.py` now has --same-latent,
  --sde-window-size/--sde-window-start, --tag and an unbiased cross-fold signal estimate (trained_rho2_per_image,
  trained_snr_unbiased_at_256 with jackknife CI; verified unbiased on synthetic data). Job 627173 (`diag_variants.slurm`,
  after 627172) -> `logs/diag_ocr_signal/v3_variants/`: v0 indep z_T full (current), v1 shared z_T full, v2 indep window
  1-4, v3 shared window 1-4, v4 shared window 2-3 (64 text prompts x 8, base, paired seeds).
  train_sd15.py fixes: same_latent now seeds by (epoch, prompt) only (old `+ i` gave a group's copies different z_T,
  since the sampler shuffles copies across batches); manual accumulation respects a fixed SDE window.
  Next: pick the variant with the highest per-image signal (and enough OCR spread), then one training run.
- **Confirmation nc_wsched_fix ep100 (627172, new seeds, 1024+1024, H200):** alignment CLIP +0.0014*, PickScore +0.0015*;
  text-prompt CLIP +0.0069* (above base 20 steps), PickScore +0.0035*; **OCR -0.007 n.s.**, exact +1.0pp n.s.
  -> alignment/quality gain confirmed (first confirmed alignment gain), OCR not. Report v8 published.
- 627173 (diag variants) FAILED after 2.4 h: NameError (Nall used before definition in the new estimator) - nothing
  saved. Fixed; raw fold statistics now saved to `<name>_raw.npz` before the analysis. Smoke test 627187
  (`diag_smoke.slurm`, 4 prompts x 4) -> full rerun 627188 (afterok).
- **Variant diag result (627188, `logs/diag_ocr_signal/v3_variants/`, base, 64 text prompts x 8, unbiased cross-fold
  estimate):** per-image signal/noise rho2 (SNR at 256 imgs, 95% CI): v0 indep full 1.3e-4 (+0.19 [-0.12,+0.29]),
  v1 shared z_T full -1e-4, v2 indep win1-4 1.0e-4, v3 shared win1-4 0.0, v4 shared win2-3 1.6e-4 (+0.20). All within
  noise of each other and of 0 -> **z_T hypothesis refuted**: sharing z_T barely lowers within-group OCR spread (std 0.270
  -> 0.253; win2-3 still 0.175), i.e. the SDE noise, not z_T, already decides the text. Nothing raises per-image signal.
  Implication: ~6-10k images per update for SNR 1; a whole 100-epoch run (25.6k imgs) carries total SNR ~ sqrt(25.6k x
  1.3e-4) ~ 1.8 -> explains the small, non-reproducible OCR gains. Epoch time (A100, 256 imgs): sampling+reward 1:39,
  training (8 transitions) 3:21 -> a 2-step window cuts training cost per image ~4x at the same per-image signal.
  Fix = more images (volume) or a lower-variance estimator (differentiable OCR proxy / direct reward backprop). Asked user.

### Option 2: differentiable text reward (user decision 2026-11-13; fallback = option 1, long RL run)
- Report v9 published (section 9 point 5 = variant test, glance + next steps updated).
- `scripts/opt2_proxy_check.py` (+ `opt2_proxy.slurm`, job 627243 -> `logs/opt2/proxy_check/`): 128 OCR test prompts x 4,
  base 10-step DDIM; agreement of TrOCR-str (base/large, PaddleOCR box crop or full image) NLL and CLIP text-match with
  the PaddleOCR score. TrOCR-str needs decoder_start_token_id=2, labels without <s>, and reads UPPERCASE without spaces.
- `scripts/train_sd15_draft.py` (+ `train_opt2_draft.slurm`): AlignProp/DRaFT-style - deterministic 10-step DDIM, grad
  from step k0 (default 2) with per-call checkpointing, VAE decode, crop from PaddleOCR box, loss = TrOCR NLL; PaddleOCR
  score logged as hacking check; LoRA r32; ckpts `<save_dir>/checkpoints/checkpoint-N/lora` (phase3_eval-compatible).
  Smoke run 627244 (after 627243) -> `logs/opt2/smoke`.
- Proxy check results (627243 + v2 627245, `logs/opt2/proxy_check{,_v2}/`, 512 base imgs, only 8 exact matches):
  no differentiable proxy agrees with PaddleOCR within prompt (best: CLIP 0.17, TrOCR-base per-line 0.12, others ~0 or
  negative); overall Spearman <= 0.12. Among images with PaddleOCR > 0: 0.37-0.48 (TrOCR-large line + CLIP best).
  Exact-match AUROC high (TrOCR-large best line 0.94, + CLIP 0.96, CLIP 0.92) but from only 8 positives. Proxies cannot
  tell PaddleOCR's partial credit (0 < score < 0.5, often garbage text with a few matching chars) from 0 (AUROC ~0.5).
  Smoke 627244 OK: 3 s/update (4 imgs), 15 GB, H200; grad norm 175-510 before clipping. Asked user: cheap pilot vs option 1.
- User: run the pilot (2026-11-13). train_sd15_draft.py: --crop line (best single PaddleOCR line + whole image, min
  NLL) and --clip-text-weight (CLIP 'a photo with the text "X"', 50 = TrOCR scale). Pilot 627248 (nsdl2):
  `logs/opt2/pilot_v1`, TrOCR-large, line crop, clip-text 50, k0 2, 8 imgs/update, lr 1e-4, 600 steps, ckpt every 100.
  Eval 627249 (afterok, `eval_opt2.slurm pilot_v1 600`): all ckpts -> `logs/phase3_eval/opt2_pilot_v1_checkpoint-N`
  (vs phase3_eval/base_10steps), final ckpt-600 pre-chosen confirmation -> `logs/phase3_confirm/opt2_pilot_v1_checkpoint-600`.
  Stop rule: TrOCR loss falls but PaddleOCR (logged on train images) does not rise -> hacking -> option 1.
- **Pilot 627248 result (done 17:49, 1h04, 600 steps):** failed by the stop rule. Training-image means, first 100 vs last
  100 steps: PaddleOCR 0.245 -> 0.125 (z -6.5), exact 18% -> 11%, text detected 83-89% -> 43-66% mid-run, TrOCR NLL did
  not fall (3.73 -> 4.01), only CLIP text-match rose (0.221 -> 0.257, z 12) -> the model optimises the CLIP term and
  loses text; grad norms 500-3300 (clipped to 1). Eval 627249 (all ckpts + ckpt-600 confirmation) still running.
  Next per plan: option 1 (long RL run) - ask the user first.

### Option 1: long cheap-per-image RL run (user go-ahead 2026-11-13, after option 2 failed)
- Config `ocr_long_win23_1gpu` (launcher `train_ocr_long.slurm`, 60 h limit): text prompts only (dataset/ocr_simple),
  NoisyCLIP x0phi alignment + OCR (100/80, no schedule), shared z_T per group, 2-step SDE window fixed at steps 2-3
  (diag v4), 64 prompts x 8 = 512 imgs/epoch, 1 optimizer step/epoch, lr 3e-4, beta 0.01, 500 epochs = 256k images
  (10x nc_wsched_fix), ckpt every 50 epochs -> `logs/phase3/ocr_long_win23` (checkpoint-N = epoch N). No resume support.
- Chain on nsdl2: 627249 (opt2 eval) -> 627262 training (afterany) -> 627263 `eval_run_confirm.slurm ocr_long_win23 500`
  (all ckpts -> `logs/phase3_eval/ocr_long_win23_checkpoint-N`; pre-chosen final ckpt-500 new seeds ->
  `logs/phase3_confirm/ocr_long_win23_checkpoint-500`).
- Pilot eval 627249 (held-out, vs base 10): OCR falls with training (ckpt-100 -0.021 n.s., ckpt-600 -0.045*); OCR-set
  PickScore -0.03..-0.07* from ckpt-200; neg-acc ckpt-600 -0.17*. Confirmation ckpt-600 (new seeds): OCR -0.052*, OCR-set
  CLIP -0.057*, PickScore -0.073*; alignment set CLIP 0.283 -> 0.175*, PickScore 0.816 -> 0.690*, neg-acc -0.18* (image quality collapsed). Option 2 (as piloted) clearly harmful.
- 627262 started 18:09 on nsdl2 (W&B wandb/offline-run-20261113_180950-gdqg0z0m). Epoch (512 imgs) = 1:50 sampling +
  0:51 training ~ 2:45 -> 500 epochs ~23 h (ends ~2026-11-14 17:00), evals ~2 h after. Sampling+OCR is now 2/3 of the cost.
- 2026-11-14 09:30: 627262 at epoch 340/500, no errors. Training-image OCR per 50 epochs (512 imgs/epoch, train words):
  0.253, 0.300, 0.302, 0.314, 0.350, 0.356, 0.364 (Spearman with epoch 0.67) - first run with a clear, steady training
  OCR rise (earlier runs: ~0.23 -> 0.25-0.27). Held-out words not checked yet (evals after training, one job at a time).

### Article (2026-11-14)
- NeurIPS 2026 template from the supervisor (Overleaf zip) unpacked in `paper/`. Draft written in `paper/neurips_2026.tex`
  (+ `references.bib`, `figures/noisy_views.pdf`), structure as the supervisor asked: Flow-GRPO background -> what is new
  (scheduled GRPO + noise judge) -> NoisyCLIP -> pruning -> GRPO results (confirmations only) -> OCR gradient diagnosis ->
  remedies. Red `\tbd{}` placeholders for the option-1 results; checklist not filled yet; bib entries marked VERIFY.
  Upload bundle: `paper_overleaf.zip`. No LaTeX on the cluster: compile in Overleaf.
- **Option 1 RESULT (627262 done 2026-11-14 16:45, 22.6 h; eval 627263 done 17:15): FIRST REAL OCR GAIN.**
  Confirmation ckpt-500 (pre-chosen, new seeds, 1024 imgs, vs base 10): OCR 0.1025 -> 0.1875 (+0.085 [+0.064,+0.106]*),
  exact 1.3% -> 4.0% (+2.7pp*), text CLIP +0.0164*, text PickScore +0.0120* (= base 20 steps); alignment CLIP +0.0002 n.s.,
  PickScore -0.0013* (CI upper 0.0000), neg-acc n.s. **vs base 50 steps: OCR +0.006 n.s., exact +0.3pp n.s.** -> a 10-step
  model at the 50-step OCR level (the thesis target). In-training evals: OCR significant at every ckpt, rising (+0.029 ep50,
  +0.085 ep300, +0.108 ep500). Caveat: grids show bigger/cleaner letters but the word is often written several times
  (PaddleOCR substring credit may reward repetition). Paper placeholders filled (paper_overleaf.zip rebuilt).

### Next-step chain (user 2026-11-15: "queue all of those jobs in that specific order"), serial on nsdl2
1. 627451 `ocr_robustness.slurm` -> `logs/ocr_robustness/v1`: `scripts/ocr_robustness.py` regenerates the confirmation OCR
   images (seed 2024 x 8) for base 10/15/20/50 and ocr_long ckpt-500 at 10/15/20 steps; metrics paddle (training reward),
   strict (no substring credit), strict_exact, line_exact, line_score, repeat/n_occ, TrOCR-large greedy reading (trocr_exact/
   score; second recogniser, PaddleOCR detector shared). EasyOCR install was denied by the permission system (needs user OK).
2. 627452 `train_ocr_long_seed2.slurm` (config `ocr_long_win23_seed2_1gpu`, seed 43) -> 3. 627453 eval_run_confirm seed2 500.
   train_sd15.py: sampler seed now = config.seed (was hard-coded 42) and same_latent base seed offset (config.seed-42)*7919
   (seed 42 unchanged).
4. 627454 `train_ocr_long_cont.slurm` (`ocr_long_win23_cont_1gpu`: train.lora_path = ocr_long ckpt-500, +500 epochs, fresh
   optimizer, KL to base; its checkpoint-N = total epoch 500+N) -> 5. 627455 eval_run_confirm cont 500.
   ETA: robustness ~1 h, each training ~23 h, each eval ~30 min -> all done ~2026-11-17 evening.
- 2026-11-15: user installed EasyOCR to third_party/easyocr_pkgs (weights in third_party/easyocr_models). ocr_robustness.py
  --easyocr adds easy_* metrics (independent detector+recogniser). Job 627456 (ocr_robustness_easy.slurm -> logs/ocr_robustness/
  v2_easyocr) inserted after 627451; 627452 (seed2) now depends on 627456.
- 2026-11-15: user asked to use the A100s where possible. 627456 (EasyOCR robustness, has its own in-job base -> paired on
  A100), 627452 (seed2 training) and 627454 (cont training) moved to `ExcNodeList=nsdl2` (A100 nodes), time limit 4 days.
  Evals 627453/627455 stay on nsdl2 (paired with the H200 base references). Dependencies unchanged (serial chain).
  Expect ~40 h per training on A100 (vs 23 h H200).
- 2026-11-15 (later): user asked to put the long jobs back on the H200: 627452 and 627454 back to NodeList=nsdl2 (limit 60 h);
  only 627456 (EasyOCR check, ~1 h) stays on the A100. Chain still serial.
- **Robustness v1 (627451, H200, `logs/ocr_robustness/v1`):** ckpt-500 at 10 steps vs base 10: paddle +0.085*, strict (no
  substring) +0.063*, line_exact 0.7% -> 2.7%*, TrOCR exact 1.9% -> 6.7%*, repeat (>=2 occurrences) 0.0% -> 0.3% n.s., n_lines
  2.54 vs 2.46 -> repetition does NOT explain the gain. vs base 50: paddle/strict +0.006 n.s., best-line +0.067*, TrOCR score
  +0.070*. Gain transfers: at 15 steps +0.092*, at 20 steps 0.234 (> base 50 by +0.052*). Paper: new Table tab:robust.
- **Robustness v2 EasyOCR (627456, A100 nscluster, `logs/ocr_robustness/v2_easyocr`):** ours 10 vs base 10: easy substring +0.058*,
  easy_strict +0.046*, easy_line_exact 0.5% -> 1.4%*, easy_repeat 0 for both. vs base 50: ours 10 easy_strict -0.011 n.s., ours 20 +0.037*.
  15/20 steps vs same-step base: easy_strict +0.067*/+0.062*. Paper placeholder filled (2026-11-16).
- **Seed 2 (627452 done 2026-11-16 10:45, eval 627453 done 11:16):** confirmation ckpt-500 vs base 10 (new seeds, H200): OCR 0.150
  (+0.047 [+0.031,+0.064]*), exact +1.2pp (borderline), text CLIP +0.0081*, text PickScore +0.0069*, alignment unchanged (PickScore -0.0011 n.s.).
  Gain reproduces but smaller than seed 1 (+0.085; seed1 - seed2 = +0.038*) and below base 50 (0.181). In-training evals: seed 2
  +0.037/+0.059/+0.055/+0.076/+0.055 at ep100-500 (seed 1 +0.034/+0.053/+0.085/+0.059/+0.108) -> large checkpoint-to-checkpoint
  variance; report both seeds (mean ~+0.066). Paper not yet updated with seed 2. 627454 (continuation) started 11:16.
- 2026-11-16: paper updated with seed 2 (abstract, contributions, tab:main row, new "A second training seed" paragraph, limitations,
  conclusion now "20-50-step level, 2-5x fewer steps"), Appendix C filled (tab:hparams, prompts, OCR, evaluation). New figures by
  `scripts/paper_figures.py` -> paper/figures/{overview,nc_steps,grad_share,ocr_curves}.pdf (fig:overview in intro, fig:nc_steps,
  fig:grad_share, fig:ocr_curves). 6 figures total: check the 9-page limit in Overleaf. Backup of the pre-edit tex in the session scratchpad.
- 2026-11-16: seed-2 robustness queued: job 627600 (`ocr_robustness_seed2.slurm`, nsdl2, afterany 627455) -> `logs/ocr_robustness/v3_seed2`
  (settings base_* and seed2_* at 10/15/20 + base 50, with EasyOCR).
- Meeting report for 2026-11-17 (covers 10-16 Nov): https://claude.ai/artifact/HRBKV1ErYxEYYkvsu5MjrE, built by
  `scripts/report/build_meeting_1117.py` (images -> scripts/report/meeting1117/img/) + hand-written `meeting1117/index.html`;
  republish with root=scripts/report/meeting1117 and the img/* files.
- 2026-11-17: long MIXED run queued (user: text + alignment prompts with the schedulers; chose weight schedule only, 500 epochs):
  config `mixed_long_win23_wsched_1gpu` (= ocr_long_win23 recipe on dataset/mixed_align_ocr, weight_schedule (60,0.5,0.5), always
  10 steps), launcher `train_mixed_long.slurm` -> job 627705 (nsdl2, afterany 627600) -> `logs/phase3/mixed_long_win23_wsched`;
  eval 627706 (`eval_run_confirm.slurm mixed_long_win23_wsched 500`, afterany 627705). ~43% text prompts -> ~110k text images.
  Meeting script artifact: https://claude.ai/artifact/SULmuyLro8UEkarbbAfCPb (scripts/report/meeting1117_script/index.html).
- **Results 2026-11-18 (all jobs done; confirmations, new seeds, 1024 imgs, H200, vs base 10):**
  cont (627454, total epoch 1000) ckpt-500: OCR 0.229 (+0.127*), exact 5.5% (+4.2pp*); vs base 50 OCR +0.048*, exact +1.8pp*;
  text PickScore +0.012*; alignment PickScore -0.0027* (CLIP n.s.) -> small alignment cost grows with training. No robustness
  (repetition/strict/EasyOCR) check of cont yet.
  mixed_long_win23_wsched (627705) ckpt-500: OCR 0.175 (+0.072*, = base 50 n.s.), exact +2.3pp*; alignment CLIP +0.0059*,
  PickScore +0.0078* (= base 20 level, base 50 -0.003*), neg-acc +0.022* -> improves text AND alignment at 10 steps.
  Seed-2 robustness (627600, v3_seed2) at 10 steps: paddle +0.047*, strict +0.036*, TrOCR score +0.037*, EasyOCR strict
  +0.027*, repeat 0.1% (n.s.) -> seed-2 gain is real but no exact-match gain (line_exact/trocr_exact n.s.).

### 2026-11-18: efficiency, StitchVM, weight study (supervisor: this week StitchVM + schedules + OCR/CLIP weights)
- Efficiency (`logs/bench_efficiency/summary.txt`, job 628037, mixed recipe, 512 imgs + 1 opt. step/epoch): epoch 143 s
  (batch 8, serial OCR; GPU 16/141 GB) -> 134 s (OCR pool) -> 121 s (batch 16) -> 113 s (32) -> 109 s (64, 43 GB). OCR is
  only ~11 s/epoch on the mixed set; the limit is the fp32 LoRA UNet (tensor pipes ~17%). New options (defaults = old
  behaviour): `sample.ocr_workers` (flow_grpo/ocr_pool.py, identical scores), `train.micro_batches` (= gas for 1 step/epoch),
  `lora_autocast` (bf16 for sampling+training). train_sd15.py prints `[TIMING]` per epoch incl. check/ratio_dev and
  check/clipfrac (should be ~0 with 1 opt. step/epoch). bf16 test: job 628072 (`bench_bf16.slurm` -> logs/bench_bf16).
- StitchVM (paper `2605.19804v1.pdf`, user chose the NoisyCLIP tail as main, OpenAI CLIP tail as ablation):
  `flow_grpo/stitchvm.py` (UNet head candidates d0,d1pre,d1,d2pre,d2,d3,mid; stitch = lstsq-init 1x1 conv + zero-init MLP +
  per-position bias + CLS; CLIP tail blocks j..23), `scripts/stitchvm_build.py data|search|train` (clean latents = final x0 of
  logs/phase1_data/{sd15_10steps_eta1,sd15_50steps}, GenAI-Bench training prompts only), phase1_noisyclip.py `--stitchvm`.
  Pipeline job 628074 (`stitchvm_pipeline.slurm`, after 628072) -> logs/stitchvm/{data.npz,search_*.json,A_noisyclip,B_openai}
  and logs/phase1_noisyclip/eval10_stitchvm (paired with nc_x0phi/TAESD/VAE). Gate for GRPO: clearly beat nc_x0phi at steps 3-5.
  In GRPO the head must use the base UNet (LoRA disabled), else the judge drifts with the policy.
- Weight study configs `mixed_ratio_{08,16,32}_1gpu` (fixed OCR/align ratio, no schedule, 300 epochs; only the ratio on text
  prompts matters with per-prompt normalisation), launcher `train_mixed_ratio.slurm <cfg> <port>` (batch 32, OCR pool). Not
  submitted yet. Compare with mixed_long_win23_wsched checkpoint-300. Step schedule needs the SDE window as a fraction of
  the trajectory first (not implemented).
- **StitchVM pilot result (628074, 2026-11-18, H200, logs/phase1_noisyclip/eval10_stitchvm, 512 imgs, 10 steps eta 1, vs OpenAI
  CLIP on the final image; Spearman / within-prompt at step 5 / step 7):** search picked head d1 (16x16x640) -> CLIP block 3 for
  both tails. A (NoisyCLIP tail+teacher) 0.31/0.29, 0.59/0.40 (end 0.70: NoisyCLIP's own clean-image score caps it);
  B (OpenAI tail+teacher) 0.50/0.34, 0.77/0.48 (end 0.89); nc_x0phi 0.43/0.29, 0.54/0.33; TAESD 0.63/0.35, 0.84/0.54.
  Steps 1-4: both StitchVMs <= nc_x0phi (negative at steps 1-3: 8.6k images, forward-noised z_t at t=901 keeps signal the
  sampler's pure-noise z_T lacks). Cost 0.30 UNet steps standalone (= TAESD+CLIP). Training 7 min per model. Gate not met.
