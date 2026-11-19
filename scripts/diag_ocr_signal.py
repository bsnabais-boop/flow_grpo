"""Why does GRPO not improve OCR? Measures, with the exact training sampler/reward/advantages of a Phase 3 config:

1. Reward signal per group (G images of the same prompt):
   - OCR mean / within-group std / max, share of text groups where no image is any good, exact matches;
   - how much of the within-group reward variance (hence of the advantage) comes from OCR vs alignment.
2. Policy gradient at the first PPO step (ratio = 1, so grad of the loss = -mean_s,j A_s * grad log p_sj):
   - per-sample gradients of the LoRA parameters (one backward per sample and transition);
   - norm of the batch gradient and its signal-to-noise ratio: ||mean|| / standard error of the mean, at the
     batch of one optimizer step in training (16 x 8 = 128 samples);
   - decomposition A = A_ocr + A_align (exact: the advantage is linear in the reward) -> which part drives the update,
     and cosine between the OCR and the alignment gradient;
   - gradient per denoising transition j (which steps carry the update);
   - KL penalty gradient (beta * grad KL to the reference) vs the policy gradient (only with a trained LoRA);
   - global grad-norm vs config.train.max_grad_norm (is every update clipped?).

Usage (one GPU, ~15 min per model):
  python scripts/diag_ocr_signal.py --config mixed_sd15_prm_1gpu --models base,logs/phase3/mixed_prm/checkpoints/checkpoint-200/lora
Output: <out>/<model_name>.json + a printed report.
"""
import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="mixed_sd15_prm_1gpu")
    p.add_argument("--models", default="base", help="comma list: 'base' or LoRA dirs")
    p.add_argument("--text-prompts", type=int, default=32)
    p.add_argument("--align-prompts", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--kl-samples", type=int, default=64, help="samples used for the KL-gradient estimate")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--skip-last", type=int, default=1, help="also report the gradient without the last k transitions")
    p.add_argument("--out", default="logs/diag_ocr_signal")
    # sampling variants (2026-11-13, "problem 2": per-update gradient is mostly noise)
    p.add_argument("--same-latent", action="store_true",
                   help="all images of a group start from the same z_T (only the SDE noise differs inside a group)")
    p.add_argument("--sde-window-size", type=int, default=None, help="override: stochastic DDIM only in a fixed window")
    p.add_argument("--sde-window-start", type=int, default=0, help="first step of the fixed SDE window")
    p.add_argument("--tag", default="", help="suffix of the output name")
    p.add_argument("--folds", type=int, default=8, help="prompt folds for the unbiased split-sample signal estimate")
    return p.parse_args()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    G = load_module("grpo_cfg", os.path.join(ROOT, "config", "grpo.py"))
    config = getattr(G, args.config)()
    T = load_module("train_sd15", os.path.join(ROOT, "scripts", "train_sd15.py"))
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, PeftModel, get_peft_model
    from flow_grpo.diffusers_patch.sd15_pipeline_with_logprob import pipeline_with_logprob
    from flow_grpo.mixed_reward import MixedReward, has_text_target

    rng = random.Random(args.seed)
    text_pool = [l.strip() for l in open("dataset/ocr_simple/train.txt") if l.strip()]
    align_pool = [l.strip() for l in open(os.path.join(config.dataset, "train.txt")) if l.strip() and not has_text_target(l)]
    prompts_all = rng.sample(text_pool, args.text_prompts) + rng.sample(align_pool, args.align_prompts)
    S = config.sample.num_steps
    if args.sde_window_size is not None:
        config.sample.sde_window_size = args.sde_window_size
        config.sample.sde_window_range = (args.sde_window_start, args.sde_window_start + args.sde_window_size)
    n_train_t = config.sample.sde_window_size if config.sample.sde_window_size > 0 else S - 1
    Gs = args.group_size
    bf16 = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    reward = MixedReward(device, align=config.sample.align_reward, prm_dir=config.sample.get("prm_dir"),
                         phi_path=config.sample.get("prm_phi"), prm_steps=tuple(config.sample.get("prm_steps", (3, 5, 7, 9))),
                         align_weight=config.sample.get("align_weight", 100.0), ocr_weight=config.sample.get("ocr_weight", 80.0),
                         noisyclip_name=config.sample.get("noisyclip_name", "asiimo/noisyclip"),
                         noisyclip_input=config.sample.get("noisyclip_input", "zt"))
    w_al, w_ocr = reward.align_weight, reward.ocr_weight

    for model in args.models.split(","):
        parts = model.strip("/").split("/")
        name = "base" if model == "base" else f"{parts[-4]}_{parts[-2]}"  # <run>_checkpoint-N
        name += args.tag
        t_start = time.time()
        torch.manual_seed(args.seed)
        pipe = StableDiffusionPipeline.from_pretrained(config.pretrained.model, safety_checker=None,
                                                       requires_safety_checker=False)
        pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
        pipe.vae.requires_grad_(False); pipe.text_encoder.requires_grad_(False); pipe.unet.requires_grad_(False)
        pipe.vae.to(device, dtype=torch.float32); pipe.text_encoder.to(device, dtype=torch.float32)
        if model == "base":
            lcfg = LoraConfig(r=config.train.lora_rank, lora_alpha=2 * config.train.lora_rank, init_lora_weights="gaussian",
                              target_modules=["to_k", "to_q", "to_v", "to_out.0"])
            pipe.unet = get_peft_model(pipe.unet, lcfg)
        else:
            pipe.unet = PeftModel.from_pretrained(pipe.unet, model, is_trainable=True)
            pipe.unet.set_adapter("default")
        pipe.unet.to(device)
        pipe.set_progress_bar_config(disable=True)
        unet = pipe.unet
        params = [p for p in unet.parameters() if p.requires_grad]
        P = sum(p.numel() for p in params)
        neg = T.compute_text_embeddings([""], pipe.text_encoder, pipe.tokenizer, device=device)

        # ---------------- 1. sample groups exactly like training ----------------
        groups = []
        for gi, prompt in enumerate(prompts_all):
            prompts = [prompt] * Gs
            emb = T.compute_text_embeddings(prompts, pipe.text_encoder, pipe.tokenizer, device=device)
            cb = reward.start(prompts, S)
            # seeded initial noise (paired across variants): shared within the group with --same-latent
            gseed = args.seed * 100003 + gi * Gs
            generator = [torch.Generator().manual_seed(gseed if args.same_latent else gseed + s) for s in range(Gs)]
            with bf16(), torch.no_grad():
                images, latents, log_probs, timesteps, _ = pipeline_with_logprob(
                    pipe, prompt=prompts, prompt_embeds=emb, negative_prompt_embeds=neg.repeat(Gs, 1, 1),
                    num_inference_steps=S, guidance_scale=config.sample.guidance_scale, output_type="pt", generator=generator,
                    height=config.resolution, width=config.resolution, noise_level=config.sample.noise_level,
                    sde_window_size=config.sample.sde_window_size, sde_window_range=config.sample.sde_window_range,
                    x0_callback=cb)
            comp = reward(images, prompts)
            groups.append(dict(prompt=prompt, text=has_text_target(prompt), emb=emb.cpu(),
                               latents=torch.stack(latents, 1).cpu(), log_probs=torch.stack(log_probs, 1).cpu(),
                               timesteps=torch.stack(timesteps).cpu(),
                               align=comp["align"].float().cpu().numpy(),
                               ocr=np.nan_to_num(comp["ocr"].float().cpu().numpy(), nan=0.0),
                               total=comp["avg"].float().cpu().numpy()))
            print(f"[{name}] sampled {gi + 1}/{len(prompts_all)}  ocr={groups[-1]['ocr'].round(2).tolist() if groups[-1]['text'] else '-'}",
                  flush=True)

        # advantages (per-prompt normalisation, as the PerPromptStatTracker with global_std=False)
        for g in groups:
            r = g["total"]
            sd = r.std() + 1e-4
            g["adv"] = np.clip((r - r.mean()) / sd, -config.train.adv_clip_max, config.train.adv_clip_max)
            g["adv_ocr"] = w_ocr * (g["ocr"] - g["ocr"].mean()) / sd if g["text"] else np.zeros(Gs)
            g["adv_al"] = w_al * (g["align"] - g["align"].mean()) / sd

        tg = [g for g in groups if g["text"]]
        ocr_m = np.array([g["ocr"].mean() for g in tg]); ocr_sd = np.array([g["ocr"].std() for g in tg])
        ocr_max = np.array([g["ocr"].max() for g in tg]); ocr_rng = np.array([g["ocr"].max() - g["ocr"].min() for g in tg])
        var_ocr = np.array([np.var(w_ocr * g["ocr"]) for g in tg]); var_al = np.array([np.var(w_al * g["align"]) for g in tg])
        var_tot = np.array([np.var(g["total"]) for g in tg])
        corr_adv_ocr = np.nanmean([np.corrcoef(g["adv"], g["ocr"])[0, 1] for g in tg if g["ocr"].std() > 0])
        signal = {
            "text_groups": len(tg), "align_groups": len(groups) - len(tg), "group_size": Gs,
            "ocr_mean": float(ocr_m.mean()), "ocr_within_group_std_mean": float(ocr_sd.mean()),
            "ocr_within_group_std_median": float(np.median(ocr_sd)),
            "ocr_group_range_mean": float(ocr_rng.mean()),
            "frac_groups_best_ocr_ge_0.5": float((ocr_max >= 0.5).mean()),
            "frac_groups_best_ocr_ge_0.8": float((ocr_max >= 0.8).mean()),
            "frac_groups_with_exact_match": float(np.mean([(g["ocr"] >= 1).any() for g in tg])),
            "frac_groups_ocr_range_lt_0.1": float((ocr_rng < 0.1).mean()),
            "exact_frac_images": float(np.mean([(g["ocr"] >= 1).mean() for g in tg])),
            "align_within_group_std_mean_text": float(np.mean([g["align"].std() for g in tg])),
            "align_within_group_std_mean_alignprompts": float(np.mean([g["align"].std() for g in groups if not g["text"]] or [np.nan])),
            "share_of_reward_variance_from_ocr": float(np.mean(var_ocr / np.maximum(var_tot, 1e-12))),
            "share_of_reward_variance_from_align": float(np.mean(var_al / np.maximum(var_tot, 1e-12))),
            "corr_advantage_vs_ocr_within_group": float(corr_adv_ocr),
            "ocr_per_group": [g["ocr"].round(3).tolist() for g in tg],
        }

        # ---------------- 2. per-sample policy gradients ----------------
        def flat_grad():
            return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1).float() for p in params])

        z = lambda: torch.zeros(P, device=device)
        acc = {k: z() for k in ["tot", "ocr", "al", "tot_text", "tot_align", "kl"]}
        acc_j = [z() for _ in range(n_train_t)]
        acc_kl_j = [z() for _ in range(n_train_t)]
        ss_j = [0.0] * n_train_t
        K = n_train_t - args.skip_last  # transitions kept in the "without last" variant
        for k in ["nolast", "nolast_text"]:
            acc[k] = z()
        ss = {"tot": 0.0, "tot_text": 0.0, "tot_align": 0.0, "ocr": 0.0, "al": 0.0, "nolast": 0.0, "nolast_text": 0.0}
        n = {"all": 0, "text": 0, "align": 0}
        F = args.folds
        fold_sum = [z() for _ in range(F)]  # per-fold sums of the trained ("nolast") per-sample gradients
        fold_n = [0] * F
        gl_norms, kl_vals, n_kl = [], [], 0
        unet.train()
        for gidx, g in enumerate(groups):
            emb = torch.cat([neg.repeat(Gs, 1, 1), g["emb"].to(device)])  # cfg: [uncond, cond]
            for s in range(Gs):
                sample = {"latents": g["latents"][s:s + 1, :-1].to(device), "next_latents": g["latents"][s:s + 1, 1:].to(device),
                          "timesteps": g["timesteps"].unsqueeze(0).to(device), "log_probs": g["log_probs"][s:s + 1].to(device)}
                e_s = torch.cat([emb[s:s + 1], emb[Gs + s:Gs + s + 1]])
                gl = z()
                do_kl = model != "base" and n_kl < args.kl_samples
                for j in range(n_train_t):
                    unet.zero_grad(set_to_none=True)
                    with bf16():
                        _, logp, mean, std = T.compute_log_prob(unet, pipe, sample, j, e_s, config)
                    logp.sum().backward()
                    gj = flat_grad()
                    gl += gj
                    acc_j[j] += float(g["adv"][s]) * gj
                    ss_j[j] += float((float(g["adv"][s]) * gj).pow(2).sum())
                    if j == K - 1:
                        gl_keep = gl.clone()
                    if do_kl:
                        unet.zero_grad(set_to_none=True)
                        with bf16():
                            _, _, mean, std = T.compute_log_prob(unet, pipe, sample, j, e_s, config)
                            with torch.no_grad(), unet.disable_adapter():
                                _, _, mean_ref, _ = T.compute_log_prob(unet, pipe, sample, j, e_s, config)
                        kl = (((mean - mean_ref) ** 2).mean(dim=(1, 2, 3), keepdim=True) / (2 * std ** 2)).mean()
                        (config.train.beta * kl).backward()
                        gk = flat_grad()
                        acc["kl"] += gk
                        acc_kl_j[j] += gk
                        kl_vals.append(float(kl))
                if do_kl:
                    n_kl += 1
                gl_norms.append(float(gl.norm()))
                a, ao, aa = float(g["adv"][s]), float(g["adv_ocr"][s]), float(g["adv_al"][s])
                c = a * gl
                acc["tot"] += c; ss["tot"] += float(c.pow(2).sum())
                acc["ocr"] += ao * gl; ss["ocr"] += float((ao * gl).pow(2).sum())
                acc["al"] += aa * gl; ss["al"] += float((aa * gl).pow(2).sum())
                cn = a * gl_keep
                acc["nolast"] += cn; ss["nolast"] += float(cn.pow(2).sum())
                fi = gidx % F
                fold_sum[fi] += cn; fold_n[fi] += 1
                if g["text"]:
                    acc["nolast_text"] += cn; ss["nolast_text"] += float(cn.pow(2).sum())
                key = "text" if g["text"] else "align"
                acc[f"tot_{key}"] += c; ss[f"tot_{key}"] += float(c.pow(2).sum())
                n["all"] += 1; n[key] += 1
            print(f"[{name}] gradients for '{g['prompt'][:50]}' done ({n['all']} samples)", flush=True)
        unet.zero_grad(set_to_none=True)

        def snr(sum_vec, sumsq, N, batch=128):
            m = sum_vec / N
            var = max(sumsq / N - float(m.pow(2).sum()), 1e-30)
            return float(m.norm()), float(m.norm() / math.sqrt(var / N)), float(m.norm() / math.sqrt(var / batch))

        cos = lambda a, b: float(torch.dot(a, b) / (a.norm() * b.norm() + 1e-30))

        # Unbiased signal estimate for the trained transitions: ||mu||^2 from cross-fold dot products (samples in
        # different folds are independent, so E<s_f, s_g> = n_f n_g ||mu||^2), noise tr(Sigma) from the sum of squares.
        # rho2 = ||mu||^2 / tr(Sigma) per image; signal/noise of a batch of B images = sqrt(rho2 * B). Jackknife over folds.
        Gm = torch.stack(fold_sum)
        Gram = (Gm @ Gm.T).double().cpu().numpy(); del Gm
        fn = np.array(fold_n, dtype=float)
        ss_fold_total = ss["nolast"]
        N_tr = n["all"]
        # raw statistics first, so a bug in the analysis below cannot lose a run
        np.savez(os.path.join(args.out, f"{name}_raw.npz"), gram=Gram, fold_n=fn, ss=ss_fold_total, N=N_tr,
                 ocr=np.array([g["ocr"] for g in groups]), total=np.array([g["total"] for g in groups]))

        def rho2(keep):
            idx = np.array(keep)
            Gk = Gram[np.ix_(idx, idx)]; nk = fn[idx]
            off = Gk.sum() - np.trace(Gk); pairs = nk.sum() ** 2 - (nk ** 2).sum()
            mu2 = off / pairs
            Ntot = nk.sum(); mean2 = Gk.sum() / Ntot ** 2
            return mu2 / max(ss_fold_total / N_tr - mean2, 1e-30)

        snr_b = lambda r, B: math.copysign(math.sqrt(abs(r) * B), r)
        r_all = rho2(list(range(F)))
        jk = np.array([rho2([f for f in range(F) if f != k]) for k in range(F)])
        r_se = float(np.sqrt((F - 1) / F * ((jk - jk.mean()) ** 2).sum()))
        Nall = n["all"]
        # gradient of the PPO loss at ratio 1 = -(1/(N*Tn)) sum A_s grad logp ; report norms in loss units
        scale = 1.0 / (Nall * n_train_t)
        m_tot, snr_N, snr_128 = snr(acc["tot"], ss["tot"], Nall)
        grad = {
            "lora_params": P,
            "policy_grad_norm_loss_units": float(acc["tot"].norm() * scale),
            "per_sample_score_grad_norm_mean": float(np.mean(gl_norms)),
            "snr_this_batch": snr_N, "snr_at_training_batch_128": snr_128,
            # ||mean||^2 is inflated by the noise (E||m||^2 = ||mu||^2 + tr(Sigma)/N): pure noise gives snr_this_batch ~ 1.
            # Bias-corrected signal/noise of a batch B: sign(d) * sqrt(|d| * B / N), d = snr_this_batch^2 - 1.
            "snr_debiased_at_128": math.copysign(math.sqrt(abs(snr_N ** 2 - 1) * 128 / Nall), snr_N ** 2 - 1),
            "snr_debiased_at_1024": math.copysign(math.sqrt(abs(snr_N ** 2 - 1) * 1024 / Nall), snr_N ** 2 - 1),
            "snr_text_prompts_at_128": snr(acc["tot_text"], ss["tot_text"], n["text"])[2] if n["text"] else None,
            "snr_align_prompts_at_128": snr(acc["tot_align"], ss["tot_align"], n["align"])[2] if n["align"] else None,
            "snr_ocr_part_at_128": snr(acc["ocr"], ss["ocr"], Nall)[2],
            "snr_align_part_at_128": snr(acc["al"], ss["al"], Nall)[2],
            "norm_ocr_part_over_total": float(acc["ocr"].norm() / acc["tot"].norm()),
            "norm_align_part_over_total": float(acc["al"].norm() / acc["tot"].norm()),
            "cos_ocr_part_vs_align_part": cos(acc["ocr"], acc["al"]),
            "cos_total_vs_ocr_part": cos(acc["tot"], acc["ocr"]),
            "per_transition_norm_share": [float(v.norm() / sum(float(w.norm()) for w in acc_j)) for v in acc_j],
            "per_transition_cos_with_total": [cos(v, acc["tot"]) for v in acc_j],
            "max_grad_norm": config.train.max_grad_norm,
            # per-transition SNR at batch 128 / 256 (is a single transition's gradient above the noise?)
            "per_transition_snr_at_128": [snr(acc_j[j], ss_j[j], Nall)[2] for j in range(n_train_t)],
            "per_transition_snr_at_256": [snr(acc_j[j], ss_j[j], Nall, 256)[2] for j in range(n_train_t)],
            # without the last k transitions (what train.skip_last=k trains on)
            "skip_last": args.skip_last,
            "nolast_snr_at_128": snr(acc["nolast"], ss["nolast"], Nall)[2],
            "nolast_snr_at_256": snr(acc["nolast"], ss["nolast"], Nall, 256)[2],
            "nolast_snr_at_384": snr(acc["nolast"], ss["nolast"], Nall, 384)[2],
            "nolast_text_snr_at_128": snr(acc["nolast_text"], ss["nolast_text"], n["text"])[2] if n["text"] else None,
            "nolast_text_snr_at_256": snr(acc["nolast_text"], ss["nolast_text"], n["text"], 256)[2] if n["text"] else None,
            "snr_at_training_batch_256": snr(acc["tot"], ss["tot"], Nall, 256)[2],
            "trained_rho2_per_image": float(r_all), "trained_rho2_se": r_se,
            "trained_snr_unbiased_at_256": snr_b(r_all, 256),
            "trained_snr_unbiased_at_256_ci": [snr_b(r_all - 1.96 * r_se, 256), snr_b(r_all + 1.96 * r_se, 256)],
            "trained_snr_unbiased_at_1024": snr_b(r_all, 1024),
            "images_for_snr_1": float(1 / r_all) if r_all > 0 else None,
            "same_latent": args.same_latent, "sde_window": [args.sde_window_start, args.sde_window_size],
            "trained_transitions": K,
            "nolast_per_transition_norm_share": [float(acc_j[j].norm() / sum(float(acc_j[i].norm()) for i in range(K))) for j in range(K)],
            "nolast_per_transition_cos_with_nolast_total": [cos(acc_j[j], acc["nolast"]) for j in range(K)],
        }
        if n_kl:
            kscale = 1.0 / (n_kl * n_train_t)
            grad["kl_mean"] = float(np.mean(kl_vals))
            grad["kl_grad_norm_loss_units"] = float(acc["kl"].norm() * kscale)
            grad["kl_grad_over_policy_grad"] = grad["kl_grad_norm_loss_units"] / grad["policy_grad_norm_loss_units"]
            grad["cos_kl_grad_vs_neg_policy_grad"] = cos(acc["kl"], -acc["tot"])
            grad["kl_per_transition_norm_share"] = [float(v.norm() / sum(float(w.norm()) for w in acc_kl_j)) for v in acc_kl_j]
            kl_keep = sum(acc_kl_j[:K]); pol_keep = sum(acc_j[:K])
            # in loss units of a run with skip_last: policy mean over Nall*K, KL mean over n_kl*K (beta of this config)
            grad["nolast_kl_grad_over_policy_grad"] = float((kl_keep.norm() / (n_kl * K)) / (pol_keep.norm() / (Nall * K)))
        # gradient that one optimizer step sees: the training loss is scaled 1/(gas*T) and summed over 16 batches of 8
        # = mean over 128 samples x T transitions -> same units as policy_grad_norm_loss_units (expected value)
        grad["note"] = ("policy_grad_norm is the norm of the expected first-step gradient (mean over samples and transitions); "
                        "with max_grad_norm=%s an update is clipped when the batch gradient norm exceeds it" % config.train.max_grad_norm)
        hyper = {k: str(v) for k, v in {
            "learning_rate": config.train.learning_rate, "clip_range": config.train.clip_range, "beta_kl": config.train.beta,
            "adv_clip_max": config.train.adv_clip_max, "max_grad_norm": config.train.max_grad_norm,
            "lora_rank": config.train.lora_rank, "samples_per_epoch": config.sample.num_batches_per_epoch * config.sample.train_batch_size,
            "optimizer_steps_per_epoch": config.sample.num_batches_per_epoch // config.train.gradient_accumulation_steps,
            "group_size": config.sample.num_image_per_prompt, "num_steps": S, "noise_level(eta)": config.sample.noise_level,
            "guidance": config.sample.guidance_scale, "weights(align,ocr)": (w_al, w_ocr), "global_std": config.sample.global_std}.items()}
        res = {"model": model, "config": args.config, "hyper": hyper, "signal": signal, "gradient": grad,
               "minutes": (time.time() - t_start) / 60}
        with open(os.path.join(args.out, f"{name}.json"), "w") as f:
            json.dump(res, f, indent=2)
        print(f"\n===== {name} =====")
        for k, v in signal.items():
            if k != "ocr_per_group":
                print(f"  {k:45s} {v}")
        for k, v in grad.items():
            print(f"  {k:45s} {v}")
        del pipe, unet, params, acc, acc_j, acc_kl_j, fold_sum
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
