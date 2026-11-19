"""Phase 1 prototype: NoisyCLIP Option A as a Process Reward Model for SD 1.5 (no training, no VAE).

At every denoising step t the UNet gives the predicted clean latent x0_hat. We score text-image alignment three ways:
  proj_x0  : Phi(x0_hat) -> 64x64 RGB -> CLIP          (NoisyCLIP Option A: fixed linear 4->3 projection, no VAE)
  proj_zt  : Phi(z_t)    -> 64x64 RGB -> CLIP          (projection of the raw noisy latent, baseline)
  taesd    : TAESD(x0_hat) -> 512x512 RGB -> CLIP      (tiny 2.4M-param decoder, cheap-decoder baseline)
  decoded  : VAE(x0_hat) -> 512x512 RGB -> CLIP        (expensive reference, every --decode-every steps)
  trained  : trained noise-aware scorer on Phi(x0_hat)   (--scorer <dir from phase1_train_scorer.py>; predicts the
             final image's CLIP embedding; scored against the same CLIP text embeddings)
and compare with the ORM: CLIP score of the final decoded image.

Phi is an affine 4->3 map fitted by least squares on calibration images (final latents vs their VAE decodes
average-pooled to latent resolution), on GenAI-Bench prompts that are not used for evaluation.

Per step and method it reports:
  pearson / spearman      correlation with the final image's CLIP score (all samples)
  grp_spearman            rank agreement with the final score among the seeds of the same prompt (what GRPO uses)
  disc_acc                fraction of samples whose own prompt scores higher than a mismatched prompt (detects a
                          major misalignment; subtle swaps need LLM-generated negatives, not done here)
  cost                    seconds per image for the scorer, also in UNet-step units

Usage (repo root, GPU node):  sbatch eval_latents_phase1.slurm [--num-prompts 128 --images-per-prompt 4 ...]
Outputs in --out: phi.npy, scores.npz, summary.json, curves.png, grids/.
"""
import argparse
import datetime
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from transformers import CLIPModel, CLIPTokenizer

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="runwayml/stable-diffusion-v1-5")
    p.add_argument("--lora", default=None)
    p.add_argument("--clip", default="openai/clip-vit-large-patch14")
    p.add_argument("--genai-bench", default="dataset/genai_bench")
    p.add_argument("--num-prompts", type=int, default=128)
    p.add_argument("--calib-prompts", type=int, default=32, help="Prompts used only to fit Phi")
    p.add_argument("--images-per-prompt", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-steps", type=int, default=50)
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--eta", type=float, default=0.0)
    p.add_argument("--decode-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--grid-prompts", type=int, default=4)
    p.add_argument("--scorer", default=None, help="Trained noise-aware scorer dir (phase1_train_scorer.py output)")
    p.add_argument("--noisyclip", default=None, help="HF id of NoisyCLIP (e.g. asiimo/noisyclip): adds nc_zt / nc_x0 / nc_x0phi")
    p.add_argument("--stitchvm", nargs="*", default=[],
                   help="StitchVM dirs (scripts/stitchvm_build.py train): adds svm_<dirname> scored on z_t, no x0/VAE")
    p.add_argument("--phi", default=None, help="Use this Phi (.npy) instead of fitting one (e.g. the scorer's)")
    p.add_argument("--negatives", default=None,
                   help="LLM subtle negatives json (phase1_make_negatives.py): also score each image against its "
                        "prompt's count/color/background/subject/relation negatives")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load_genai_bench(root):
    with open(os.path.join(root, "genai_image.json")) as f:
        items = json.load(f)
    with open(os.path.join(root, "genai_skills.json")) as f:
        skills = json.load(f)
    ids = sorted(items)
    prompts = {int(i): items[i]["prompt"] for i in ids}
    tags = {int(i): [s for s, members in skills.items() if int(i) in set(members)] for i in ids}
    return prompts, tags


class Clip:
    def __init__(self, name, device):
        self.model = CLIPModel.from_pretrained(name, torch_dtype=torch.float16).to(device).eval()
        try:
            self.tok = CLIPTokenizer.from_pretrained(name)
        except OSError:  # e.g. asiimo/noisyclip ships no tokenizer; its text tower is CLIP ViT-L/14's, unchanged
            self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.device = device

    @torch.no_grad()
    def text(self, prompts):
        t = self.tok(prompts, padding="max_length", truncation=True, max_length=77, return_tensors="pt").to(self.device)
        e = self.model.get_text_features(**t).float()
        return F.normalize(e, dim=-1)

    @torch.no_grad()
    def image(self, rgb01):
        """rgb01: (B,3,H,W) in [0,1] at any size; resize shortest edge to 224 + center crop, like CLIPProcessor."""
        x = rgb01.float()
        h, w = x.shape[-2:]
        s = 224 / min(h, w)
        x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bicubic", align_corners=False).clamp(0, 1)
        top, left = (x.shape[-2] - 224) // 2, (x.shape[-1] - 224) // 2
        x = x[..., top:top + 224, left:left + 224]
        x = (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
        e = self.model.get_image_features(pixel_values=x.half()).float()
        return F.normalize(e, dim=-1)


def project(latents, phi):
    """Affine 4->3 projection of (B,4,h,w) latents to RGB in [0,1]."""
    w, b = phi[:4], phi[4]
    rgb = torch.einsum("bchw,cd->bdhw", latents.float(), w) + b.view(1, 3, 1, 1)
    return ((rgb + 1) / 2).clamp(0, 1)


NC_W = torch.tensor([[60, -60, 25, -70], [60, -5, 15, -50], [60, 10, -5, -35]], dtype=torch.float32).t()  # (4,3)
NC_B = torch.tensor([150, 140, 130], dtype=torch.float32)


def nc_project(latents):
    """NoisyCLIP's fixed latent->RGB approximation (made for SDXL latents), as in novasearch/noisyclip
    src/latent_to_image.py; returns (B,3,h,w) in [0,1]."""
    rgb = torch.einsum("bchw,cd->bdhw", latents.float(), NC_W.to(latents.device)) + NC_B.to(latents.device).view(1, 3, 1, 1)
    return (rgb.clamp(0, 255).floor() / 255.0)


@torch.no_grad()
def decode(pipe, latents):
    img = pipe.vae.decode((latents / pipe.vae.config.scaling_factor).to(pipe.vae.dtype), return_dict=False)[0]
    return ((img.float() + 1) / 2).clamp(0, 1)


@torch.no_grad()
def decode_taesd(taesd, latents):
    """TAESD takes the UNet-space (already scaled) SD latents directly (its scaling_factor is 1.0)."""
    img = taesd.decode(latents.to(taesd.dtype), return_dict=False)[0]
    return ((img.float() + 1) / 2).clamp(0, 1)


@torch.no_grad()
def sd_text(pipe, prompts, device):
    ids = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt").input_ids.to(device)
    return pipe.text_encoder(ids)[0]


def sync():
    torch.cuda.synchronize()


@torch.no_grad()
def run_trajectories(pipe, prompts, neg, args, generator, step_cb=None):
    """Deterministic/stochastic DDIM sampling. Calls step_cb(i, z_t, x0_hat) after each step. Returns final latents
    and the UNet time per step."""
    device = pipe._execution_device
    cond = torch.cat([neg.repeat(len(prompts), 1, 1), sd_text(pipe, prompts, device)])
    latents = pipe.prepare_latents(len(prompts), pipe.unet.config.in_channels, 512, 512, cond.dtype, device,
                                   generator)
    pipe.scheduler.set_timesteps(args.num_steps, device=device)
    unet_t = 0.0
    for i, t in enumerate(pipe.scheduler.timesteps):
        sync(); t0 = time.perf_counter()
        inp = pipe.scheduler.scale_model_input(torch.cat([latents] * 2), t)
        eps = pipe.unet(inp, t, encoder_hidden_states=cond, return_dict=False)[0]
        eu, ec = eps.chunk(2)
        eps = eu + args.guidance_scale * (ec - eu)
        out = pipe.scheduler.step(eps, t, latents, eta=args.eta, generator=generator)
        sync(); unet_t += time.perf_counter() - t0
        if step_cb is not None:
            step_cb(i, latents, out.pred_original_sample)
        latents = out.prev_sample
    return latents, unet_t / args.num_steps


def fit_phi(pipe, prompts, neg, args, generator, device):
    zs, ys = [], []
    for b0 in range(0, len(prompts), args.batch_size):
        lat, _ = run_trajectories(pipe, prompts[b0:b0 + args.batch_size], neg, args, generator)
        img = decode(pipe, lat) * 2 - 1                      # [-1,1]
        img = F.avg_pool2d(img, kernel_size=8)                # latent resolution
        zs.append(lat.float().permute(0, 2, 3, 1).reshape(-1, 4).cpu())
        ys.append(img.permute(0, 2, 3, 1).reshape(-1, 3).cpu())
    z, y = torch.cat(zs), torch.cat(ys)
    x = torch.cat([z, torch.ones(len(z), 1)], dim=1)
    phi = torch.linalg.lstsq(x, y).solution                  # (5,3)
    r2 = 1 - ((x @ phi - y) ** 2).sum(0) / ((y - y.mean(0)) ** 2).sum(0)
    return phi.to(device), r2.tolist()


def safe(fn, a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(fn(a, b)[0])


def metrics(own, mis, final, groups):
    m = {"pearson": safe(pearsonr, own, final), "spearman": safe(spearmanr, own, final)}
    g = [safe(spearmanr, own[groups == k], final[groups == k]) for k in np.unique(groups)]
    m["grp_spearman"] = float(np.nanmean(g)) if np.any(~np.isnan(g)) else float("nan")
    ok = ~np.isnan(own)
    m["disc_acc"] = float((own[ok] > mis[ok]).mean()) if ok.any() else float("nan")
    m["mean_score"] = float(np.nanmean(own)) if ok.any() else float("nan")
    return m


def save_grid(path, frames, final, rows, thumb=160):
    """frames: list over steps of (B,3,h,w) in [0,1] (projection previews); last column = final image."""
    cols = len(frames) + 1
    grid = Image.new("RGB", (cols * thumb, len(rows) * thumb), "white")
    for r, n in enumerate(rows):
        for c, img in enumerate([f[n] for f in frames] + [final[n]]):
            arr = (img.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
            grid.paste(Image.fromarray(arr).resize((thumb, thumb), Image.NEAREST), (c * thumb, r * thumb))
    grid.save(path)


def main():
    args = parse_args()
    out = args.out or os.path.join("logs", "phase1_noisyclip", datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))
    os.makedirs(os.path.join(out, "grids"), exist_ok=True)
    with open(os.path.join(out, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = torch.device("cuda")

    pipe = StableDiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    if args.lora:
        from peft import PeftModel
        pipe.unet = PeftModel.from_pretrained(pipe.unet, args.lora).merge_and_unload()
    pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    clip = Clip(args.clip, device)
    nclip = Clip(args.noisyclip, device) if args.noisyclip else None
    from diffusers import AutoencoderTiny
    taesd = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=torch.float16).to(device).eval()

    prompts_by_id, tags = load_genai_bench(args.genai_bench)
    ids = sorted(prompts_by_id)
    random.Random(args.seed).shuffle(ids)
    calib_ids = ids[:args.calib_prompts]
    eval_ids = ids[args.calib_prompts:args.calib_prompts + args.num_prompts]
    # mismatched prompt for each eval prompt: another eval prompt (a "major" misalignment)
    mis_of = {pid: eval_ids[(j + len(eval_ids) // 2) % len(eval_ids)] for j, pid in enumerate(eval_ids)}

    neg = sd_text(pipe, [""], device)
    generator = torch.Generator(device=device).manual_seed(args.seed)

    phi, r2 = fit_phi(pipe, [prompts_by_id[i] for i in calib_ids], neg, args, generator, device)
    if args.phi:  # still run the calibration trajectories above so the eval samples stay identical
        phi = torch.from_numpy(np.load(args.phi)).to(device)
        print(f"using Phi from {args.phi} (freshly fitted R^2 would be {[round(v, 3) for v in r2]})", flush=True)
    else:
        print(f"Phi fitted on {len(calib_ids)} calibration images, R^2 per RGB channel: {[round(v, 3) for v in r2]}",
              flush=True)
    np.save(os.path.join(out, "phi.npy"), phi.cpu().numpy())
    svms = {}
    if args.stitchvm:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from flow_grpo.stitchvm import load_stitchvm, unet_head_features
        for d in args.stitchvm:
            m, meta = load_stitchvm(d, device)
            svms["svm_" + os.path.basename(os.path.normpath(d))] = (m, meta["head"])
            print(f"StitchVM {d}: head {meta['head']} -> {meta['clip']} block {meta['j']} (epoch {meta['epoch']})", flush=True)
    scorer = None
    if args.scorer:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from phase1_train_scorer import load_scorer
        scorer = load_scorer(args.scorer, args.clip, device)

    NEG_CATS = ["count", "color", "background", "subject", "relation"]
    negs = None
    if args.negatives:
        with open(args.negatives) as f:
            negs = {int(k): v for k, v in json.load(f).items()}
        missing = [p for p in eval_ids if p not in negs]
        assert not missing, f"{len(missing)} eval prompts have no negatives"

    flat = [pid for pid in eval_ids for _ in range(args.images_per_prompt)]
    groups = np.repeat(np.arange(len(eval_ids)), args.images_per_prompt)
    N, S = len(flat), args.num_steps
    dec_steps = sorted(set(range(args.decode_every - 1, S, args.decode_every)) | {S - 1})
    scores = {k: np.full((N, S), np.nan, dtype=np.float32)
              for k in ["proj_x0_own", "proj_x0_mis", "proj_zt_own", "proj_zt_mis", "taesd_own", "taesd_mis",
                        "dec_own", "dec_mis", "trained_own", "trained_mis",
                        "nc_zt_own", "nc_zt_mis", "nc_x0_own", "nc_x0_mis", "nc_x0phi_own", "nc_x0phi_mis"]
              + [f"{k}_{w}" for k in svms for w in ("own", "mis")]}
    final_own, final_mis = np.zeros(N, np.float32), np.zeros(N, np.float32)
    # subtle negatives: score of each method against each category's negative prompt, NaN where not applicable
    neg_meths = ["proj_x0", "taesd", "dec"] + (["trained"] if args.scorer else []) + (["nc_zt", "nc_x0"] if nclip else [])
    neg_scores = ({m: np.full((N, S, len(NEG_CATS)), np.nan, np.float32) for m in neg_meths} if negs else {})
    final_neg = np.full((N, len(NEG_CATS)), np.nan, np.float32)
    t_unet, t_proj, t_dec, t_taesd, n_proj, n_dec, n_taesd = 0.0, 0.0, 0.0, 0.0, 0, 0, 0
    t_trained, n_trained = 0.0, 0
    t_nc, n_nc = 0.0, 0
    t_svm, n_svm = 0.0, 0
    grids_done = 0

    for b0 in range(0, N, args.batch_size):
        bp = flat[b0:b0 + args.batch_size]
        B = len(bp)
        te_own = clip.text([prompts_by_id[p] for p in bp])
        te_mis = clip.text([prompts_by_id[mis_of[p]] for p in bp])
        frames = []
        if nclip is not None:  # NoisyCLIP keeps CLIP's text tower; computed with its own model to be safe
            nte_own = nclip.text([prompts_by_id[p] for p in bp])
            nte_mis = nclip.text([prompts_by_id[mis_of[p]] for p in bp])
        if negs:
            # (B, C, D) negative text embeddings; NaN rows for categories that do not apply
            te_neg = torch.full((B, len(NEG_CATS), te_own.shape[-1]), float("nan"), device=device)
            for c, cat in enumerate(NEG_CATS):
                rows = [k for k, p in enumerate(bp) if negs[p].get(cat)]
                if rows:
                    te_neg[rows, c] = clip.text([negs[bp[k]][cat] for k in rows])
            if nclip is not None:
                nte_neg = torch.full_like(te_neg, float("nan"))
                for c, cat in enumerate(NEG_CATS):
                    rows = [k for k, p in enumerate(bp) if negs[p].get(cat)]
                    if rows:
                        nte_neg[rows, c] = nclip.text([negs[bp[k]][cat] for k in rows])

        def neg_score(meth, i, e):
            if negs:
                neg_scores[meth][b0:b0 + B, i] = torch.einsum("bd,bcd->bc", e, te_neg).cpu().numpy()

        def cb(i, z_t, x0):
            nonlocal t_proj, t_dec, t_taesd, n_proj, n_dec, n_taesd, t_trained, n_trained, t_nc, n_nc, t_svm, n_svm
            for k, (m, head) in svms.items():  # StitchVM: frozen UNet head on z_t (null text) + stitched NoisyCLIP/CLIP tail
                sync(); t0 = time.perf_counter()
                feat = unet_head_features(pipe.unet, z_t, pipe.scheduler.timesteps[i], neg, [head])[head]
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    e = m(feat)
                sync(); t_svm += time.perf_counter() - t0; n_svm += B
                scores[f"{k}_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
                scores[f"{k}_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
            if nclip is not None:
                sync(); t0 = time.perf_counter()
                e = nclip.image(nc_project(z_t))
                sync(); t_nc += time.perf_counter() - t0; n_nc += B
                for nm, ee in [("nc_zt", e), ("nc_x0", nclip.image(nc_project(x0))), ("nc_x0phi", nclip.image(project(x0, phi)))]:
                    scores[f"{nm}_own"][b0:b0 + B, i] = (ee * nte_own).sum(-1).cpu().numpy()
                    scores[f"{nm}_mis"][b0:b0 + B, i] = (ee * nte_mis).sum(-1).cpu().numpy()
                    if nm in neg_scores:
                        neg_scores[nm][b0:b0 + B, i] = torch.einsum("bd,bcd->bc", ee, nte_neg).cpu().numpy()
            if scorer is not None:
                sync(); t0 = time.perf_counter()
                tt = pipe.scheduler.timesteps[i].to(device).long().expand(B)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    e = scorer(project(x0, phi), tt, x0 if scorer.latent_input else None)
                e = e.float()
                sync(); t_trained += time.perf_counter() - t0; n_trained += B
                scores["trained_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
                scores["trained_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
                neg_score("trained", i, e)
            sync(); t0 = time.perf_counter()
            e = clip.image(project(x0, phi))
            sync(); t_proj += time.perf_counter() - t0; n_proj += B
            scores["proj_x0_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
            scores["proj_x0_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
            neg_score("proj_x0", i, e)
            e = clip.image(project(z_t, phi))
            scores["proj_zt_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
            scores["proj_zt_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
            sync(); t0 = time.perf_counter()
            e = clip.image(decode_taesd(taesd, x0))
            sync(); t_taesd += time.perf_counter() - t0; n_taesd += B
            scores["taesd_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
            scores["taesd_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
            neg_score("taesd", i, e)
            if i in dec_steps:
                sync(); t0 = time.perf_counter()
                e = clip.image(decode(pipe, x0))
                sync(); t_dec += time.perf_counter() - t0; n_dec += B
                scores["dec_own"][b0:b0 + B, i] = (e * te_own).sum(-1).cpu().numpy()
                scores["dec_mis"][b0:b0 + B, i] = (e * te_mis).sum(-1).cpu().numpy()
                neg_score("dec", i, e)
            if grids_done < args.grid_prompts and i % max(1, S // 10) == 0:
                frames.append(project(x0, phi).cpu())

        lat, unet_step = run_trajectories(pipe, [prompts_by_id[p] for p in bp], neg, args, generator, cb)
        t_unet += unet_step  # mean UNet step time of this batch
        final = decode(pipe, lat)
        e = clip.image(final)
        final_own[b0:b0 + B] = (e * te_own).sum(-1).cpu().numpy()
        final_mis[b0:b0 + B] = (e * te_mis).sum(-1).cpu().numpy()
        if negs:
            final_neg[b0:b0 + B] = torch.einsum("bd,bcd->bc", e, te_neg).cpu().numpy()
        if grids_done < args.grid_prompts:
            save_grid(os.path.join(out, "grids", f"batch_{b0 // args.batch_size:03d}.png"), frames, final.cpu(),
                      list(range(B)))
            grids_done += 1
        print(f"[{b0 + B}/{N}] final clip own={final_own[b0:b0 + B].mean():.3f} mis={final_mis[b0:b0 + B].mean():.3f}",
              flush=True)

    n_batches = (N + args.batch_size - 1) // args.batch_size
    unet_step_per_img = t_unet / n_batches / args.batch_size
    cost = {"unet_step_s_per_img": unet_step_per_img,
            "proj_clip_s_per_img": t_proj / max(n_proj, 1),
            "taesd_clip_s_per_img": t_taesd / max(n_taesd, 1),
            "trained_s_per_img": t_trained / max(n_trained, 1),
            "decode_clip_s_per_img": t_dec / max(n_dec, 1),
            "nc_s_per_img": t_nc / max(n_nc, 1),
            "svm_s_per_img": t_svm / max(n_svm, 1)}  # per image and per StitchVM
    cost["svm_in_unet_steps"] = cost["svm_s_per_img"] / unet_step_per_img
    cost["nc_in_unet_steps"] = cost["nc_s_per_img"] / unet_step_per_img
    cost["proj_in_unet_steps"] = cost["proj_clip_s_per_img"] / unet_step_per_img
    cost["taesd_in_unet_steps"] = cost["taesd_clip_s_per_img"] / unet_step_per_img
    cost["trained_in_unet_steps"] = cost["trained_s_per_img"] / unet_step_per_img
    cost["decode_in_unet_steps"] = cost["decode_clip_s_per_img"] / unet_step_per_img

    timesteps = pipe.scheduler.timesteps.cpu().tolist()
    per_step = []
    for i in range(S):
        row = {"step": i + 1, "timestep": int(timesteps[i]), "frac": (i + 1) / S}
        for meth in ["proj_x0", "proj_zt", "taesd", "dec"] + (["trained"] if scorer is not None else []) + \
                (["nc_zt", "nc_x0", "nc_x0phi"] if nclip is not None else []) + list(svms):
            for k, v in metrics(scores[f"{meth}_own"][:, i], scores[f"{meth}_mis"][:, i], final_own, groups).items():
                row[f"{meth}_{k}"] = v
        per_step.append(row)
    summary = {"phi_r2": r2, "cost": cost, "N": N, "prompts": len(eval_ids),
               "final": {"clip_own": float(final_own.mean()), "clip_mis": float(final_mis.mean()),
                         "disc_acc": float((final_own > final_mis).mean())},
               "per_step": per_step}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    np.savez(os.path.join(out, "scores.npz"), final_own=final_own, final_mis=final_mis, groups=groups,
             prompt_ids=np.array(flat), skills=np.array([";".join(tags[p]) for p in flat]), **scores,
             **({"neg_cats": np.array(NEG_CATS), "final_neg": final_neg,
                 **{f"{m}_neg": v for m, v in neg_scores.items()}} if negs else {}))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        x = [r["frac"] for r in per_step]
        meths = [("proj_x0", "Phi(x0_hat) (no VAE)"), ("proj_zt", "Phi(z_t)"), ("taesd", "TAESD(x0_hat)"),
                 ("dec", "VAE(x0_hat)")] + ([("trained", "trained scorer")] if scorer is not None else [])
        for meth, lab in meths:
            for a, k in zip(ax, ["spearman", "grp_spearman", "disc_acc"]):
                y = [r[f"{meth}_{k}"] for r in per_step]
                a.plot(*zip(*[(xx, yy) for xx, yy in zip(x, y) if not np.isnan(yy)]), marker=".", label=lab)
        for a, t in zip(ax, ["Spearman with final CLIP score", "within-prompt Spearman", "own > mismatched prompt"]):
            a.set_title(t); a.set_xlabel("fraction of denoising steps"); a.grid(alpha=0.3)
        ax[0].legend()
        fig.tight_layout(); fig.savefig(os.path.join(out, "curves.png"), dpi=120)
    except Exception as e:  # plotting is optional
        print("plot failed:", e)

    print(f"\nN={N} images ({len(eval_ids)} GenAI-Bench prompts x {args.images_per_prompt}), {S} DDIM steps, "
          f"eta={args.eta}, Phi R^2={[round(v, 3) for v in r2]}")
    print(f"final image: CLIP own={summary['final']['clip_own']:.3f} mismatched={summary['final']['clip_mis']:.3f} "
          f"own>mis={summary['final']['disc_acc']:.3f}")
    print(f"cost per image: UNet step {cost['unet_step_s_per_img'] * 1e3:.1f} ms | Phi+CLIP "
          f"{cost['proj_clip_s_per_img'] * 1e3:.1f} ms ({cost['proj_in_unet_steps']:.2f} UNet steps) | TAESD+CLIP "
          f"{cost['taesd_clip_s_per_img'] * 1e3:.1f} ms ({cost['taesd_in_unet_steps']:.2f} UNet steps) | VAE+CLIP "
          f"{cost['decode_clip_s_per_img'] * 1e3:.1f} ms ({cost['decode_in_unet_steps']:.2f} UNet steps)")
    hdr = (f"{'step':>4} {'frac':>5} | {'Phi(x0) sp':>10} {'grp':>5} {'disc':>5} | {'Phi(zt) sp':>10} {'grp':>5} "
           f"{'disc':>5} | {'TAESD sp':>8} {'grp':>5} {'disc':>5} | {'VAE sp':>6} {'grp':>5} {'disc':>5}")
    print(hdr); print("-" * len(hdr))
    for r in per_step:
        if r["step"] % max(1, S // 20) and r["step"] != S:
            continue
        def f(v):
            return "  nan" if np.isnan(v) else f"{v:5.2f}"
        print(f"{r['step']:>4} {r['frac']:>5.2f} | {f(r['proj_x0_spearman']):>10} {f(r['proj_x0_grp_spearman'])} "
              f"{f(r['proj_x0_disc_acc'])} | {f(r['proj_zt_spearman']):>10} {f(r['proj_zt_grp_spearman'])} "
              f"{f(r['proj_zt_disc_acc'])} | {f(r['taesd_spearman']):>8} {f(r['taesd_grp_spearman'])} "
              f"{f(r['taesd_disc_acc'])} | {f(r['dec_spearman']):>6} {f(r['dec_grp_spearman'])} "
              f"{f(r['dec_disc_acc'])}")
    if scorer is not None:
        print(f"\ntrained scorer ({args.scorer}): {cost['trained_s_per_img'] * 1e3:.1f} ms/image "
              f"({cost['trained_in_unet_steps']:.2f} UNet steps)")
        print(f"{'step':>4} {'frac':>5} | {'trained sp':>10} {'grp':>5} {'disc':>5} | {'Phi(x0) sp':>10} {'grp':>5} | "
              f"{'VAE sp':>6} {'grp':>5}")
        for r in per_step:
            if r["step"] % max(1, S // 20) and r["step"] != S:
                continue
            def g(v):
                return "  nan" if np.isnan(v) else f"{v:5.2f}"
            print(f"{r['step']:>4} {r['frac']:>5.2f} | {g(r['trained_spearman']):>10} {g(r['trained_grp_spearman'])} "
                  f"{g(r['trained_disc_acc'])} | {g(r['proj_x0_spearman']):>10} {g(r['proj_x0_grp_spearman'])} | "
                  f"{g(r['dec_spearman']):>6} {g(r['dec_grp_spearman'])}")
    for k in svms:
        print(f"\n{k}: {cost['svm_s_per_img'] * 1e3:.1f} ms/image ({cost['svm_in_unet_steps']:.2f} UNet steps, standalone head)")
        print(f"{'step':>4} | {'svm sp':>6} {'grp':>5} {'disc':>5}" + (" | nc_x0phi sp   grp" if nclip is not None else ""))
        for r in per_step:
            extra = f" | {r['nc_x0phi_spearman']:11.2f} {r['nc_x0phi_grp_spearman']:5.2f}" if nclip is not None else ""
            print(f"{r['step']:>4} | {r[k + '_spearman']:6.2f} {r[k + '_grp_spearman']:5.2f} {r[k + '_disc_acc']:5.2f}" + extra)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
