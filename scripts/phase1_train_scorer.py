"""Phase 1: train a noise-aware alignment scorer (NoisyCLIP-style PRM) for SD1.5 by distillation.

Student: CLIP ViT-L/14 vision tower (+ visual projection) with LoRA, fed the cheap linear projection Phi(x0_hat_t)
(64x64 RGB, upsampled to 224) plus a learned timestep embedding added to its output embedding.
Teacher target: the CLIP image embedding of the FINAL decoded image of the same trajectory.
So the student learns to predict, from any denoising step, where the trajectory will end up; its PRM score for a
prompt is cos(student embedding, CLIP text embedding), i.e. a predicted final CLIPScore. Text tower stays frozen.

Loss = (1 - cos(student, teacher)) + --text-weight * InfoNCE(student, text embeddings in the batch)
       + --neg-weight * SmoothL1(100 * (student margin - teacher margin)) over the prompt's LLM subtle negatives
         (--negatives, phase1_make_negatives.py), where margin = cos(emb, own text) - cos(emb, negative text).
         This distils CLIP's verdict on count/color/background/subject/relation swaps of the final image.

v2 option --latent-input: instead of only the fixed 64x64 projection Phi(x0_hat), a small conv decoder on the 4-channel
latent x0_hat adds a learned residual at 256x256 (PixelShuffle x4) on top of the upsampled Phi projection (the residual
starts at zero, so training starts from v1's input).

Data: shards from phase1_make_data.py. A fraction of training prompts (--val-frac) is held out for validation;
the evaluation prompts were never in the data. Output: <out>/lora/ (peft), <out>/time_mlp.pt, <out>/log.json.

Usage: sbatch train_phase1_scorer.slurm [--epochs 3 --lr 1e-4 ...]
"""
import argparse
import glob
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import CLIPVisionModelWithProjection

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="logs/phase1_data/sd15_50steps")
    p.add_argument("--clip", default="openai/clip-vit-large-patch14")
    p.add_argument("--out", default="logs/phase1_scorer/distill_v1")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=16)
    p.add_argument("--text-weight", type=float, default=0.0)
    p.add_argument("--val-frac", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--latent-input", action="store_true", help="v2: learned conv decoder on x0_hat latents")
    p.add_argument("--negatives", default=None, help="subtle negatives json for the margin-distillation loss")
    p.add_argument("--neg-weight", type=float, default=0.1)
    return p.parse_args()


def timestep_features(t, dim=128):
    """Sinusoidal features of the DDPM timestep (0..999)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([a.sin(), a.cos()], dim=-1)


class LatentDecoder(nn.Module):
    """Tiny conv decoder: 4-ch latent (64x64) -> RGB residual at 256x256. Last layer zero-initialised."""
    def __init__(self, width=64):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(4, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, 3 * 16, 3, padding=1), nn.PixelShuffle(4))
        nn.init.zeros_(self.net[-2].weight)
        nn.init.zeros_(self.net[-2].bias)

    def forward(self, lat):
        return self.net(lat.float())


class NoiseAwareScorer(nn.Module):
    def __init__(self, clip_name, lora_rank, latent_input=False):
        super().__init__()
        vis = CLIPVisionModelWithProjection.from_pretrained(clip_name)
        cfg = LoraConfig(r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.0,
                         target_modules=["q_proj", "k_proj", "v_proj", "out_proj"])
        self.vision = get_peft_model(vis, cfg)
        dim = vis.config.projection_dim
        self.time_mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, dim))
        nn.init.zeros_(self.time_mlp[-1].weight)
        nn.init.zeros_(self.time_mlp[-1].bias)  # starts as plain CLIP on the projection
        self.latent_input = latent_input
        self.decoder = LatentDecoder() if latent_input else None

    def forward(self, rgb01, t, lat=None):
        """rgb01: (B,3,64,64) Phi projection in [0,1]; t: (B,) DDPM timesteps; lat: (B,4,64,64) x0_hat (v2 only).
        Returns normalised (B,768) embeddings."""
        if self.decoder is not None:
            x = F.interpolate(rgb01.float(), size=(256, 256), mode="bicubic", align_corners=False)
            x = x + self.decoder(lat)
            x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False, antialias=True)
        else:
            x = F.interpolate(rgb01.float(), size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)
        x = (x - CLIP_MEAN.to(x.device)) / CLIP_STD.to(x.device)
        e = self.vision(pixel_values=x).image_embeds
        e = e + self.time_mlp(timestep_features(t))
        return F.normalize(e, dim=-1)


def load_scorer(out_dir, clip_name, device):
    """Load a trained scorer (LoRA + timestep MLP) for inference."""
    from peft import PeftModel
    model = NoiseAwareScorer.__new__(NoiseAwareScorer)
    nn.Module.__init__(model)
    vis = CLIPVisionModelWithProjection.from_pretrained(clip_name)
    model.vision = PeftModel.from_pretrained(vis, os.path.join(out_dir, "lora")).merge_and_unload()
    dim = vis.config.projection_dim
    model.time_mlp = nn.Sequential(nn.Linear(128, 256), nn.SiLU(), nn.Linear(256, dim))
    model.time_mlp.load_state_dict(torch.load(os.path.join(out_dir, "time_mlp.pt"), map_location="cpu"))
    dec_path = os.path.join(out_dir, "decoder.pt")
    model.latent_input = os.path.exists(dec_path)
    model.decoder = None
    if model.latent_input:
        model.decoder = LatentDecoder()
        model.decoder.load_state_dict(torch.load(dec_path, map_location="cpu"))
    return model.to(device).eval()


NEG_CATS = ["count", "color", "background", "subject", "relation"]


@torch.no_grad()
def negative_text_embeddings(neg_path, prompt_ids, clip_name, device):
    """(P, 5, 768) normalised CLIP text embeddings of each training prompt's negatives (NaN where missing), indexed
    by position in np.unique(prompt_ids)."""
    from transformers import CLIPModel, CLIPTokenizer
    with open(neg_path) as f:
        negs = {int(k): v for k, v in json.load(f).items()}
    clip = CLIPModel.from_pretrained(clip_name, torch_dtype=torch.float16).to(device).eval()
    tok = CLIPTokenizer.from_pretrained(clip_name)
    uniq = np.unique(prompt_ids)
    out = torch.full((len(uniq), len(NEG_CATS), clip.config.projection_dim), float("nan"), device=device)
    pairs = [(u, c, negs[p][cat]) for u, p in enumerate(uniq) if p in negs
             for c, cat in enumerate(NEG_CATS) if negs[p].get(cat)]
    for i in range(0, len(pairs), 256):
        chunk = pairs[i:i + 256]
        t = tok([x[2] for x in chunk], padding="max_length", truncation=True, max_length=77,
                return_tensors="pt").to(device)
        e = F.normalize(clip.get_text_features(**t).float(), dim=-1)
        out[[x[0] for x in chunk], [x[1] for x in chunk]] = e
    del clip
    torch.cuda.empty_cache()
    print(f"negatives: {len(pairs)} for {len(uniq)} prompts ({sum(p in negs for p in uniq)} prompts covered)",
          flush=True)
    return out


def load_data(path, load_latents=False):
    shards = sorted(glob.glob(os.path.join(path, "shard_*.npz")))
    assert shards, f"no shards in {path}"
    rgb, img, txt, pid, lat = [], [], [], [], []
    timesteps = None
    for s in shards:
        d = np.load(s)
        rgb.append(d["proj_rgb"]); img.append(d["final_img_emb"]); txt.append(d["text_emb"]); pid.append(d["prompt_ids"])
        if load_latents:
            lat.append(d["x0_lat"])
        timesteps = d["timesteps"] if timesteps is None else timesteps
        assert (d["timesteps"] == timesteps).all()
    return (np.concatenate(rgb), np.concatenate(img), np.concatenate(txt), np.concatenate(pid),
            torch.from_numpy(timesteps.astype(np.int64)), np.concatenate(lat) if load_latents else None)


def batches(n_idx, S, bs, rng):
    """All (trajectory, step) pairs, shuffled, in batches."""
    pairs = np.stack(np.meshgrid(n_idx, np.arange(S), indexing="ij"), -1).reshape(-1, 2)
    rng.shuffle(pairs)
    for i in range(0, len(pairs), bs):
        yield pairs[i:i + bs]


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")

    rgb, img, txt, pid, timesteps, lat = load_data(args.data, args.latent_input)
    N, S = rgb.shape[:2]
    prompts = np.unique(pid)
    val_prompts = set(rng.choice(prompts, size=max(1, int(len(prompts) * args.val_frac)), replace=False).tolist())
    is_val = np.array([p in val_prompts for p in pid])
    tr_idx, va_idx = np.where(~is_val)[0], np.where(is_val)[0]
    print(f"data: {N} trajectories x {S} steps; train {len(tr_idx)} / val {len(va_idx)} trajectories "
          f"({len(prompts) - len(val_prompts)} / {len(val_prompts)} prompts); timesteps {timesteps.tolist()}",
          flush=True)

    rgb_t = torch.from_numpy(rgb)                 # uint8, stays on CPU
    img_t = torch.from_numpy(img).float().to(device)
    txt_t = torch.from_numpy(txt).float().to(device)
    ts_t = timesteps.to(device)
    lat_t = torch.from_numpy(lat) if lat is not None else None   # fp16, stays on CPU
    neg_t, pidx_t = None, None
    if args.negatives:
        neg_t = negative_text_embeddings(args.negatives, pid, args.clip, device)
        pidx_t = torch.from_numpy(np.searchsorted(np.unique(pid), pid)).to(device)

    model = NoiseAwareScorer(args.clip, args.lora_rank, args.latent_input).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"trainable params: {sum(p.numel() for p in params) / 1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01)
    steps_per_epoch = math.ceil(len(tr_idx) * S / args.batch_size)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 200) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    logit_scale = 100.0

    def step_loss(pairs, train):
        n, j = torch.from_numpy(pairs[:, 0]), torch.from_numpy(pairs[:, 1])
        x = rgb_t[n, j].to(device, non_blocking=True).float() / 255
        t = ts_t[j.to(device)]
        la = lat_t[n, j].to(device, non_blocking=True) if lat_t is not None else None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            e = model(x, t, la)
        e = e.float()
        tgt, te = img_t[n.to(device)], txt_t[n.to(device)]
        cos_t = (e * tgt).sum(-1)
        loss = (1 - cos_t).mean()
        nl = torch.zeros((), device=device)
        if neg_t is not None:
            tn = neg_t[pidx_t[n.to(device)]]                                   # (B,5,D), NaN rows = missing
            ok = ~torch.isnan(tn[..., 0])
            tn = torch.nan_to_num(tn)
            ms = (e * te).sum(-1, keepdim=True) - torch.einsum("bd,bcd->bc", e, tn)
            mt = (tgt * te).sum(-1, keepdim=True) - torch.einsum("bd,bcd->bc", tgt, tn)
            if ok.any():
                nl = F.smooth_l1_loss(100 * ms[ok], 100 * mt[ok])
                loss = loss + args.neg_weight * nl
        step_loss.last_neg = nl.detach()
        if args.text_weight > 0:
            logits = logit_scale * e @ te.T
            loss = loss + args.text_weight * F.cross_entropy(logits, torch.arange(len(e), device=device))
        return loss, cos_t.detach(), j

    def evaluate():
        model.eval()
        cos_by_step = torch.zeros(S); cnt = torch.zeros(S); base_by_step = torch.zeros(S)
        with torch.no_grad():
            for pairs in batches(va_idx, S, 256, np.random.default_rng(0)):
                _, c, j = step_loss(pairs, False)
                cos_by_step.index_add_(0, j, c.cpu()); cnt.index_add_(0, j, torch.ones(len(j)))
        model.train()
        return (cos_by_step / cnt).tolist()

    log = {"args": vars(args), "timesteps": timesteps.tolist(), "val_cos_by_step": [], "train_loss": []}
    v = evaluate()
    log["val_cos_by_step"].append({"epoch": 0, "cos": v})
    print(f"epoch 0 (before training) val cos(student, final) by step: {[round(c, 3) for c in v]}", flush=True)
    model.train()
    gstep, t0, run, run_neg = 0, time.time(), [], []
    for epoch in range(1, args.epochs + 1):
        for pairs in batches(tr_idx, S, args.batch_size, rng):
            loss, _, _ = step_loss(pairs, True)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); gstep += 1
            run.append(loss.item()); run_neg.append(step_loss.last_neg.item())
            if gstep % 100 == 0:
                log["train_loss"].append({"step": gstep, "loss": float(np.mean(run)), "neg": float(np.mean(run_neg))})
                print(f"epoch {epoch} step {gstep}/{total} loss {np.mean(run):.4f} neg {np.mean(run_neg):.3f} lr {sched.get_last_lr()[0]:.2e} "
                      f"{(time.time() - t0) / gstep:.2f}s/step", flush=True)
                run, run_neg = [], []
        v = evaluate()
        log["val_cos_by_step"].append({"epoch": epoch, "cos": v})
        print(f"epoch {epoch} val cos(student, final) by step: {[round(c, 3) for c in v]}", flush=True)
        model.vision.save_pretrained(os.path.join(args.out, "lora"))
        torch.save(model.time_mlp.state_dict(), os.path.join(args.out, "time_mlp.pt"))
        if model.decoder is not None:
            torch.save(model.decoder.state_dict(), os.path.join(args.out, "decoder.pt"))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
    print(f"saved to {args.out}", flush=True)


if __name__ == "__main__":
    main()
