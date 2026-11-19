"""Build a StitchVM value model for SD1.5 (flow_grpo/stitchvm.py; Go et al. 2026, arXiv 2605.19804). 2026-11-18.

  data    clean SD1.5 latents (final x0 of the phase-1 trajectories, GenAI-Bench TRAINING prompts only) + their VAE
          decodes at 224 px -> <out>/data.npz
  search  stage 1: closed-form least-squares fit of every (UNet head candidate i, CLIP block j<=4) pair on a probe set;
          reports held-out relative MSE -> <out>/search_<clip>.json
  train   stage 2: init F by least squares, finetune stitch + CLIP tail by multi-level distillation from the same frozen
          CLIP on the clean image (paper Eq. 8/78) -> <out>/<name>/stitchvm.pt (+ val metrics per timestep)

Noisy latents follow the forward process z_t = sqrt(abar_t) z0 + sqrt(1-abar_t) eps (off-policy, as in the paper);
t is drawn from the 10-step DDIM grid used by GRPO (901..1) with prob. --p-grid, else uniformly from 1..999.
Usage: sbatch stitchvm.slurm <subcommand> [args]   (GPU)
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from flow_grpo.stitchvm import CANDIDATES, CAND_CH, StitchVM, clip_tokens, to_grid, unet_head_features  # noqa: E402

CLIP_MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
CLIP_STD = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
SD = "runwayml/stable-diffusion-v1-5"
GRID10 = [901, 801, 701, 601, 501, 401, 301, 201, 101, 1]


def load_sd(device):
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    pipe = StableDiffusionPipeline.from_pretrained(SD, torch_dtype=torch.float16, safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)
    ids = pipe.tokenizer([""], padding="max_length", max_length=77, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        null = pipe.text_encoder(ids)[0]
    return pipe, null


def pixel_values(img_u8, device):
    x = img_u8.to(device).float() / 255.0
    return ((x - CLIP_MEAN.to(device)) / CLIP_STD.to(device))


def noisy(pipe, z0, t):
    ab = pipe.scheduler.alphas_cumprod.to(z0.device)[t].view(-1, 1, 1, 1).float()
    return ab.sqrt() * z0.float() + (1 - ab).sqrt() * torch.randn_like(z0.float())


def sample_t(n, p_grid, gen):
    grid = torch.tensor(GRID10)[torch.randint(0, 10, (n,), generator=gen)]
    uni = torch.randint(1, 1000, (n,), generator=gen)
    return torch.where(torch.rand(n, generator=gen) < p_grid, grid, uni)


# ------------------------------------------------------------------ data
def cmd_data(a):
    device = torch.device("cuda")
    pipe, _ = load_sd(device)
    lats, pids, temb = [], [], []
    for d in a.sources:
        for f in sorted(glob.glob(os.path.join(ROOT, "logs/phase1_data", d, "shard_*.npz"))):
            z = np.load(f)
            lats.append(z["x0_lat"][:, -1]); pids.append(z["prompt_ids"]); temb.append(z["text_emb"])
            print(f, z["x0_lat"].shape, "last saved step", int(z["save_steps"][-1]), flush=True)
    lat = np.concatenate(lats); pid = np.concatenate(pids); te = np.concatenate(temb)
    imgs = np.zeros((len(lat), 3, 224, 224), np.uint8)
    for b0 in range(0, len(lat), 32):
        x = torch.from_numpy(lat[b0:b0 + 32]).to(device)
        with torch.no_grad():
            im = pipe.vae.decode(x / pipe.vae.config.scaling_factor, return_dict=False)[0]
        im = ((im.float() + 1) / 2).clamp(0, 1)
        im = F.interpolate(im, size=(224, 224), mode="bicubic", align_corners=False).clamp(0, 1)  # 512 square -> 224
        imgs[b0:b0 + 32] = (im * 255).round().byte().cpu().numpy()
    os.makedirs(a.out, exist_ok=True)
    np.savez(os.path.join(a.out, "data.npz"), lat=lat, img=imgs, prompt_ids=pid, text_emb=te)
    print(f"saved {len(lat)} images ({len(np.unique(pid))} prompts) to {a.out}/data.npz", flush=True)


def load_data(a):
    z = np.load(os.path.join(a.out, "data.npz"))
    pid = z["prompt_ids"]
    up = np.unique(pid)
    rng = np.random.default_rng(0)
    val_p = set(rng.choice(up, size=int(round(len(up) * a.val_frac)), replace=False).tolist())
    val = np.array([p in val_p for p in pid])
    return {k: z[k] for k in z.files}, ~val, val


# ------------------------------------------------------------------ closed-form fit
@torch.no_grad()
def collect(pipe, null, clip, D, idx, heads, layers, p_grid, seed, device, bs=32):
    """Features of the requested heads (on the 16x16 grid) and CLIP hidden states entering the requested blocks."""
    gen = torch.Generator().manual_seed(seed)
    X = {h: [] for h in heads}; Y = {j: [] for j in layers}
    for b0 in range(0, len(idx), bs):
        ii = idx[b0:b0 + bs]
        z0 = torch.from_numpy(D["lat"][ii]).to(device)
        t = sample_t(len(ii), p_grid, gen).to(device)
        zt = noisy(pipe, z0, t)
        feats = unet_head_features(pipe.unet, zt, t, null, heads)
        for h in heads:
            X[h].append(to_grid(feats[h]).float().flatten(2).transpose(1, 2).cpu())   # (B,256,C)
        hs, _, _ = clip_tokens(clip, pixel_values(torch.from_numpy(D["img"][ii]), device).half(), layers)
        for j in layers:
            Y[j].append(hs[j].float().cpu())                                          # (B,257,1024)
    return {h: torch.cat(v) for h, v in X.items()}, {j: torch.cat(v) for j, v in Y.items()}


def fit_linear(X, Y):
    """X (N,256,C), Y (N,256,W) -> W (C,W) and per-position bias b (256,W) minimising ||X W + b - Y||^2 (centred lstsq)."""
    mx, my = X.mean(0), Y.mean(0)                                   # per-position means
    Xc = (X - mx).reshape(-1, X.shape[-1]).double().cuda()
    Yc = (Y - my).reshape(-1, Y.shape[-1]).double().cuda()
    Wm = torch.linalg.lstsq(Xc, Yc).solution                        # (C,W)
    b = my.double().cuda() - mx.double().cuda() @ Wm
    return Wm.float().cpu(), b.float().cpu()


def rel_mse(X, Y, Wm, b):
    pred = X @ Wm + b
    return float(((pred - Y) ** 2).sum() / ((Y - Y.mean(0)) ** 2).sum())


def cmd_search(a):
    from transformers import CLIPModel
    device = torch.device("cuda")
    pipe, null = load_sd(device)
    clip = CLIPModel.from_pretrained(a.clip, torch_dtype=torch.float16).to(device).eval()
    D, tr, _ = load_data(a)
    rng = np.random.default_rng(1)
    idx = rng.choice(np.where(tr)[0], size=a.n_probe + a.n_probe_test, replace=False)
    layers = list(range(0, a.max_j + 1))
    X, Y = collect(pipe, null, clip, D, idx, CANDIDATES, layers, 1.0, 1, device)
    n = a.n_probe
    res = {}
    for h in CANDIDATES:
        for j in layers:
            Yp = Y[j][:, 1:]                                         # patch tokens
            Wm, b = fit_linear(X[h][:n], Yp[:n])
            res[f"{h}|{j}"] = {"train": rel_mse(X[h][:n], Yp[:n], Wm, b), "test": rel_mse(X[h][n:], Yp[n:], Wm, b)}
            print(f"head {h:6s} ({CAND_CH[h]:4d} ch) -> CLIP block {j}: rel. MSE train {res[f'{h}|{j}']['train']:.3f} "
                  f"held-out {res[f'{h}|{j}']['test']:.3f}", flush=True)
    best = min(res, key=lambda k: res[k]["test"] if int(k.split("|")[1]) >= a.min_j else 1e9)
    tag = a.clip.replace("/", "_")
    with open(os.path.join(a.out, f"search_{tag}.json"), "w") as f:
        json.dump({"results": res, "best": best, "n_probe": n, "n_probe_test": a.n_probe_test, "clip": a.clip}, f, indent=2)
    print(f"best (j >= {a.min_j}): {best} {res[best]}", flush=True)


# ------------------------------------------------------------------ training
@torch.no_grad()
def evaluate(model, pipe, null, clip, D, idx, head, device, bs=64):
    """Per grid timestep: cosine(student, teacher) embedding and Spearman of the text scores (student vs teacher)."""
    from scipy.stats import spearmanr
    te = torch.from_numpy(D["text_emb"][idx]).float().to(device)
    te = F.normalize(te, dim=-1)
    out = {}
    for t_val in GRID10:
        cs, ss, st = [], [], []
        for b0 in range(0, len(idx), bs):
            ii = idx[b0:b0 + bs]
            z0 = torch.from_numpy(D["lat"][ii]).to(device)
            t = torch.full((len(ii),), t_val, device=device)
            torch.manual_seed(1000 + b0)
            zt = noisy(pipe, z0, t)
            feat = unet_head_features(pipe.unet, zt, t, null, [head])[head]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                es = model(feat)
            _, _, et = clip_tokens(clip, pixel_values(torch.from_numpy(D["img"][ii]), device).half(), [])
            cs.append((es * et).sum(-1)); ss.append((es * te[b0:b0 + bs]).sum(-1)); st.append((et * te[b0:b0 + bs]).sum(-1))
        cs, ss, st = torch.cat(cs).cpu().numpy(), torch.cat(ss).cpu().numpy(), torch.cat(st).cpu().numpy()
        out[t_val] = {"cos_teacher": float(cs.mean()), "spearman_text_score": float(spearmanr(ss, st)[0])}
    return out


def cmd_train(a):
    from transformers import CLIPModel
    device = torch.device("cuda")
    torch.manual_seed(a.seed)
    pipe, null = load_sd(device)
    teacher = CLIPModel.from_pretrained(a.clip, torch_dtype=torch.float16).to(device).eval()
    src = CLIPModel.from_pretrained(a.clip, torch_dtype=torch.float32)
    model = StitchVM(src, a.head, a.j, CAND_CH[a.head]).to(device)
    del src
    D, tr, va = load_data(a)
    tr_idx, va_idx = np.where(tr)[0], np.where(va)[0]
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)

    # stage-1 init: closed-form F + per-position bias, CLS = mean teacher CLS state entering block j
    rng = np.random.default_rng(2)
    probe = rng.choice(tr_idx, size=min(a.n_init, len(tr_idx)), replace=False)
    X, Y = collect(pipe, null, teacher, D, probe, [a.head], [a.j], a.p_grid, 2, device)
    Wm, b = fit_linear(X[a.head], Y[a.j][:, 1:])
    with torch.no_grad():
        model.F.weight.copy_(Wm.t().reshape(1024, -1, 1, 1)); model.F.bias.zero_()
        model.pos_bias.copy_(b); model.cls.copy_(Y[a.j][:, :1].mean(0, keepdim=True))
    init_mse = rel_mse(X[a.head], Y[a.j][:, 1:], Wm, b)
    print(f"init: head {a.head} -> block {a.j}, rel. MSE on {len(probe)} probe images {init_mse:.3f}", flush=True)
    del X, Y

    ev0 = evaluate(model, pipe, null, teacher, D, va_idx, a.head, device)
    print("val before finetuning:", {t: round(v["spearman_text_score"], 3) for t, v in ev0.items()}, flush=True)

    opt = torch.optim.AdamW([{"params": model.stitch_params(), "lr": a.lr * a.stitch_mult},
                             {"params": model.tail_params(), "lr": a.lr}], weight_decay=0.0, fused=True)
    steps_per_epoch = len(tr_idx) // a.batch
    total = steps_per_epoch * a.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / a.warmup))
    gen = torch.Generator().manual_seed(a.seed)
    log = {"init_rel_mse": init_mse, "val_before": ev0, "epochs": []}
    step, t0 = 0, time.time()
    for ep in range(a.epochs):
        perm = tr_idx[torch.randperm(len(tr_idx), generator=gen).numpy()]
        model.train(); losses = []
        for s in range(steps_per_epoch):
            ii = np.sort(perm[s * a.batch:(s + 1) * a.batch])
            z0 = torch.from_numpy(D["lat"][ii]).to(device)
            t = sample_t(len(ii), a.p_grid, gen).to(device)
            with torch.no_grad():
                feat = unet_head_features(pipe.unet, noisy(pipe, z0, t), t, null, [a.head])[a.head]
                _, tok_t, e_t = clip_tokens(teacher, pixel_values(torch.from_numpy(D["img"][ii]), device).half(), [])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e_s, tok_s = model(feat, return_tokens=True)
            loss_emb = ((e_s - e_t.float()) ** 2).sum(-1).mean()
            loss_tok = F.mse_loss(tok_s.float(), tok_t.float())
            loss = loss_emb + a.lambda_tok * loss_tok
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); step += 1
            losses.append([loss_emb.item(), loss_tok.item()])
        model.eval()
        ev = evaluate(model, pipe, null, teacher, D, va_idx, a.head, device)
        l = np.mean(losses, 0)
        log["epochs"].append({"epoch": ep + 1, "loss_emb": float(l[0]), "loss_tok": float(l[1]), "val": ev})
        print(f"epoch {ep + 1}/{a.epochs} ({time.time() - t0:.0f}s) loss emb {l[0]:.4f} tok {l[1]:.4f} | val Spearman "
              f"(student vs teacher text score) " + " ".join(f"t{t}:{v['spearman_text_score']:.2f}" for t, v in ev.items()),
              flush=True)
        model.save(out, {"clip": a.clip, "head": a.head, "j": a.j, "epoch": ep + 1, "args": vars(a)})
        with open(os.path.join(out, "train_log.json"), "w") as f:
            json.dump(log, f, indent=2)
    print(f"saved {out}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["data", "search", "train"])
    p.add_argument("--out", default=os.path.join(ROOT, "logs/stitchvm"))
    p.add_argument("--sources", nargs="+", default=["sd15_10steps_eta1", "sd15_50steps"])
    p.add_argument("--clip", default="asiimo/noisyclip")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--n-probe", type=int, default=200)
    p.add_argument("--n-probe-test", type=int, default=100)
    p.add_argument("--max-j", type=int, default=4)
    p.add_argument("--min-j", type=int, default=1)
    p.add_argument("--head", default="auto", choices=CANDIDATES + ["auto"], help="auto: best pair of `search` for --clip")
    p.add_argument("--j", type=int, default=-1, help="-1 with --head auto: from the search")
    p.add_argument("--name", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--stitch-mult", type=float, default=5.0)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--lambda-tok", type=float, default=1.0)
    p.add_argument("--p-grid", type=float, default=0.5)
    p.add_argument("--n-init", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    if a.cmd == "train" and a.head == "auto":
        with open(os.path.join(a.out, f"search_{a.clip.replace('/', '_')}.json")) as f:
            a.head, j = json.load(f)["best"].split("|")
            a.j = int(j)
    a.name = a.name or f"{a.clip.split('/')[-1]}_{a.head}_j{a.j}"
    {"data": cmd_data, "search": cmd_search, "train": cmd_train}[a.cmd](a)


if __name__ == "__main__":
    main()
