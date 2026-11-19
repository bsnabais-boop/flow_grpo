"""Option 2 (2026-11-13): direct reward back-propagation for text rendering on SD1.5 (DRaFT / AlignProp style).

Why: RL (GRPO) on the PaddleOCR reward carries only ~1-2e-4 gradient signal per image (logs/diag_ocr_signal/
v3_variants), so a 100-epoch run adds up to ~2x the noise. Here the gradient comes from differentiating a text
recogniser instead of from sampling noise:

  10-step deterministic DDIM (eta 0, CFG 7.5) -> steps < k0 without grad, steps >= k0 with grad (each UNet call
  checkpointed) -> VAE decode (checkpointed) -> crop of the text region (box from PaddleOCR on the detached image;
  crop coordinates are constants, so the crop is differentiable w.r.t. the pixels; whole image if nothing is detected)
  -> TrOCR scene-text model, loss = NLL of the target text (uppercase, with/without spaces, the lower one).

PaddleOCR (det + rec) runs on the same images anyway for the box, so its score is logged as an independent check:
if TrOCR's loss drops while the PaddleOCR score does not rise, the model is fooling TrOCR (reward hacking).
Optional --clip-weight adds -cos(CLIP image, CLIP prompt) to keep prompt alignment.

Checkpoints: <save_dir>/checkpoints/checkpoint-N/lora (N = optimizer steps), loadable by scripts/phase3_eval.py --lora.
Log: <save_dir>/log.jsonl (one line per optimizer step).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="dataset/ocr_simple/train.txt")
    p.add_argument("--save-dir", default="logs/opt2/draft_v1")
    p.add_argument("--trocr", default="microsoft/trocr-base-str")
    p.add_argument("--crop", default="line", choices=["line", "box", "full"],
                   help="TrOCR input: best single PaddleOCR line (+ whole image), union box (fallback full), or full image")
    p.add_argument("--num-steps", type=int, default=10)
    p.add_argument("--k0", type=int, default=2, help="first denoising step with gradient (text forms around steps 2-5)")
    p.add_argument("--detach-input", action="store_true",
                   help="DRaFT-LV style: no gradient through the latent input of each UNet call, only through its weights")
    p.add_argument("--guidance-scale", type=float, default=7.5)
    p.add_argument("--batch-size", type=int, default=2, help="prompts per micro-batch (one image each)")
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=600, help="optimizer steps")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lora-rank", type=int, default=32)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--clip-weight", type=float, default=0.0, help="weight of -cos(CLIP image, CLIP prompt)")
    p.add_argument("--clip-text-weight", type=float, default=0.0,
                   help='weight of -cos(CLIP image, CLIP \'a photo with the text "X"\'); 50 ~ same scale as the TrOCR NLL '
                        '(std 2.2 vs 0.044 on base images, logs/opt2/proxy_check_v2)')
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-hours", type=float, default=11.5, help="stop (after saving) before the SLURM time limit")
    return p.parse_args()


def target_of(prompt):
    parts = prompt.split('"')
    return parts[1].strip() if len(parts) > 2 else ""


def paddle_score(rec_text, target):
    from Levenshtein import distance
    r = rec_text.replace(" ", "").lower(); t = target.replace(" ", "").lower()
    if not t:
        return 0.0
    d = 0 if t in r else min(distance(r, t), len(t))
    return 1 - d / len(t)


def union_box(lines, H, W, pad=0.15):
    if not lines:
        return None
    pts = np.concatenate([np.array(l[0], dtype=float) for l in lines], 0)
    x0, y0 = pts.min(0); x1, y1 = pts.max(0)
    px, py = pad * (x1 - x0) + 2, pad * (y1 - y0) + 2
    x0, x1 = int(max(0, x0 - px)), int(min(W, x1 + px))
    y0, y1 = int(max(0, y0 - py)), int(min(H, y1 + py))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return x0, y0, x1, y1


def main():
    args = parse()
    os.makedirs(os.path.join(args.save_dir, "checkpoints"), exist_ok=True)
    with open(os.path.join(args.save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    device = "cuda"
    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    from diffusers import DDIMScheduler, StableDiffusionPipeline
    from peft import LoraConfig, get_peft_model
    from transformers import CLIPModel, CLIPTokenizer, TrOCRProcessor, VisionEncoderDecoderModel
    from flow_grpo.ocr import OcrScorer

    prompts = [l.strip() for l in open(args.prompts) if l.strip() and l.count('"') >= 2]
    print(f"{len(prompts)} text prompts", flush=True)

    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", safety_checker=None,
                                                   requires_safety_checker=False)
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    for m in (pipe.vae, pipe.text_encoder, pipe.unet):
        m.requires_grad_(False)
    pipe.vae.to(device); pipe.text_encoder.to(device)
    lcfg = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank, init_lora_weights="gaussian",
                      target_modules=["to_k", "to_q", "to_v", "to_out.0"])
    unet = get_peft_model(pipe.unet, lcfg).to(device)
    unet.train()
    params = [p for p in unet.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = pipe.scheduler
    sched.set_timesteps(args.num_steps, device=device)
    timesteps = sched.timesteps
    vae_scale = pipe.vae.config.scaling_factor

    def embed(texts):
        ids = pipe.tokenizer(texts, padding="max_length", max_length=pipe.tokenizer.model_max_length, truncation=True,
                             return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            return pipe.text_encoder(ids)[0]

    neg = embed([""])

    proc = TrOCRProcessor.from_pretrained(args.trocr)
    trocr = VisionEncoderDecoderModel.from_pretrained(args.trocr).to(device).eval()
    trocr.requires_grad_(False)
    trocr.config.decoder_start_token_id = trocr.generation_config.decoder_start_token_id
    trocr.config.pad_token_id = trocr.generation_config.pad_token_id
    size = proc.image_processor.size
    S = size["height"] if isinstance(size, dict) else size
    t_mean = torch.tensor(proc.image_processor.image_mean, device=device).view(1, 3, 1, 1)
    t_std = torch.tensor(proc.image_processor.image_std, device=device).view(1, 3, 1, 1)
    ocr = OcrScorer().ocr

    clip = clip_tok = None
    if args.clip_weight > 0 or args.clip_text_weight > 0:
        clip = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device).eval(); clip.requires_grad_(False)
        clip_tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        c_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        c_std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)

    def unet_eps(lat, t, cond):
        inp = torch.cat([lat] * 2)
        inp = sched.scale_model_input(inp, t)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = unet(inp, t, encoder_hidden_states=cond, return_dict=False)[0]
        u, c = out.float().chunk(2)
        return u + args.guidance_scale * (c - u)

    def decode(lat):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            x = pipe.vae.decode(lat / vae_scale, return_dict=False)[0]
        return ((x.float() + 1) / 2).clamp(0, 1)

    def generate(batch_prompts):
        B = len(batch_prompts)
        cond = torch.cat([neg.repeat(B, 1, 1), embed(batch_prompts)])
        lat = torch.randn(B, 4, 64, 64, device=device)
        for i, t in enumerate(timesteps):
            if i < args.k0:
                with torch.no_grad():
                    eps = unet_eps(lat, t, cond)
            else:
                x_in = lat.detach() if args.detach_input else lat
                eps = checkpoint(unet_eps, x_in, t, cond, use_reentrant=False)
            lat = sched.step(eps, t, lat, eta=0.0, return_dict=False)[0]
        return checkpoint(decode, lat, use_reentrant=False)

    def trocr_nll(crop, target):
        x = F.interpolate(crop, size=(S, S), mode="bilinear", align_corners=False)
        x = (x - t_mean) / t_std
        best = None
        for t in {target.upper(), target.upper().replace(" ", "")}:
            lab = torch.tensor([proc.tokenizer(t).input_ids[1:]], device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = trocr(pixel_values=x, labels=lab).loss.float()
            best = loss if best is None or loss.item() < best.item() else best
        return best

    log = open(os.path.join(args.save_dir, "log.jsonl"), "a")
    t_start = time.time()
    step = 0
    while step < args.max_steps:
        t_step = time.time()
        stats = {"nll": [], "paddle": [], "exact": [], "detected": [], "clip": [], "clip_text": [], "n_lines": []}
        for _ in range(args.grad_accum):
            batch = random.sample(prompts, args.batch_size)
            imgs = generate(batch)
            loss = 0.0
            for b, p in enumerate(batch):
                tgt = target_of(p)
                arr = (imgs[b].detach().permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                res = ocr.ocr(arr, cls=False)
                lines = res[0] if res and res[0] else []
                txt = "".join(l[1][0] for l in lines if l and len(l) > 1 and l[1] and l[1][1] > 0)
                ps = paddle_score(txt, tgt)
                if args.crop == "line":
                    # each detected text line separately + the whole image; the lowest NLL is the loss
                    lbs = [x for x in (union_box([l], arr.shape[0], arr.shape[1]) for l in lines) if x is not None]
                    crops = [imgs[b:b + 1, :, y0:y1, x0:x1] for x0, y0, x1, y1 in lbs] + [imgs[b:b + 1]]
                    nlls = [trocr_nll(c, tgt) for c in crops]
                    nll = min(nlls, key=lambda v: v.item())
                    box = lbs[0] if lbs else None
                    stats["n_lines"].append(len(lbs))
                else:
                    box = union_box(lines, arr.shape[0], arr.shape[1]) if args.crop == "box" else None
                    crop = imgs[b:b + 1] if box is None else imgs[b:b + 1, :, box[1]:box[3], box[0]:box[2]]
                    nll = trocr_nll(crop, tgt)
                loss = loss + nll
                stats["nll"].append(nll.item()); stats["paddle"].append(ps); stats["exact"].append(float(ps >= 1))
                stats["detected"].append(float(box is not None))
            if clip is not None:
                x = F.interpolate(imgs, size=(224, 224), mode="bicubic", align_corners=False)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    ie = F.normalize(clip.get_image_features(pixel_values=(x - c_mean) / c_std).float(), dim=-1)
                for w, texts, key in [(args.clip_weight, batch, "clip"),
                                      (args.clip_text_weight, [f'a photo with the text "{target_of(p)}"' for p in batch], "clip_text")]:
                    if w <= 0:
                        continue
                    tt = clip_tok(texts, padding=True, truncation=True, return_tensors="pt").to(device)
                    with torch.no_grad():
                        te = F.normalize(clip.get_text_features(**tt).float(), dim=-1)
                    cs = (ie * te).sum(-1)
                    loss = loss - w * cs.sum()
                    stats[key] += cs.tolist()
            (loss / (args.batch_size * args.grad_accum)).backward()
        gn = torch.nn.utils.clip_grad_norm_(params, args.max_grad_norm)
        opt.step(); opt.zero_grad(set_to_none=True)
        step += 1
        rec = {"step": step, "grad_norm": float(gn), "sec": time.time() - t_step,
               "max_mem_gb": torch.cuda.max_memory_allocated() / 2 ** 30,
               **{k: float(np.mean(v)) for k, v in stats.items() if v}}
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(json.dumps(rec), flush=True)
        out_of_time = (time.time() - t_start) / 3600 > args.max_hours
        if step % args.save_every == 0 or step == args.max_steps or out_of_time:
            d = os.path.join(args.save_dir, "checkpoints", f"checkpoint-{step}", "lora")
            unet.save_pretrained(d)
            print(f"saved {d}", flush=True)
        if out_of_time:
            print("time limit reached, stopping", flush=True)
            break


if __name__ == "__main__":
    main()
