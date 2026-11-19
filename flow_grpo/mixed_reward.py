"""Mixed prompt-alignment + text-rendering reward for SD1.5 GRPO (Phase 3).

reward = align_weight * alignment + ocr_weight * OCR      (OCR only for prompts with a quoted target text)

alignment, two variants:
  "clip"  ORM: cos(CLIP ViT-L/14 image embedding of the FINAL decoded image, CLIP text embedding)
  "prm"   PRM: our noise-aware scorer (scripts/phase1_train_scorer.py, e.g. v2_lat_10steps_eta1) on the predicted
          clean latent x0_hat at a few denoising steps (no VAE decode), averaged over those steps = cumulative
          process reward (the slides' Phase 2/3 formulation).

alignment "noisyclip": NoisyCLIP (Ramos et al., ECCV 2026, HF asiimo/noisyclip; CLIP ViT-L/14 with a fine-tuned vision
          tower, trained on SDXL latents) on the latent at the PRM steps, rendered to RGB with NoisyCLIP's fixed linear
          approximation. noisyclip_input: "zt" (the noisy latent, as in the paper), "x0" (the predicted clean latent) or
          "x0phi" (x0 with our SD1.5-fitted Phi). Supervisor (2026-11-11): use NoisyCLIP, do not train CLIP ourselves.

Weight curriculum (supervisor's suggestion): weight_schedule = (ramp_epochs, align_end_frac[, ocr_start_frac]). Over the
first ramp_epochs the OCR weight rises linearly ocr_start_frac*ocr_weight -> ocr_weight (supervisor 2026-11-11: do not
start OCR at zero) and the alignment weight falls align_weight -> align_end_frac*align_weight
(alignment first, text later; alignment never goes to zero). With per-prompt normalisation only the ratio matters,
and only on prompts with a text target.

PRM query points: prm_fracs (fractions of the trajectory, e.g. 0.4, 0.6, 0.8, 1.0) are mapped to step indices for the
current number of denoising steps (needed with a steps curriculum); prm_steps (fixed indices) is used otherwise.

Advantages must be normalised per prompt (config.sample.global_std = False): OCR rewards are ~10x larger than the
alignment spread, so a global std would mute the alignment-only prompts.
"""
import os
import sys

import numpy as np
import torch

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)


def has_text_target(prompt):
    return prompt.count('"') >= 2


class MixedReward:
    def __init__(self, device, align="prm", prm_dir=None, phi_path=None, prm_steps=(3, 5, 7, 9),
                 align_weight=100.0, ocr_weight=80.0, clip_name="openai/clip-vit-large-patch14",
                 weight_schedule=None, prm_fracs=None, noisyclip_name="asiimo/noisyclip", noisyclip_input="zt",
                 ocr_workers=0):
        from phase1_noisyclip import Clip, project, nc_project  # noqa: E402
        import flow_grpo.rewards
        self.device, self.align = device, align
        self.align_weight, self.ocr_weight = align_weight, ocr_weight
        self.base_align, self.base_ocr = align_weight, ocr_weight
        self.weight_schedule = weight_schedule
        self.prm_fracs = prm_fracs
        self.prm_steps = set(prm_steps)
        self.clip = Clip(clip_name, device)
        # ocr_workers > 0: PaddleOCR in a pool of CPU processes (flow_grpo/ocr_pool.py), used through submit()
        self.ocr_pool = None
        if ocr_workers > 0:
            from flow_grpo.ocr_pool import ParallelOcr
            self.ocr_pool = ParallelOcr(ocr_workers)
            self.ocr = None
        else:
            self.ocr = flow_grpo.rewards.ocr_score(device)
        self.project = project
        self.nc_project, self.nc_input = nc_project, noisyclip_input
        if align == "noisyclip":
            self.nclip = Clip(noisyclip_name, device)
            if noisyclip_input == "x0phi":
                self.phi = torch.from_numpy(np.load(phi_path)).to(device)
        if align == "prm":
            from phase1_train_scorer import load_scorer
            self.scorer = load_scorer(prm_dir, clip_name, device)
            self.phi = torch.from_numpy(np.load(phi_path)).to(device)
        self._scores, self._text = None, None

    def set_epoch(self, epoch):
        if not self.weight_schedule:
            return
        ramp, end_frac = self.weight_schedule[:2]
        ocr_start = self.weight_schedule[2] if len(self.weight_schedule) > 2 else 0.0
        r = min(1.0, epoch / float(ramp))
        self.ocr_weight = self.base_ocr * (ocr_start + (1.0 - ocr_start) * r)
        self.align_weight = self.base_align * (1.0 - (1.0 - end_frac) * r)

    # ---------- PRM: called by the sampling pipeline at every step ----------
    def start(self, prompts, num_steps=None):
        """Call before sampling a batch; returns the x0 callback for pipeline_with_logprob."""
        if self.prm_fracs and num_steps:
            self.prm_steps = {min(num_steps - 1, max(0, int(round(f * num_steps)) - 1)) for f in self.prm_fracs}
        self._text = self.clip.text(list(prompts))
        self._scores = []
        if self.align not in ("prm", "noisyclip"):
            return None

        @torch.no_grad()
        def cb(i, t, x0, zt=None):
            if i not in self.prm_steps:
                return
            if self.align == "noisyclip":
                lat = zt if self.nc_input == "zt" else x0
                rgb = self.project(x0.float(), self.phi) if self.nc_input == "x0phi" else self.nc_project(lat.float())
                e = self.nclip.image(rgb)
                self._scores.append((e.float() * self._text).sum(-1))
                return
            tt = t.to(self.device).long().reshape(-1).expand(x0.shape[0])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                e = self.scorer(self.project(x0.float(), self.phi), tt, x0.float() if self.scorer.latent_input else None)
            self._scores.append((e.float() * self._text).sum(-1))
        return cb

    # ---------- after sampling ----------
    @torch.no_grad()
    def submit(self, images, prompts):
        """Computes the GPU parts now and starts the OCR; returns finish() -> the dict __call__ returns.
        With the OCR pool the OCR runs on CPU workers while the caller samples the next batch; without it the OCR runs
        inside finish(). The reward weights are those of the current epoch (captured here)."""
        B = len(prompts)
        if self.align in ("prm", "noisyclip"):
            assert len(self._scores) == len(self.prm_steps), "pipeline did not call the x0 callback"
            align = torch.stack(self._scores, 1).mean(1)
        else:
            align = (self.clip.image(images.float()) * self._text).sum(-1)
        # CLIP ORM score of the final image is always logged (it is the alignment metric) - cheap
        clip_final = (self.clip.image(images.float()) * self._text).sum(-1) if self.align != "clip" else align
        idx = [k for k in range(B) if has_text_target(prompts[k])]
        w_align, w_ocr = self.align_weight, self.ocr_weight
        futures = None
        if idx and self.ocr_pool is not None:
            arr = (images[idx] * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(0, 2, 3, 1)
            futures = self.ocr_pool.submit(list(arr), [prompts[k] for k in idx])
        images_text = images[idx] if (idx and futures is None) else None

        def finish():
            ocr = torch.full((B,), float("nan"), device=self.device)
            if idx:
                if futures is not None:
                    scores = [f.result() for f in futures]
                else:
                    scores, _ = self.ocr(images_text, [prompts[k] for k in idx], [{"target_text": prompts[k]} for k in idx])
                ocr[idx] = torch.as_tensor([float(x) for x in scores], device=self.device)
            total = w_align * align + w_ocr * torch.nan_to_num(ocr, nan=0.0)
            return {"avg": total, "align": align, "clip_final": clip_final, "ocr": ocr}
        return finish

    @torch.no_grad()
    def __call__(self, images, prompts):
        """images: (B,3,H,W) in [0,1] (final decoded images; needed for OCR and for the ORM variant)."""
        return self.submit(images, prompts)()
