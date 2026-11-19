"""StitchVM for SD1.5 + CLIP ViT-L/14 / NoisyCLIP (Go et al., "Stitched Value Model for Diffusion Alignment",
arXiv 2605.19804; 2026-11-18).

value(z_t, t) = CLIP_tail_{>=j}( stitch( UNet_head_{<=i}(z_t, t, null text) ) )  -> image embedding, scored by cosine with
the CLIP text embedding (NoisyCLIP's text tower = CLIP ViT-L/14's, unchanged).

- head: the frozen SD1.5 UNet truncated after a down block (candidates below), conditioned on the timestep and the empty
  prompt (as in the paper). In GRPO sampling these features exist already in the unconditional half of the CFG batch.
- stitch: per-token linear map F (1x1 conv, initialised by closed-form least squares) + bilinear resize to CLIP's 16x16
  patch grid + a zero-initialised residual MLP G (1x1 conv - SiLU - 1x1 conv, bottleneck 1:8) + a per-position bias
  (absorbs CLIP's positional embedding) + a learned [CLS] token.
- tail: CLIP vision blocks j..23 + post-layernorm + visual projection (initialised from NoisyCLIP or OpenAI CLIP),
  finetuned; the teacher is the same (frozen) CLIP on the clean final image (multi-level distillation, paper Eq. 78).

Head candidates (SD1.5, 64x64 latents): d0 32x32x320, d1pre 32x32x640, d1 16x16x640, d2pre 16x16x1280, d2 8x8x1280,
d3 8x8x1280, mid 8x8x1280 ("pre" = before the block's downsampler).
"""
import copy
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

CANDIDATES = ["d0", "d1pre", "d1", "d2pre", "d2", "d3", "mid"]
CAND_CH = {"d0": 320, "d1pre": 640, "d1": 640, "d2pre": 1280, "d2": 1280, "d3": 1280, "mid": 1280}
GRID = 16  # CLIP ViT-L/14 at 224 px


def _last_block_needed(names):
    order = {"d0": 0, "d1pre": 1, "d1": 1, "d2pre": 2, "d2": 2, "d3": 3, "mid": 4}
    return max(order[n] for n in names)


@torch.no_grad()
def unet_head_features(unet, z_t, t, null_ehs, names):
    """Runs the SD1.5 UNet only as far as needed and returns {name: (B,C,h,w)} for the requested head candidates.
    z_t: (B,4,64,64) UNet-space latents; t: (B,) or scalar timesteps; null_ehs: (1,77,768) empty-prompt embedding."""
    B = z_t.shape[0]
    dtype = unet.dtype
    t = torch.as_tensor(t, device=z_t.device).reshape(-1).expand(B)
    emb = unet.time_embedding(unet.time_proj(t).to(dtype))
    ehs = null_ehs.to(dtype).expand(B, -1, -1)
    h = unet.conv_in(z_t.to(dtype))
    out, last = {}, _last_block_needed(names)
    for k, blk in enumerate(unet.down_blocks):
        if getattr(blk, "has_cross_attention", False):
            h, res = blk(hidden_states=h, temb=emb, encoder_hidden_states=ehs)
        else:
            h, res = blk(hidden_states=h, temb=emb)
        tag = f"d{k}"
        if tag in names:
            out[tag] = h
        if f"{tag}pre" in names:
            out[f"{tag}pre"] = res[-2] if blk.downsamplers is not None else h
        if k >= last:
            break
    if "mid" in names:
        out["mid"] = unet.mid_block(h, emb, encoder_hidden_states=ehs)
    return out


def to_grid(feat):
    """(B,C,h,w) -> (B,C,16,16) bilinear (F and the resize commute: both linear)."""
    if feat.shape[-1] == GRID:
        return feat
    return F.interpolate(feat.float(), size=(GRID, GRID), mode="bilinear", align_corners=False)


class StitchVM(nn.Module):
    def __init__(self, clip_model, head, j, in_ch, width=1024, bottleneck=8):
        super().__init__()
        vm = clip_model.vision_model
        self.head, self.j = head, j
        self.F = nn.Conv2d(in_ch, width, 1)
        self.G = nn.Sequential(nn.SiLU(), nn.Conv2d(width, width // bottleneck, 1), nn.SiLU(),
                               nn.Conv2d(width // bottleneck, width, 1))
        nn.init.zeros_(self.G[-1].weight); nn.init.zeros_(self.G[-1].bias)
        self.pos_bias = nn.Parameter(torch.zeros(GRID * GRID, width))
        self.cls = nn.Parameter(torch.zeros(1, 1, width))
        self.layers = nn.ModuleList([copy.deepcopy(vm.encoder.layers[k]) for k in range(j, len(vm.encoder.layers))])
        self.post_layernorm = copy.deepcopy(vm.post_layernorm)
        self.visual_projection = copy.deepcopy(clip_model.visual_projection)

    def stitch_params(self):
        return [self.F.weight, self.F.bias, *self.G.parameters(), self.pos_bias, self.cls]

    def tail_params(self):
        return [*self.layers.parameters(), *self.post_layernorm.parameters(), *self.visual_projection.parameters()]

    def forward(self, feat, return_tokens=False):
        """feat: head feature (B,C,h,w). Returns L2-normalised image embedding (B,768) [and the last-layer tokens]."""
        x = self.F(to_grid(feat).to(self.F.weight.dtype))
        x = x + self.G(x)
        x = x.flatten(2).transpose(1, 2) + self.pos_bias              # (B,256,1024)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1)    # (B,257,1024)
        for layer in self.layers:
            x = layer(x, None, None)[0]
        e = self.visual_projection(self.post_layernorm(x[:, 0]))
        e = F.normalize(e.float(), dim=-1)
        return (e, x) if return_tokens else e

    def save(self, out, meta):
        os.makedirs(out, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(out, "stitchvm.pt"))
        with open(os.path.join(out, "stitchvm.json"), "w") as f:
            json.dump(meta, f, indent=2)


def clip_tokens(clip_model, pixel_values, layers):
    """Teacher: hidden states entering the given vision blocks (index j = input of block j) + last-layer tokens +
    normalised image embedding, for CLIP-preprocessed pixel_values (B,3,224,224)."""
    out = clip_model.vision_model(pixel_values=pixel_values, output_hidden_states=True)
    hs = out.hidden_states  # hs[0] = after pre_layernorm (input of block 0), hs[j] = input of block j
    e = F.normalize(clip_model.visual_projection(out.pooler_output).float(), dim=-1)
    return {j: hs[j] for j in layers}, out.last_hidden_state, e


def load_stitchvm(path, device, dtype=torch.float32):
    """Loads a trained StitchVM; returns (model, meta). The CLIP weights are rebuilt from meta['clip'] then overwritten."""
    from transformers import CLIPModel
    with open(os.path.join(path, "stitchvm.json")) as f:
        meta = json.load(f)
    clip = CLIPModel.from_pretrained(meta["clip"], torch_dtype=torch.float32)
    m = StitchVM(clip, meta["head"], meta["j"], CAND_CH[meta["head"]])
    m.load_state_dict(torch.load(os.path.join(path, "stitchvm.pt"), map_location="cpu"))
    del clip
    return m.to(device=device, dtype=dtype).eval(), meta
