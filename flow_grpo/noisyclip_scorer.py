import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel

# 1. Your Custom Adapter
class LatentCLIPAdapter(nn.Module):
    def __init__(self, latent_channels=16, clip_embed_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_channels, 256),
            nn.ReLU(),
            nn.Linear(256, clip_embed_dim)
        )

    def forward(self, latents):
        pooled_latents = latents.mean(dim=[-1, -2]) 
        return self.mlp(pooled_latents)

# 2. The Evaluator
class NoisyCLIPScorer(torch.nn.Module):
    def __init__(self, device="cuda", dtype=torch.float32):
        super().__init__()
        self.device = device
        self.dtype = dtype
        
        model_path = "openai/clip-vit-large-patch14" 
        self.processor = CLIPProcessor.from_pretrained(model_path)
        self.model = CLIPModel.from_pretrained(model_path).eval().to(device, dtype=dtype)
        
        self.adapter = LatentCLIPAdapter().to(device, dtype=dtype)
        self.adapter.train() 

    @torch.no_grad()
    def __call__(self, prompts, latents):
        # 1. Process Text
        text_inputs = self.processor(
            text=prompts, padding=True, truncation=True, max_length=77, return_tensors="pt"
        )
        text_inputs = {k: v.to(device=self.device) for k, v in text_inputs.items()}
        text_embs = self.model.get_text_features(**text_inputs)
        text_embs = text_embs / text_embs.norm(p=2, dim=-1, keepdim=True)
            
        # 2. Process Noisy Latents
        latent_embs = self.adapter(latents.to(self.dtype))
        latent_embs = latent_embs / latent_embs.norm(p=2, dim=-1, keepdim=True)
        
        # 3. Calculate Score
        logit_scale = self.model.logit_scale.exp()
        scores = logit_scale * (text_embs @ latent_embs.T)
        scores = scores.diag() / 100.0 
        
        return scores