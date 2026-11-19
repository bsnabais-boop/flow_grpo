# SD 1.5 counterpart of sd3_pipeline_with_logprob_fast.py.
# - Uses a stochastic DDIM step (ddim_with_logprob) inside the SDE window and deterministic DDIM (eta=0) outside it.
# - Returns the intermediate latents / log-probs / timesteps inside the window, plus per-step rewards computed on
#   the predicted x0 at each window step (same contract as the SD3 fast pipeline).
from typing import Dict, List, Optional, Union
import random

import torch

from .ddim_with_logprob import ddim_step_with_logprob, predict_x0


def _decode(pipe, latents):
    latents = latents / pipe.vae.config.scaling_factor
    shift = getattr(pipe.vae.config, "shift_factor", None)
    if shift is not None:
        latents = latents + shift
    image = pipe.vae.decode(latents.to(dtype=pipe.vae.dtype), return_dict=False)[0]
    return image


@torch.no_grad()
def pipeline_with_logprob(
    self,
    prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 10,
    guidance_scale: float = 7.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    output_type: Optional[str] = "pt",
    noise_level: float = 1.0,  # DDIM eta inside the SDE window
    sde_window_size: int = 0,
    sde_window_range: tuple = (0, 5),
    reward_fn: Optional[callable] = None,
    prompt_metadata: Optional[List[Dict]] = None,
    sde_generator: Optional[torch.Generator] = None,  # seeds the SDE-window noise (eval); None = global RNG (training)
    x0_callback: Optional[callable] = None,  # x0_callback(i, t, x0_hat, z_t) at every step (process reward model, no VAE)
):
    height = height or self.unet.config.sample_size * self.vae_scale_factor
    width = width or self.unet.config.sample_size * self.vae_scale_factor
    device = self._execution_device
    batch_size = prompt_embeds.shape[0]
    do_cfg = guidance_scale > 1.0

    if do_cfg:
        cond = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
    else:
        cond = prompt_embeds

    # latents
    num_channels_latents = self.unet.config.in_channels
    latents = self.prepare_latents(
        batch_size, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, latents
    ).float()

    # timesteps
    self.scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = self.scheduler.timesteps

    if sde_window_size > 0:
        start = random.randint(sde_window_range[0], sde_window_range[1] - sde_window_size)
        sde_window = (start, start + sde_window_size)
    else:
        sde_window = (0, len(timesteps) - 1)

    all_latents, all_log_probs, all_timesteps, all_step_rewards = [], [], [], []

    with self.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            in_window = sde_window[0] <= i < sde_window[1]
            if i == sde_window[0]:
                all_latents.append(latents)

            latent_model_input = torch.cat([latents] * 2) if do_cfg else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(
                latent_model_input.to(self.unet.dtype),
                t,
                encoder_hidden_states=cond,
                return_dict=False,
            )[0].float()
            if do_cfg:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            if x0_callback is not None:
                x0_callback(i, t, predict_x0(self.scheduler, noise_pred, t, latents), latents)

            if in_window:
                prev_latents = latents
                latents, log_prob, _, _ = ddim_step_with_logprob(
                    self.scheduler, noise_pred, t, latents, eta=noise_level, generator=sde_generator
                )
                all_latents.append(latents)
                all_log_probs.append(log_prob)
                all_timesteps.append(t)

                if reward_fn is not None:
                    x0_pred = predict_x0(self.scheduler, noise_pred, t, prev_latents)
                    intermediate_img = _decode(self, x0_pred)
                    intermediate_img = self.image_processor.postprocess(intermediate_img, output_type="pt")
                    step_rewards_dict, _ = reward_fn(intermediate_img, prompt, prompt_metadata, only_strict=True)
                    all_step_rewards.append(step_rewards_dict["avg"])
            else:
                latents = self.scheduler.step(noise_pred, t, latents, eta=0.0, return_dict=False)[0].float()

            progress_bar.update()

    image = _decode(self, latents)
    image = self.image_processor.postprocess(image, output_type=output_type)
    self.maybe_free_model_hooks()

    return image, all_latents, all_log_probs, all_timesteps, all_step_rewards
