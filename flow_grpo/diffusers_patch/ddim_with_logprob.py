# Adapted from https://github.com/kvablack/ddpo-pytorch/blob/main/ddpo_pytorch/diffusers_patch/ddim_with_logprob.py
# Stochastic DDIM step (eta > 0) for epsilon-prediction models such as SD 1.5, returning the exact Gaussian
# log-prob of the transition. This plays the role that `sde_step_with_logprob` plays for flow-matching models.
# Return signature matches `sde_step_with_logprob`: (prev_sample, log_prob, prev_sample_mean, std_dev_t).

import math
from typing import Optional, Union

import torch
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.utils.torch_utils import randn_tensor


def _left_broadcast(t, shape):
    assert t.ndim <= len(shape)
    return t.reshape(t.shape + (1,) * (len(shape) - t.ndim)).broadcast_to(shape)


def _get_variance(self, timestep, prev_timestep):
    alpha_prod_t = torch.gather(self.alphas_cumprod, 0, timestep.cpu()).to(timestep.device)
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu().clamp(min=0)),
        self.final_alpha_cumprod,
    ).to(timestep.device)
    beta_prod_t = 1 - alpha_prod_t
    beta_prod_t_prev = 1 - alpha_prod_t_prev
    return (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)


def predict_x0(self: DDIMScheduler, model_output, timestep, sample):
    """x0 = (x_t - sqrt(1 - a_t) * eps) / sqrt(a_t) for epsilon prediction."""
    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.reshape(-1).long()
    if timestep.numel() == 1 and sample.shape[0] > 1:
        timestep = timestep.expand(sample.shape[0])
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu()).to(sample.device)
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.dtype)
    return (sample - (1 - alpha_prod_t) ** 0.5 * model_output) / alpha_prod_t ** 0.5


def ddim_step_with_logprob(
    self: DDIMScheduler,
    model_output: torch.FloatTensor,
    timestep: Union[int, torch.Tensor],
    sample: torch.FloatTensor,
    eta: float = 1.0,
    prev_sample: Optional[torch.FloatTensor] = None,
    generator: Optional[torch.Generator] = None,
):
    assert isinstance(self, DDIMScheduler)
    if self.num_inference_steps is None:
        raise ValueError("Run scheduler.set_timesteps(...) before stepping.")
    assert eta > 0, "eta must be > 0, otherwise the step is deterministic and has no log-prob."

    # bf16 can overflow here; do everything in fp32
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    if not torch.is_tensor(timestep):
        timestep = torch.tensor(timestep)
    timestep = timestep.reshape(-1).long().to(sample.device)
    if timestep.numel() == 1 and sample.shape[0] > 1:
        timestep = timestep.expand(sample.shape[0])

    # 1. previous timestep
    prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps
    prev_timestep = torch.clamp(prev_timestep, 0, self.config.num_train_timesteps - 1)

    # 2. alphas / betas
    alpha_prod_t = self.alphas_cumprod.gather(0, timestep.cpu())
    alpha_prod_t_prev = torch.where(
        prev_timestep.cpu() >= 0,
        self.alphas_cumprod.gather(0, prev_timestep.cpu()),
        self.final_alpha_cumprod,
    )
    alpha_prod_t = _left_broadcast(alpha_prod_t, sample.shape).to(sample.device)
    alpha_prod_t_prev = _left_broadcast(alpha_prod_t_prev, sample.shape).to(sample.device)
    beta_prod_t = 1 - alpha_prod_t

    # 3. predicted x0 (epsilon prediction)
    if self.config.prediction_type == "epsilon":
        pred_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        pred_epsilon = model_output
    elif self.config.prediction_type == "v_prediction":
        pred_original_sample = alpha_prod_t ** 0.5 * sample - beta_prod_t ** 0.5 * model_output
        pred_epsilon = alpha_prod_t ** 0.5 * model_output + beta_prod_t ** 0.5 * sample
    else:
        raise ValueError(f"Unsupported prediction_type {self.config.prediction_type}")

    if self.config.clip_sample:
        pred_original_sample = pred_original_sample.clamp(-self.config.clip_sample_range, self.config.clip_sample_range)

    # 4. sigma_t = eta * sqrt(variance)
    variance = _get_variance(self, timestep, prev_timestep)
    std_dev_t = eta * variance ** 0.5
    std_dev_t = _left_broadcast(std_dev_t, sample.shape).to(sample.device)

    # 5. mean of x_{t-1}
    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t ** 2) ** 0.5 * pred_epsilon
    prev_sample_mean = alpha_prod_t_prev ** 0.5 * pred_original_sample + pred_sample_direction

    if prev_sample is not None and generator is not None:
        raise ValueError("Cannot pass both generator and prev_sample.")

    if prev_sample is None:
        variance_noise = randn_tensor(
            model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
        )
        prev_sample = prev_sample_mean + std_dev_t * variance_noise

    # log prob of prev_sample under N(prev_sample_mean, std_dev_t^2), averaged over non-batch dims
    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (std_dev_t ** 2))
        - torch.log(std_dev_t)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))

    return prev_sample.type(sample.dtype), log_prob, prev_sample_mean, std_dev_t
