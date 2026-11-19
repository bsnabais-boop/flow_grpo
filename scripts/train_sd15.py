from collections import defaultdict
import contextlib
import os
import datetime
from concurrent import futures
import time
import json
import hashlib
from absl import app, flags
from accelerate import Accelerator
from ml_collections import config_flags
from accelerate.utils import set_seed, ProjectConfiguration
from accelerate.logging import get_logger
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils.torch_utils import is_compiled_module
import numpy as np
import flow_grpo.prompts
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.sd15_pipeline_with_logprob import pipeline_with_logprob
from flow_grpo.diffusers_patch.ddim_with_logprob import ddim_step_with_logprob
import torch
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = get_logger(__name__)

class TextPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}.txt')
        with open(self.file_path, 'r') as f:
            self.prompts = [line.strip() for line in f.readlines()]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": {}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split='train'):
        self.file_path = os.path.join(dataset, f'{split}_metadata.jsonl')
        with open(self.file_path, 'r', encoding='utf-8') as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item['prompt'] for item in self.metadatas]
        
    def __len__(self):
        return len(self.prompts)
    
    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size  # Batch size per replica
        self.k = k                    # Number of repetitions per sample
        self.num_replicas = num_replicas  # Total number of replicas
        self.rank = rank              # Current replica rank
        self.seed = seed              # Random seed for synchronization
        
        # Compute the number of unique samples needed per iteration
        self.total_samples = self.num_replicas * self.batch_size
        assert self.total_samples % self.k == 0, f"k can not divide n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k  # Number of unique samples
        self.epoch = 0

    def __iter__(self):
        while True:
            # Generate a deterministic random sequence to ensure all replicas are synchronized
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Randomly select m unique samples
            indices = torch.randperm(len(self.dataset), generator=g)[:self.m].tolist()
            
            # Repeat each sample k times to generate n*b total samples
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            
            # Shuffle to ensure uniform distribution
            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]
            
            # Split samples to each replica
            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            
            # Return current replica's sample indices
            yield per_card_samples[self.rank]
    
    def set_epoch(self, epoch):
        self.epoch = epoch  # Used to synchronize random state across epochs


def compute_text_embeddings(prompt, text_encoder, tokenizer, device):
    # SD 1.5: a single CLIP text encoder, max 77 tokens (longer prompts are truncated).
    with torch.no_grad():
        ids = tokenizer(
            prompt,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(device)
        prompt_embeds = text_encoder(ids)[0]
    return prompt_embeds

def calculate_zero_std_ratio(prompts, gathered_rewards):
    """
    Calculate the proportion of unique prompts whose reward standard deviation is zero.
    
    Args:
        prompts: List of prompts.
        gathered_rewards: Dictionary containing rewards, must include the key 'ori_avg'.
        
    Returns:
        zero_std_ratio: Proportion of prompts with zero standard deviation.
        prompt_std_devs: Mean standard deviation across all unique prompts.
    """
    # Convert prompt list to NumPy array
    prompt_array = np.array(prompts)
    
    # Get unique prompts and their group information
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, 
        return_inverse=True,
        return_counts=True
    )
    
    # Group rewards for each prompt
    grouped_rewards = gathered_rewards['ori_avg'][np.argsort(inverse_indices)]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    
    # Calculate standard deviation for each group
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    
    # Calculate the ratio of zero standard deviation
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    
    return zero_std_ratio, prompt_std_devs.mean()

def create_generator(prompts, base_seed):
    generators = []
    for prompt in prompts:
        # Use a stable hash (SHA256), then convert it to an integer seed
        hash_digest = hashlib.sha256(prompt.encode()).digest()
        prompt_hash_int = int.from_bytes(hash_digest[:4], 'big')  # Take the first 4 bytes as part of the seed
        seed = (base_seed + prompt_hash_int) % (2**31) # Ensure the number is within a valid range
        gen = torch.Generator().manual_seed(seed)
        generators.append(gen)
    return generators

        
def steps_at(config, epoch):
    """Denoising steps used for sampling/training at this epoch. config.sample.steps_schedule: list of
    (start_epoch, num_steps), e.g. [(0, 4), (20, 6), (40, 8), (60, 10)]; None = fixed config.sample.num_steps."""
    sched = config.sample.get("steps_schedule", None)
    if not sched:
        return config.sample.num_steps
    n = sched[0][1]
    for start, k in sched:
        if epoch >= start:
            n = k
    return int(n)


def compute_log_prob(unet, pipeline, sample, j, embeds, config):
    if config.train.cfg:
        noise_pred = unet(
            torch.cat([sample["latents"][:, j]] * 2),
            torch.cat([sample["timesteps"][:, j]] * 2),
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = (
            noise_pred_uncond
            + config.sample.guidance_scale
            * (noise_pred_text - noise_pred_uncond)
        )
    else:
        noise_pred = unet(
            sample["latents"][:, j],
            sample["timesteps"][:, j],
            encoder_hidden_states=embeds,
            return_dict=False,
        )[0]

    # compute the log prob of next_latents given latents under the current model
    prev_sample, log_prob, prev_sample_mean, std_dev_t = ddim_step_with_logprob(
        pipeline.scheduler,
        noise_pred.float(),
        sample["timesteps"][:, j],
        sample["latents"][:, j].float(),
        eta=config.sample.noise_level,
        prev_sample=sample["next_latents"][:, j].float(),
    )

    return prev_sample, log_prob, prev_sample_mean, std_dev_t

def eval(pipeline, test_dataloader, text_encoder, tokenizer, config, accelerator, global_step, reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters):
    if config.train.ema:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
    neg_prompt_embed = compute_text_embeddings([""], text_encoder, tokenizer, device=accelerator.device)
    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.test_batch_size, 1, 1)

    # test_dataloader = itertools.islice(test_dataloader, 2)
    all_rewards = defaultdict(list)
    # Fixed-seed eval: the same initial noise at every eval (so checkpoints are compared on identical samples)
    # and eval_images_per_prompt images per prompt (to average out per-sample OCR noise).
    eval_k = config.sample.get("eval_images_per_prompt", 1)
    eval_seed = config.sample.get("eval_seed", None)
    # "standard": diffusers pipeline with eval_num_steps / eval_guidance_scale (deterministic DDIM).
    # "train": the training sampler (see below), so eval tracks what GRPO actually optimises.
    eval_sampler = config.sample.get("eval_sampler", "standard")
    assert eval_sampler in ("standard", "train"), eval_sampler
    for batch_idx, test_batch in enumerate(tqdm(
            test_dataloader,
            desc="Eval: ",
            disable=not accelerator.is_local_main_process,
            position=0,
        )):
        prompts, prompt_metadata = test_batch
        prompt_embeds = compute_text_embeddings(prompts, text_encoder, tokenizer, device=accelerator.device)
        # The last batch may not be full batch_size
        if len(prompt_embeds)<len(sample_neg_prompt_embeds):
            sample_neg_prompt_embeds = sample_neg_prompt_embeds[:len(prompt_embeds)]
        images = []
        for r in range(eval_k):
            generator = sde_generator = None
            seed = None
            if eval_seed is not None:
                seed = eval_seed + 100003 * accelerator.process_index + 1009 * batch_idx + r
                generator = torch.Generator(device=accelerator.device).manual_seed(seed)
                sde_generator = torch.Generator(device=accelerator.device).manual_seed(seed + 7)
            if eval_sampler == "train":
                # Same sampler as training: num_steps DDIM steps, stochastic (eta=noise_level) inside a random
                # SDE window, deterministic elsewhere. The window choice is seeded too, without disturbing the
                # global `random` state that training uses.
                rng_state = random.getstate()
                if seed is not None:
                    random.seed(seed)
                with autocast():
                    with torch.no_grad():
                        out = pipeline_with_logprob(
                            pipeline,
                            prompt=prompts,
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=sample_neg_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            generator=generator,
                            sde_window_size=config.sample.sde_window_size,
                            sde_window_range=config.sample.sde_window_range,
                            reward_fn=None,
                            sde_generator=sde_generator,
                        )
                random.setstate(rng_state)
                images.append(out[0])
                continue
            with autocast():
                with torch.no_grad():
                    images.append(pipeline(
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds,
                        num_inference_steps=config.sample.eval_num_steps,
                        guidance_scale=config.sample.eval_guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution,
                        generator=generator,
                    ).images)
        images = torch.cat(images, dim=0)
        prompts = list(prompts) * eval_k  # images are ordered [repeat 0: all prompts, repeat 1: all prompts, ...]
        # the OCR reward reads the target text from metadata["target_text"]
        prompt_metadata = [{"target_text": p} for p in prompts]
        rewards = executor.submit(reward_fn, images, prompts, prompt_metadata, only_strict=False)
        # yield to to make sure reward computation starts
        time.sleep(0)
        rewards, reward_metadata = rewards.result()

        print("AVALIAÇÃO:", rewards.keys())

        for key, value in rewards.items():
            rewards_gather = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()
            all_rewards[key].append(rewards_gather)
    
    last_batch_images_gather = accelerator.gather(torch.as_tensor(images, device=accelerator.device)).cpu().numpy()
    last_batch_prompt_ids = tokenizer(
        prompts,
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    ).input_ids.to(accelerator.device)
    last_batch_prompt_ids_gather = accelerator.gather(last_batch_prompt_ids).cpu().numpy()
    last_batch_prompts_gather = pipeline.tokenizer.batch_decode(
        last_batch_prompt_ids_gather, skip_special_tokens=True
    )
    last_batch_rewards_gather = {}
    for key, value in rewards.items():
        last_batch_rewards_gather[key] = accelerator.gather(torch.as_tensor(value, device=accelerator.device)).cpu().numpy()

    all_rewards = {key: np.concatenate(value) for key, value in all_rewards.items()}
    if accelerator.is_main_process:
        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples = min(15, len(last_batch_images_gather))
            # sample_indices = random.sample(range(len(images)), num_samples)
            sample_indices = range(num_samples)
            for idx, index in enumerate(sample_indices):
                image = last_batch_images_gather[index]
                pil = Image.fromarray(
                    (image.transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))
            sampled_prompts = [last_batch_prompts_gather[index] for index in sample_indices]
            sampled_rewards = [{k: last_batch_rewards_gather[k][i].item() for k in last_batch_rewards_gather} for i in sample_indices]
            for key, value in all_rewards.items():
                print(key, value.shape)
            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | " + " | ".join(f"{k}: {v:.2f}" for k, v in reward.items() if v != -10),
                        )
                        for idx, (prompt, reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                    ],
                    **{f"eval_reward_{key}": np.mean(value[value != -10]) for key, value in all_rewards.items()},
                },
                step=global_step,
            )
    if config.train.ema:
        ema.copy_temp_to(transformer_trainable_parameters)

def unwrap_model(model, accelerator):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model

def save_ckpt(save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config):
    save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
    save_root_lora = os.path.join(save_root, "lora")
    os.makedirs(save_root_lora, exist_ok=True)
    if accelerator.is_main_process:
        if config.train.ema:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)
        unwrap_model(transformer, accelerator).save_pretrained(save_root_lora)
        if config.train.ema:
            ema.copy_temp_to(transformer_trainable_parameters)

def main(_):
    # basic Accelerate and logging setup
    config = FLAGS.config

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # number of timesteps within each trajectory to train on
    if config.sample.sde_window_size > 0:
        num_train_timesteps = config.sample.sde_window_size
    else:
        num_train_timesteps = config.sample.num_steps - 1

    # "intermediate" (old behaviour): rewards from x0 previews inside the SDE window.
    # "final": reward of the finished image, shared by every window step.
    reward_mode = config.sample.get("reward_mode", "intermediate")
    # With a denoising-steps curriculum the number of trained transitions changes between epochs, so accelerate's
    # fixed gradient_accumulation_steps (= samples x transitions) no longer fits: accumulate manually instead.
    # config.train.skip_last = k leaves the last k denoising transitions out of the update (2026-11-11 diagnostic: 57% of
    # the policy gradient came from the last transition t=101->1, where the tiny sigma inflates log-prob gradients but
    # the text is already fixed). The number of trained transitions then differs from num_train_timesteps -> manual
    # accumulation as well.
    skip_last = int(config.train.get("skip_last", 0))
    manual_accum = bool(config.sample.get("steps_schedule", None)) or skip_last > 0
    if config.sample.get("steps_schedule", None):
        assert config.sample.sde_window_size == 0, "steps_schedule assumes stochastic DDIM at every step"
    # "mixed": prompt alignment (CLIP ORM or our noise-aware PRM) for every prompt + OCR for prompts with a quoted
    # target text (flow_grpo/mixed_reward.py). Uses per-prompt advantage normalisation.
    assert reward_mode in ("intermediate", "final", "mixed"), reward_mode

    accelerator_config = ProjectConfiguration(
        project_dir=os.path.join(config.logdir, config.run_name),
        automatic_checkpoint_naming=True,
        total_limit=config.num_checkpoint_limit,
    )

    accelerator = Accelerator(
        # log_with="wandb",
        mixed_precision=config.mixed_precision,
        project_config=accelerator_config,
        # we always accumulate gradients across timesteps; we want config.train.gradient_accumulation_steps to be the
        # number of *samples* we accumulate across, so we need to multiply by the number of training timesteps to get
        # the total number of optimizer steps to accumulate across.
        gradient_accumulation_steps=1 if manual_accum else config.train.gradient_accumulation_steps * num_train_timesteps,
    )
    if accelerator.is_main_process:
        wandb.init(
            project="flow_grpo",
        )
        # accelerator.init_trackers(
        #     project_name="flow-grpo",
        #     config=config.to_dict(),
        #     init_kwargs={"wandb": {"name": config.run_name}},
        # )
    logger.info(f"\n{config}")

    # set seed (device_specific is very important to get different prompts on different devices)
    set_seed(config.seed, device_specific=True)

    # load scheduler, tokenizer and models.
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.pretrained.model, safety_checker=None, requires_safety_checker=False
    )
    # DDIM is required for the stochastic step + exact log-prob
    pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
    # freeze parameters of models to save more memory
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.unet.requires_grad_(not config.use_lora)

    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    # make the progress bar nicer
    pipeline.set_progress_bar_config(
        position=1,
        disable=not accelerator.is_local_main_process,
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora transformer) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    inference_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        inference_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        inference_dtype = torch.bfloat16

    # Move vae and text_encoder to device and cast to inference_dtype
    pipeline.vae.to(accelerator.device, dtype=torch.float32)
    
    if getattr(config, "enable_vae_slicing", False):
        pipeline.vae.enable_slicing()

    # SD1.5's CLIP encoder is small; keep it in fp32 so its embeddings match the fp32 UNet (LoRA, no autocast).
    pipeline.text_encoder.to(accelerator.device, dtype=torch.float32)

    pipeline.unet.to(accelerator.device)

    if config.use_lora:
        # Set correct lora layers
        # UNet self- and cross-attention projections
        target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
        transformer_lora_config = LoraConfig(
            r=config.train.get("lora_rank", 32),
            lora_alpha=2 * config.train.get("lora_rank", 32),
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            pipeline.unet = PeftModel.from_pretrained(pipeline.unet, config.train.lora_path, is_trainable=True)
            pipeline.unet.set_adapter("default")
        else:
            pipeline.unet = get_peft_model(pipeline.unet, transformer_lora_config)

    if config.activation_checkpointing:
        pipeline.unet.enable_gradient_checkpointing()

    # keep the name `transformer` below so the GRPO loop stays identical to train_sd3_fast.py
    transformer = pipeline.unet
    transformer_trainable_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))
    # This ema setting affects the previous 20 × 8 = 160 steps on average.
    ema = EMAModuleWrapper(transformer_trainable_parameters, decay=0.9, update_step_interval=8, device=accelerator.device)
    
    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    if config.train.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # prepare prompt and reward fn
    reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    eval_reward_fn = getattr(flow_grpo.rewards, 'multi_score')(accelerator.device, config.reward_fn)
    mixed_reward = None
    if reward_mode == "mixed":
        from flow_grpo.mixed_reward import MixedReward
        mixed_reward = MixedReward(accelerator.device, align=config.sample.align_reward,
                                   prm_dir=config.sample.get("prm_dir", None), phi_path=config.sample.get("prm_phi", None),
                                   prm_steps=tuple(config.sample.get("prm_steps", (3, 5, 7, 9))),
                                   align_weight=config.sample.get("align_weight", 100.0),
                                   ocr_weight=config.sample.get("ocr_weight", 80.0),
                                   weight_schedule=config.sample.get("weight_schedule", None),
                                   prm_fracs=config.sample.get("prm_fracs", None),
                                   noisyclip_name=config.sample.get("noisyclip_name", "asiimo/noisyclip"),
                                   noisyclip_input=config.sample.get("noisyclip_input", "zt"),
                                   ocr_workers=int(config.sample.get("ocr_workers", 0)))

    if config.prompt_fn == "general_ocr":
        train_dataset = TextPromptDataset(config.dataset, 'train')
        test_dataset = TextPromptDataset(config.dataset, 'test')

        # Create an infinite-loop DataLoader
        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=config.seed  # was hard-coded 42 (= default config.seed); now follows the run seed
        )

        # Create a DataLoader; note that shuffling is not needed here because it’s controlled by the Sampler.
        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=TextPromptDataset.collate_fn,
            # persistent_workers=True
        )

        # Create a regular DataLoader
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=TextPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, 'train')
        test_dataset = GenevalPromptDataset(config.dataset, 'test')

        train_sampler = DistributedKRepeatSampler( 
            dataset=train_dataset,
            batch_size=config.sample.train_batch_size,
            k=config.sample.num_image_per_prompt,
            num_replicas=accelerator.num_processes,
            rank=accelerator.process_index,
            seed=config.seed  # was hard-coded 42 (= default config.seed); now follows the run seed
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            num_workers=1,
            collate_fn=GenevalPromptDataset.collate_fn,
            # persistent_workers=True
        )
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=config.sample.test_batch_size,
            collate_fn=GenevalPromptDataset.collate_fn,
            shuffle=False,
            num_workers=8,
        )
    else:
        raise NotImplementedError("Only general_ocr is supported with dataset")


    neg_prompt_embed = compute_text_embeddings([""], text_encoder, tokenizer, device=accelerator.device)

    sample_neg_prompt_embeds = neg_prompt_embed.repeat(config.sample.train_batch_size, 1, 1)
    train_neg_prompt_embeds = neg_prompt_embed.repeat(config.train.batch_size, 1, 1)

    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    # initialize stat tracker
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(config.sample.global_std)

    # for some reason, autocast is necessary for non-lora training but for lora training it isn't necessary and it uses
    # more memory
    autocast = contextlib.nullcontext if config.use_lora else accelerator.autocast
    # config.lora_autocast (2026-11-18, default off = old behaviour): with LoRA the UNet ran in fp32 (+TF32 matmuls);
    # True uses bf16 autocast for sampling AND training (same context in both -> consistent log-probs).
    if config.use_lora and config.get("lora_autocast", False):
        autocast = accelerator.autocast
    # autocast = accelerator.autocast

    # Prepare everything with our `accelerator`.
    transformer, optimizer, train_dataloader, test_dataloader = accelerator.prepare(transformer, optimizer, train_dataloader, test_dataloader)

    # executor to perform callbacks asynchronously. this is beneficial for the llava callbacks which makes a request to a
    # remote server running llava inference.
    executor = futures.ThreadPoolExecutor(max_workers=8)

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * accelerator.num_processes
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size
        * accelerator.num_processes
        * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")
    # assert config.sample.train_batch_size >= config.train.batch_size
    # assert config.sample.train_batch_size % config.train.batch_size == 0
    # assert samples_per_epoch % total_train_batch_size == 0

    train_chunks = int(config.train.get("micro_batches", 0)) or config.sample.num_batches_per_epoch
    epoch = 0
    global_step = 0
    train_iter = iter(train_dataloader)

    while True:
        #################### EVAL ####################
        pipeline.unet.eval()
        if config.eval_freq > 0 and epoch % config.eval_freq == 0:
            eval(pipeline, test_dataloader, text_encoder, tokenizer, config, accelerator, global_step, eval_reward_fn, executor, autocast, num_train_timesteps, ema, transformer_trainable_parameters)
        if epoch % config.save_freq == 0 and epoch > 0 and accelerator.is_main_process:
            save_ckpt(config.save_dir, transformer, global_step, accelerator, ema, transformer_trainable_parameters, config)
        if epoch >= config.num_epochs:  # checkpoint of the last epoch is saved above (num_epochs % save_freq == 0)
            break

        #################### SAMPLING ####################
        pipeline.unet.eval()
        if manual_accum:
            cur_steps = steps_at(config, epoch)
            num_train_timesteps = config.sample.sde_window_size if config.sample.sde_window_size > 0 else cur_steps - 1
        else:
            cur_steps = config.sample.num_steps
        if mixed_reward is not None:
            mixed_reward.set_epoch(epoch)
        if accelerator.is_main_process:
            wandb.log({"num_steps": cur_steps,
                       **({"w_align": mixed_reward.align_weight, "w_ocr": mixed_reward.ocr_weight}
                          if mixed_reward is not None else {})}, step=global_step)
        samples = []
        prompts = []
        t_epoch0 = time.time()
        ep_ratio_dev, ep_clipfrac = [], []  # |ratio-1| and clip fraction over the epoch (1 opt. step/epoch: should be ~0)
        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not accelerator.is_local_main_process,
            position=0,
        ):
            train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)
            prompts, prompt_metadata = next(train_iter)
            prompt_metadata = [{"target_text": p} for p in prompts]

            prompt_embeds = compute_text_embeddings(prompts, text_encoder, tokenizer, device=accelerator.device)
            prompt_ids = tokenizer(
                prompts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(accelerator.device)

            # sample
            if config.sample.same_latent:
                # Seed from (epoch, prompt) only: the sampler shuffles a prompt's k copies across batches, so a
                # batch-index term (the original `+ i`) would give the copies of one group different z_T.
                generator = create_generator(prompts, base_seed=epoch*10000 + (config.seed - 42) * 7919)  # unchanged for seed 42
            else:
                generator = None
            
            x0_cb = mixed_reward.start(prompts, cur_steps) if mixed_reward is not None else None
            with autocast():
                with torch.no_grad():
                    images, latents, log_probs, timesteps, step_rewards = pipeline_with_logprob(
                        pipeline,
                        prompt=prompts,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=sample_neg_prompt_embeds[:len(prompt_embeds)],
                        num_inference_steps=cur_steps,
                        guidance_scale=config.sample.guidance_scale,
                        output_type="pt",
                        height=config.resolution,
                        width=config.resolution, 
                        noise_level=config.sample.noise_level,
                        generator=generator,
                        sde_window_size=config.sample.sde_window_size,
                        sde_window_range=config.sample.sde_window_range,
                        # "final": only the finished image is scored (below). "intermediate": score the x0 preview
                        # at every SDE-window step (previews before step ~5/10 barely predict the final reward,
                        # see scripts/eval_intermediate_latents.py).
                        reward_fn=reward_fn if reward_mode == "intermediate" else None,
                        prompt_metadata=prompt_metadata,
                        x0_callback=x0_cb,
                )

            if reward_mode == "mixed":
                # GPU parts now; OCR runs on the CPU pool (if any) while the next batch is sampled -> resolved below
                finish = mixed_reward.submit(images, prompts)
                step_rewards, final_components = None, {}
            elif reward_mode == "final":
                final_details, _ = reward_fn(images, prompts, prompt_metadata, only_strict=True)
                # every window step gets the final reward (Flow-GRPO style) -> shape (batch, window)
                step_rewards = [final_details["avg"]] * num_train_timesteps
                final_components = {
                    k: torch.as_tensor([float(x) for x in final_details[k]], device=accelerator.device)
                    for k in config.reward_fn
                }
            else:
                final_components = {}

            latents = torch.stack(
                latents, dim=1
            )  # (batch_size, window + 1, 4, 64, 64)
            log_probs = torch.stack(log_probs, dim=1)  # shape after stack (batch_size, num_steps)
            timesteps = torch.stack(timesteps).unsqueeze(0).repeat(config.sample.train_batch_size, 1)  # shape after stack (batch_size, num_steps)

            if reward_mode == "mixed":
                samples.append({"prompt_ids": prompt_ids, "prompt_embeds": prompt_embeds, "timesteps": timesteps,
                                "latents": latents[:, :-1], "next_latents": latents[:, 1:], "log_probs": log_probs,
                                "rewards": finish})
                continue
           # Stack the step-rewards into a tensor of shape (batch_size, num_steps)
            stacked_step_rewards = torch.stack([
                torch.as_tensor(r, device=accelerator.device).float() 
                for r in step_rewards
            ], dim=1)
            
            # Format as the expected dictionary
            rewards_dict = {
                "avg": stacked_step_rewards,
                "ori_avg": stacked_step_rewards[:, -1], # Use final step for standard logging
                **final_components,  # per-reward components of the final image (final mode only), for logging
            }

            samples.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "timesteps": timesteps,
                    "latents": latents[:, :-1],  # each entry is the latent before timestep t
                    "next_latents": latents[:, 1:],  # each entry is the latent after timestep t
                    "log_probs": log_probs,
                    "rewards": rewards_dict,
                }
            )


        t_gpu_done = time.time()
        if reward_mode == "mixed":
            # resolve the deferred rewards (waits for the OCR workers) -> same rewards dict as before
            for s_ in samples:
                comp = s_["rewards"]()
                s_["rewards"] = {"avg": torch.stack([comp["avg"].float()] * num_train_timesteps, dim=1),
                                 "ori_avg": comp["avg"].float(),
                                 **{k: comp[k].float() for k in ("align", "clip_final", "ocr")}}
            rewards_dict = samples[-1]["rewards"]
        t_sampled = time.time()
        # collate samples into dict where each entry has shape (num_batches_per_epoch * sample.batch_size, ...)
        samples = {
            k: torch.cat([s[k] for s in samples], dim=0)
            if not isinstance(samples[0][k], dict)
            else {
                sub_key: torch.cat([s[k][sub_key] for s in samples], dim=0)
                for sub_key in samples[0][k]
            }
            for k in samples[0].keys()
        }

        if epoch % 10 == 0 and accelerator.is_main_process:
            # this is a hack to force wandb to log the images as JPEGs instead of PNGs
            with tempfile.TemporaryDirectory() as tmpdir:
                num_samples = min(15, len(images))
                sample_indices = random.sample(range(len(images)), num_samples)

                for idx, i in enumerate(sample_indices):
                    image = images[i]
                    pil = Image.fromarray(
                        (image.float().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))  # 使用新的索引

                sampled_prompts = [prompts[i] for i in sample_indices]
                sampled_rewards = [rewards_dict['ori_avg'][i].item() for i in sample_indices]

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompt:.100} | avg: {avg_reward:.2f}",
                            )
                            for idx, (prompt, avg_reward) in enumerate(zip(sampled_prompts, sampled_rewards))
                        ],
                    },
                    step=global_step,
                )
        samples["rewards"]["ori_avg"] = samples["rewards"]["avg"]

        if reward_mode == "intermediate":
            step_rewards = samples["rewards"]["avg"]
            delta_rewards = step_rewards[:, 1:] - step_rewards[:, :-1]
            first_step = step_rewards[:, :1]
            samples["rewards"]["avg"] = torch.cat([first_step, delta_rewards], dim=1)
        
        # gather rewards across processes
        gathered_rewards = {key: accelerator.gather(value) for key, value in samples["rewards"].items()}
        gathered_rewards = {key: value.cpu().numpy() for key, value in gathered_rewards.items()}
        # log rewards and images
        if accelerator.is_main_process:
            wandb.log(
                {
                    "epoch": epoch,
                    **{f"reward_{key}": np.nanmean(value) for key, value in gathered_rewards.items() if '_strict_accuracy' not in key and '_accuracy' not in key},
                },
                step=global_step,
            )

        # per-prompt mean/std tracking
        if config.per_prompt_stat_tracking:
            # gather the prompts across processes
            prompt_ids = accelerator.gather(samples["prompt_ids"]).cpu().numpy()
            prompts = pipeline.tokenizer.batch_decode(
                prompt_ids, skip_special_tokens=True
            )

            print("\n=== PONTUAÇÕES DO OCR ===")
            print(samples['rewards']['ori_avg'])
            print("=========================\n")

            advantages = stat_tracker.update(prompts, samples['rewards']['ori_avg'].cpu())
            if accelerator.is_local_main_process:
                print("len(prompts)", len(prompts))
                print("len unique prompts", len(set(prompts)))

            group_size, trained_prompt_num = stat_tracker.get_stats()

            zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(prompts, gathered_rewards)

            if accelerator.is_main_process:
                wandb.log(
                    {
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            advantages = (samples['rewards']['ori_avg'] - samples['rewards']['ori_avg'].mean()) / (samples['rewards']['ori_avg'].std() + 1e-4)

        # ungather advantages; we only need to keep the entries corresponding to the samples on this process
        advantages = torch.as_tensor(advantages)
        samples["advantages"] = (
            advantages.reshape(accelerator.num_processes, -1, advantages.shape[-1])[accelerator.process_index]
            .to(accelerator.device)
        )
        if accelerator.is_local_main_process:
            print("advantages: ", samples["advantages"].abs().mean())

        del samples["rewards"]
        del samples["prompt_ids"]

        # Get the mask for samples where all advantages are zero across the time dimension
        mask = (samples["advantages"].abs().sum(dim=1) != 0)
        
        # If the number of True values in mask is not divisible by config.sample.num_batches_per_epoch,
        # randomly change some False values to True to make it divisible
        # train_chunks = number of training micro-batches per epoch (config.train.micro_batches, default: one per
        # sampling batch). Must equal config.train.gradient_accumulation_steps for one optimizer step per epoch.
        num_batches = train_chunks
        true_count = mask.sum()
        if true_count % num_batches != 0 or true_count == 0:
            false_indices = torch.where(~mask)[0]
            num_to_change = num_batches - (true_count % num_batches)
            if len(false_indices) >= num_to_change:
                random_indices = torch.randperm(len(false_indices))[:num_to_change]
                mask[false_indices[random_indices]] = True
        if accelerator.is_main_process:
            wandb.log(
                {
                    "actual_batch_size": mask.sum().item()//train_chunks,
                },
                step=global_step,
            )
        # Filter out samples where the entire time dimension of advantages is zero
        samples = {k: v[mask] for k, v in samples.items()}

        total_batch_size, num_timesteps = samples["timesteps"].shape
        # assert (
        #     total_batch_size
        #     == config.sample.train_batch_size * config.sample.num_batches_per_epoch
        # )

        #################### TRAINING ####################
        for inner_epoch in range(config.train.num_inner_epochs):
            # rebatch for training
            samples_batched = {
                k: v.reshape(-1, total_batch_size//train_chunks, *v.shape[1:])
                for k, v in samples.items()
            }

            # dict of lists -> list of dicts for easier iteration
            samples_batched = [
                dict(zip(samples_batched, x)) for x in zip(*samples_batched.values())
            ]

            # train
            pipeline.unet.train()
            info = defaultdict(list)
            for i, sample in tqdm(
                list(enumerate(samples_batched)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not accelerator.is_local_main_process,
            ):
                if config.train.cfg:
                    # concat negative prompts to sample prompts to avoid two forward passes
                    embeds = torch.cat(
                        [train_neg_prompt_embeds[:len(sample["prompt_embeds"])], sample["prompt_embeds"]]
                    )
                else:
                    embeds = sample["prompt_embeds"]

                train_timesteps = [step_index  for step_index in range(num_train_timesteps - skip_last)]
                for j in tqdm(
                    train_timesteps,
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ):
                    acc_ctx = contextlib.nullcontext() if manual_accum else accelerator.accumulate(transformer)
                    with acc_ctx:
                        with autocast():
                            prev_sample, log_prob, prev_sample_mean, std_dev_t = compute_log_prob(transformer, pipeline, sample, j, embeds, config)
                            if config.train.beta > 0:
                                with torch.no_grad():
                                    with unwrap_model(transformer, accelerator).disable_adapter():
                                        _, _, prev_sample_mean_ref, _ = compute_log_prob(transformer, pipeline, sample, j, embeds, config)

                        # grpo logic
                        advantages = torch.clamp(
                            sample["advantages"][:, j],
                            -config.train.adv_clip_max,
                            config.train.adv_clip_max,
                        )
                        ratio = torch.exp(log_prob - sample["log_probs"][:, j])
                        adv_reshaped = advantages.view(-1, *[1] * (ratio.ndim - 1))
                        unclipped_loss = -adv_reshaped * ratio
                        clipped_loss = -adv_reshaped * torch.clamp(
                            ratio,
                            1.0 - config.train.clip_range,
                            1.0 + config.train.clip_range,
                        )

                        policy_loss = torch.mean(torch.maximum(unclipped_loss, clipped_loss))
                        if config.train.beta > 0:
                            if prev_sample_mean is not None and prev_sample_mean_ref is not None and std_dev_t is not None:
                                kl_loss = ((prev_sample_mean - prev_sample_mean_ref) ** 2).mean(dim=(1,2,3), keepdim=True) / (2 * std_dev_t ** 2)
                                kl_loss = torch.mean(kl_loss)
                            else:
                                kl_loss = policy_loss * 0.0
                            loss = policy_loss + config.train.beta * kl_loss
                        else:
                            loss = policy_loss

                        ep_ratio_dev.append(torch.mean(torch.abs(ratio.detach() - 1.0)).item())
                        ep_clipfrac.append(torch.mean((torch.abs(ratio.detach() - 1.0) > config.train.clip_range).float()).item())
                        info["approx_kl"].append(
                            0.5
                            * torch.mean((log_prob - sample["log_probs"][:, j]) ** 2)
                        )
                        info["clipfrac"].append(
                            torch.mean(
                                (
                                    torch.abs(ratio - 1.0) > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_gt_one"].append(
                            torch.mean(
                                (
                                    ratio - 1.0 > config.train.clip_range
                                ).float()
                            )
                        )
                        info["clipfrac_lt_one"].append(
                            torch.mean(
                                (
                                    1.0 - ratio > config.train.clip_range
                                ).float()
                            )
                        )
                        info["policy_loss"].append(policy_loss)
                        if config.train.beta > 0:
                            info["kl_loss"].append(kl_loss)

                        info["loss"].append(loss)

                        # backward pass
                        if not loss.requires_grad:
                            loss.requires_grad_(True)

                        if manual_accum:
                            # same scaling as accelerate: average over samples-batches x transitions per update
                            loss = loss / (config.train.gradient_accumulation_steps * len(train_timesteps))
                            do_step = (j == train_timesteps[-1]) and ((i + 1) % config.train.gradient_accumulation_steps == 0)
                        accelerator.backward(loss)

                        if (do_step if manual_accum else accelerator.sync_gradients):
                            if config.train.max_grad_norm is not None and config.train.max_grad_norm > 0:
                                accelerator.clip_grad_norm_(
                                    transformer.parameters(), config.train.max_grad_norm
                                )
                            optimizer.step()
                            optimizer.zero_grad()

                    # Checks if the accelerator has performed an optimization step behind the scenes
                    if (do_step if manual_accum else accelerator.sync_gradients):
                        # assert (j == train_timesteps[-1]) and (
                        #     i + 1
                        # ) % config.train.gradient_accumulation_steps == 0
                        # log training-related stuff
                        info = {k: torch.mean(torch.stack(v)) for k, v in info.items()}
                        info = accelerator.reduce(info, reduction="mean")
                        info.update({"epoch": epoch, "inner_epoch": inner_epoch})
                        if accelerator.is_main_process:
                            wandb.log(info, step=global_step)
                        global_step += 1
                        info = defaultdict(list)
                if config.train.ema:
                    ema.step(transformer_trainable_parameters, global_step)

        # wall-clock per epoch (2026-11-18, efficiency work): GPU sampling, waiting for the OCR, training
        t_end = time.time()
        timing = {"time/sampling_gpu": t_gpu_done - t_epoch0, "time/ocr_wait": t_sampled - t_gpu_done,
                  "time/train": t_end - t_sampled, "time/epoch": t_end - t_epoch0,
                  "mem/max_alloc_gb": torch.cuda.max_memory_allocated() / 2**30,
                  "check/ratio_dev": float(np.mean(ep_ratio_dev)) if ep_ratio_dev else float("nan"),
                  "check/clipfrac": float(np.mean(ep_clipfrac)) if ep_clipfrac else float("nan")}
        if accelerator.is_main_process:
            wandb.log(timing, step=global_step)
            print("[TIMING] epoch %d " % epoch + " ".join("%s=%.1f" % (k, v) for k, v in timing.items()), flush=True)
        epoch+=1
        
if __name__ == "__main__":
    app.run(main)

