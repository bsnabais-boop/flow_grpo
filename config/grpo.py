import ml_collections
import imp
import os

base = imp.load_source("base", os.path.join(os.path.dirname(__file__), "base.py"))

def compressibility():
    config = base.get_config()

    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    config.use_lora = True

    config.sample.batch_size = 8
    config.sample.num_batches_per_epoch = 4

    config.train.batch_size = 4
    config.train.gradient_accumulation_steps = 2

    # prompting
    config.prompt_fn = "general_ocr"

    # rewards
    config.reward_fn = {"jpeg_compressibility": 1}
    config.per_prompt_stat_tracking = True
    return config

def general_ocr_wan2_1():
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # config.pretrained.model = "hf_cache/Wan2.1-T2V-14B-Diffusers"
    config.pretrained.model = "hf_cache/Wan2.1-T2V-1.3B-Diffusers"
    config.sample.num_steps = 20
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale=4.5
    config.run_name = "wan_flow_grpo"
    
    config.height = 240
    config.width = 416
    config.frames = 33
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 4 # 12
    config.sample.num_batches_per_epoch = 2
    config.sample.sample_time_per_prompt = 1
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt // 2 if (config.sample.num_batches_per_epoch * config.sample.sample_time_per_prompt) > 1 else 1
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.004
    config.train.learning_rate = 1e-4
    config.train.clip_range=1e-3
    # kl reward
    # KL reward and KL loss are two ways to incorporate KL divergence. KL reward adds KL to the reward, while KL loss, introduced by GRPO, directly adds KL loss to the policy loss. We support both methods, but KL loss is recommended as the preferred option.
    config.sample.kl_reward = 0
    # We also support using SFT data in RL training for supervised learning to prevent quality drop, but this option was unused
    config.train.sft=0.0
    config.train.sft_batch_size=3
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std=False
    config.train.ema=True
    config.mixed_precision = "bf16"
    config.diffusion_loss = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.num_epochs = 100000
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = f'logs/video_ocr/{config.run_name}'
    config.resume_from = None
    config.reward_fn = {
        "video_ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = 4
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    # Whether to use the same noise for the same prompt
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 80.0,
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.04
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = f'logs/geneval/sd3.5-M'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def geneval_sd3_fast_nocfg():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1
    config.sample.eval_guidance_scale = 1
    config.train.cfg = False

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/geneval/sd3.5-M-fast-nocfg'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def clipscore_sd3():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.02
    config.sample.global_std = True
    config.sample.same_latent = True
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/clipscore/sd3.5-M'
    config.reward_fn = {
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.train_num_steps = 2
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    # 这里固定为1
    config.sample.train_batch_size = 1
    config.sample.num_image_per_prompt = 24
    config.sample.mini_num_image_per_prompt = 9
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.mini_num_image_per_prompt/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.mini_num_image_per_prompt
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M-fast'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_sd3_fast_nocfg():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 1
    config.sample.eval_guidance_scale = 1
    config.train.cfg = False

    config.resolution = 512
    config.sample.train_batch_size = 9
    config.sample.num_image_per_prompt = 18
    config.sample.num_batches_per_epoch = int(64/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/geneval/sd3.5-M-fast-nocfg'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_4gpu():
    gpu_number = 4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    # A large num_epochs is intentionally set here. Training will be manually stopped once sufficient
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_sd3_4gpu():
    gpu_number=4
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(16/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0.01
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.save_freq = 60 # epoch
    config.eval_freq = 60
    config.save_dir = 'logs/pickscore/sd3.5-M'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd3_1gpu():
    gpu_number = 1
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr")
    config.resume_from_checkpoint = 'latest'

    # sd3.5 medium
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 4.5

    config.resolution = 512
    config.sample.train_batch_size = 8
    config.sample.num_image_per_prompt = 8
    config.sample.num_batches_per_epoch = int(8/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # 16 is a special design, the test set has a total of 1018, to make 8*16*n as close as possible to 1018, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    # kl loss
    config.train.beta = 0.04
    # Whether to use the std of all samples or the current group's.
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.learning_rate = 1e-5
    config.train.ema = True
    config.save_freq = 120 # epoch
    config.eval_freq = 120
    config.save_dir = 'logs/ocr/sd3.5-M'
    config.reward_fn = {
        "ocr": 80.0,
        "clipscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def general_ocr_sd15_1gpu():
    # SD 1.5 (epsilon-prediction UNet) trained with GRPO over stochastic DDIM steps (DDPO-style log-probs).
    # Use with scripts/train_sd15.py.
    gpu_number = 1
    config = general_ocr_sd3_1gpu()
    config.resume_from_checkpoint = None

    config.pretrained.model = "runwayml/stable-diffusion-v1-5"
    config.resolution = 512  # SD1.5 native resolution
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 40
    config.sample.guidance_scale = 7.5
    config.sample.eval_guidance_scale = 7.5
    # DDIM eta inside the SDE window (1.0 = DDPM-like variance, as in DDPO)
    config.sample.noise_level = 1.0

    config.train.lora_rank = 32
    config.train.learning_rate = 1e-4
    config.save_dir = 'logs/ocr/sd1.5'
    return config

def general_ocr_sd15_simple_1gpu():
    # Test run after the intermediate-latent evaluation (logs/intermediate_eval):
    # - reward_mode "final": score the finished image instead of x0 previews inside the SDE window
    #   (previews before step ~5/10 have ~0 rank correlation with the final reward)
    # - short 1-2 word targets (dataset/ocr_simple) so SD 1.5 gets a non-trivial OCR signal
    # - SDE window in the first half of the trajectory, where layout/text are decided (Flow-GRPO default)
    config = general_ocr_sd15_1gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_simple")
    config.sample.reward_mode = "final"
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps // 2)
    config.sample.test_batch_size = 16  # test set = 128 prompts
    config.eval_freq = 20
    config.save_freq = 40
    config.save_dir = 'logs/ocr/sd1.5_simple'
    return config

def general_ocr_sd15_simple_32p_1gpu():
    # Same as general_ocr_sd15_simple_1gpu (run 625986 was flat: LoRA weights moved but reward did not),
    # with a less noisy signal:
    # - 32 prompts x 8 images per epoch instead of 8 x 8 (num_batches_per_epoch 8 -> 32), still 2 updates/epoch
    # - fixed-seed eval with 4 images per prompt (128 test prompts -> 512 images), so evals are comparable
    config = general_ocr_sd15_simple_1gpu()
    config.sample.num_batches_per_epoch = 32  # each batch = 1 prompt x 8 images (train_batch_size 8, k 8)
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.sample.eval_images_per_prompt = 4
    config.sample.eval_seed = 1234
    # Evaluate with the training sampler (10 steps, stochastic SDE window), so evals measure what GRPO optimises.
    # Under that sampler, checkpoint-140 of run 626027 was only +0.011 OCR over base (0.112 -> 0.123, not significant).
    config.sample.eval_sampler = "train"
    config.eval_freq = 10  # epochs are ~4x longer now
    config.save_freq = 10
    config.save_dir = 'logs/ocr/sd1.5_simple_32p'
    return config

def general_ocr_sd15_simple_32p_win28_1gpu():
    # Same as general_ocr_sd15_simple_32p_1gpu, but the 2-step SDE window is drawn from steps 2-7 instead of 0-4.
    # Per-step previews show text forming at steps 3-5 of 10, while the old window mostly trained steps before that.
    config = general_ocr_sd15_simple_32p_1gpu()
    config.sample.sde_window_range = (2, 8)  # start = randint(2, 6) -> trained steps within 2..7
    config.save_dir = 'logs/ocr/sd1.5_simple_32p_win28'
    return config

def general_ocr_sd15_simple_32p_win4_1gpu():
    # Fallback if the 2-step window in steps 2-7 stays flat: train 4 of the 10 steps per trajectory, all inside
    # the steps where text forms (start = randint(2, 4) -> trained steps within 2..7). Training phase ~2x longer.
    config = general_ocr_sd15_simple_32p_win28_1gpu()
    config.sample.sde_window_size = 4
    config.sample.sde_window_range = (2, 8)
    config.save_dir = 'logs/ocr/sd1.5_simple_32p_win4'
    return config

# W&B diagnostics of the win4 run (8xplk4di): the 1st of the 2 optimizer steps per epoch has clipfrac 0, the 2nd has
# clipfrac ~0.86 (|log ratio| ~5e-4 after one step vs clip_range 1e-4), so ~half of every epoch gives no gradient, and
# approx_kl ~1e-7 per step. The three options below are tried one after the other, each only if the previous one
# still does not learn.

def general_ocr_sd15_win4_clip1e3_1gpu():
    # Option 1: widen the PPO clip range so the 2nd optimizer step of each epoch is not almost entirely clipped.
    config = general_ocr_sd15_simple_32p_win4_1gpu()
    config.train.clip_range = 1e-3
    config.train.clip_range_lt = 1e-3
    config.train.clip_range_gt = 1e-3
    config.save_dir = 'logs/ocr/sd1.5_win4_clip1e3'
    return config

def general_ocr_sd15_win4_1step_1gpu():
    # Option 2: one optimizer step per epoch (accumulate over all 32 batches), so no sample is seen by a moved policy
    # and nothing is clipped. clip_range back at the default 1e-4.
    config = general_ocr_sd15_simple_32p_win4_1gpu()
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.save_dir = 'logs/ocr/sd1.5_win4_1step'
    return config

def general_ocr_sd15_win4_1step_lr3e4_1gpu():
    # Option 3: option 2 plus learning rate 1e-4 -> 3e-4 (config/base.py default, DDPO's LoRA value), so each of the
    # unclipped steps moves the policy more.
    config = general_ocr_sd15_win4_1step_1gpu()
    config.train.learning_rate = 3e-4
    config.save_dir = 'logs/ocr/sd1.5_win4_1step_lr3e4'
    return config

def mixed_sd15_orm_1gpu():
    # Phase 3, baseline (outcome reward): prompt alignment + text rendering on a mixed prompt set
    # (dataset/mixed_align_ocr = 1440 GenAI-Bench training prompts + 1088 OCR prompts; held-out eval prompts excluded).
    # reward = 100 * CLIP(final image, prompt) + 80 * OCR (OCR only for prompts with a quoted text), see
    # flow_grpo/mixed_reward.py. 10-step sampler, all 9 stochastic steps trained (no window), per-prompt advantages.
    config = general_ocr_sd15_simple_32p_1gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/mixed_align_ocr")
    config.sample.reward_mode = "mixed"
    config.sample.align_reward = "clip"
    config.sample.sde_window_size = 0          # stochastic DDIM at all steps, train on all 9 transitions
    config.sample.global_std = False           # per-prompt std: OCR and alignment rewards have different scales
    config.sample.num_batches_per_epoch = 32   # 32 prompts x 8 images per epoch
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch // 2
    config.train.learning_rate = 3e-4
    config.train.clip_range = 1e-3
    config.train.clip_range_lt = 1e-3
    config.train.clip_range_gt = 1e-3
    config.train.beta = 0.04                   # KL to the reference model (slides: against reward hacking)
    config.eval_freq = 0                       # evaluated offline with scripts/phase3_eval.py on saved checkpoints
    config.save_freq = 10
    config.save_dir = 'logs/phase3/mixed_orm'
    return config

def mixed_sd15_prm_1gpu():
    # Phase 3, our method: same as mixed_sd15_orm_1gpu, but the alignment reward is the noise-aware PRM
    # (logs/phase1_scorer/v2_lat_10steps_eta1) on x0_hat at steps 4, 6, 8, 10 (no VAE decode), averaged
    # = cumulative process reward. OCR still needs the final image.
    config = mixed_sd15_orm_1gpu()
    config.sample.align_reward = "prm"
    config.sample.prm_dir = "logs/phase1_scorer/v2_lat_10steps_eta1"
    config.sample.prm_phi = "logs/phase1_noisyclip/sd15_50steps/phi.npy"
    config.sample.prm_steps = (3, 5, 7, 9)
    config.save_dir = 'logs/phase3/mixed_prm'
    return config

def mixed_sd15_prm_wsched_1gpu():
    # Supervisor's suggestion 1: reward-weight curriculum. Alignment first, text later: over the first 60 epochs the
    # OCR weight rises 0 -> 80 and the alignment weight falls 100 -> 50 (never to zero). Otherwise = mixed_sd15_prm.
    config = mixed_sd15_prm_1gpu()
    config.sample.weight_schedule = (60, 0.5)
    config.save_dir = 'logs/phase3/mixed_prm_wsched'
    return config

def mixed_sd15_prm_ssched_1gpu():
    # Supervisor's suggestion 2: denoising-steps curriculum 4 -> 6 -> 8 -> 10 (20 epochs each; 2 steps give SD1.5
    # images without usable content). Early epochs are 2-2.5x cheaper. PRM queried at 40/60/80/100% of the
    # trajectory for the current step count. Manual gradient accumulation (2 optimizer steps/epoch as before).
    config = mixed_sd15_prm_1gpu()
    config.sample.steps_schedule = [(0, 4), (20, 6), (40, 8), (60, 10)]
    config.sample.prm_fracs = (0.4, 0.6, 0.8, 1.0)
    config.save_dir = 'logs/phase3/mixed_prm_ssched'
    return config

def mixed_sd15_prm_both_1gpu():
    # Both curricula together.
    config = mixed_sd15_prm_ssched_1gpu()
    config.sample.weight_schedule = (60, 0.5)
    config.save_dir = 'logs/phase3/mixed_prm_both'
    return config

def mixed_sd15_nc_1gpu():
    # Supervisor (2026-11-11): do not train our own CLIP scorer, use the published NoisyCLIP (HF asiimo/noisyclip) as
    # the process reward for alignment. Same steps (4, 6, 8, 10 of 10) and everything else as mixed_sd15_prm_1gpu.
    # noisyclip_input: "zt" (noisy latent, as in the paper) | "x0" | "x0phi" -> pick from logs/phase1_noisyclip/eval10_noisyclip.
    config = mixed_sd15_prm_1gpu()
    config.sample.align_reward = 'noisyclip'
    config.sample.noisyclip_name = 'asiimo/noisyclip'
    config.sample.noisyclip_input = 'zt'
    config.save_dir = 'logs/phase3/mixed_nc'
    return config

def mixed_sd15_nc_wsched_1gpu():
    # Weight curriculum without starting OCR at zero (supervisor 2026-11-11): over 60 epochs OCR 40 -> 80 (starts at
    # half), alignment 100 -> 50.
    config = mixed_sd15_nc_1gpu()
    config.sample.weight_schedule = (60, 0.5, 0.5)
    config.save_dir = 'logs/phase3/mixed_nc_wsched'
    return config

def mixed_sd15_nc_both_1gpu():
    # Both curricula (steps 4 -> 10 and weights with OCR starting at half) with NoisyCLIP.
    config = mixed_sd15_nc_wsched_1gpu()
    config.sample.steps_schedule = [(0, 4), (20, 6), (40, 8), (60, 10)]
    config.sample.prm_fracs = (0.4, 0.6, 0.8, 1.0)
    config.save_dir = 'logs/phase3/mixed_nc_both'
    return config

def mixed_sd15_nc_wsched_fix_1gpu():
    # NoisyCLIP + weight curriculum (OCR 40 -> 80) + the three fixes suggested by scripts/diag_ocr_signal.py (2026-11-11):
    # 1. skip_last=1: the last transition (t=101->1) gave 57% of the policy gradient but cannot change the text;
    #    train transitions 0-7 only (the KL is then also computed without it).
    # 2. beta 0.04 -> 0.01: the KL gradient was 0.7-1.2x the policy gradient at the end of training.
    # 3. one optimizer step per epoch on all 256 samples (was 2 x 128): per-step gradient SNR was ~0.6 at 128.
    # NoisyCLIP input x0phi: the best NoisyCLIP variant on SD1.5 (logs/phase1_noisyclip/eval10_noisyclip;
    # z_t with the SDXL colour map is near chance on SD1.5).
    config = mixed_sd15_nc_wsched_1gpu()
    config.sample.noisyclip_input = 'x0phi'
    config.train.skip_last = 1
    config.train.beta = 0.01
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.save_dir = 'logs/phase3/mixed_nc_wsched_fix'
    return config

def ocr_long_win23_1gpu():
    # Option 1 (2026-11-13): the OCR policy-gradient signal is ~1-2e-4 per image (logs/diag_ocr_signal/v3_variants), so a
    # run needs ~7-10x the images of nc_wsched_fix (25.6k) for a reproducible OCR effect. Make each image cheaper and use
    # many more of them:
    # - text prompts only (dataset/ocr_simple, 1088 train words; test words disjoint), NoisyCLIP x0phi alignment kept
    #   as a regulariser (default weights 100 align + 80 OCR; OCR is ~85% of the within-group variance)
    # - shared z_T per group + 2-step SDE window fixed at steps 2-3 (diag variant v4: highest point estimate of the
    #   per-image signal, 1.6e-4) -> 2 trained transitions instead of 8 (~4x cheaper training per image)
    # - 64 prompts x 8 = 512 images per epoch, one optimizer step per epoch; 500 epochs = 256k images (10x nc_wsched_fix)
    config = mixed_sd15_nc_1gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/ocr_simple")
    config.sample.noisyclip_input = 'x0phi'
    config.sample.same_latent = True
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (2, 4)
    config.sample.num_batches_per_epoch = 64
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch
    config.train.skip_last = 0
    config.train.beta = 0.01
    config.num_epochs = 500
    config.save_freq = 50
    config.eval_freq = 0
    config.save_dir = 'logs/phase3/ocr_long_win23'
    return config

def ocr_long_win23_seed2_1gpu():
    # Second seed of ocr_long_win23_1gpu (2026-11-15): different prompt order, initial noise and SDE noise.
    config = ocr_long_win23_1gpu()
    config.seed = 43
    config.save_dir = 'logs/phase3/ocr_long_win23_seed2'
    return config

def ocr_long_win23_cont_1gpu():
    # Continue ocr_long_win23 for another 500 epochs from its final LoRA (epoch 500). Fresh optimizer state; KL still to the
    # base model. checkpoint-N here = epoch 500 + N in total.
    config = ocr_long_win23_1gpu()
    config.train.lora_path = 'logs/phase3/ocr_long_win23/checkpoints/checkpoint-500/lora'
    config.save_dir = 'logs/phase3/ocr_long_win23_cont'
    return config

def mixed_long_win23_wsched_1gpu():
    # Long run on text + alignment prompts (2026-11-17, user request after the meeting prep): the volume recipe of
    # ocr_long_win23_1gpu (shared z_T, 2-step SDE window at steps 2-3, 512 imgs/epoch, 1 optimizer step/epoch, beta 0.01,
    # 500 epochs) on dataset/mixed_align_ocr (1440 GenAI-Bench + 1088 OCR prompts) with the weight schedule of the
    # NoisyCLIP run (alignment 100 -> 50, OCR 40 -> 80 over 60 epochs; only on text prompts). Always 10 steps (no step
    # schedule: with the window at 2-3 a 4-step phase would train the last step). ~43% text prompts -> ~110k text images.
    config = ocr_long_win23_1gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/mixed_align_ocr")
    config.sample.weight_schedule = (60, 0.5, 0.5)
    config.save_dir = 'logs/phase3/mixed_long_win23_wsched'
    return config

def _mixed_ratio(ratio, tag):
    # Weight study (supervisor 2026-11-18: explore the OCR / CLIP weights and the schedules). With per-prompt advantage
    # normalisation only the OCR/alignment ratio on text prompts matters (alignment-only prompts are unaffected), so the
    # study is one knob: fixed ratio, no schedule. Same recipe as mixed_long_win23_wsched (shared z_T, window steps 2-3,
    # 512 imgs/epoch, 1 optimizer step/epoch, beta 0.01), 300 epochs (the mixed run's OCR gain appears at ep 200-300).
    # Reference: mixed_long_win23_wsched = ramp 0.4 -> 1.6 over 60 epochs.
    config = ocr_long_win23_1gpu()
    config.dataset = os.path.join(os.getcwd(), "dataset/mixed_align_ocr")
    config.sample.weight_schedule = None
    config.sample.ocr_weight = 80.0
    config.sample.align_weight = 80.0 / ratio
    config.num_epochs = 300
    config.save_dir = f'logs/phase3/mixed_ratio_{tag}'
    return config

def mixed_ratio_08_1gpu():
    return _mixed_ratio(0.8, "08")  # = the default weights 80 OCR / 100 alignment, no schedule

def mixed_ratio_16_1gpu():
    return _mixed_ratio(1.6, "16")  # = the end point of the current schedule (80 / 50)

def mixed_ratio_32_1gpu():
    return _mixed_ratio(3.2, "32")  # text-heavy: does alignment hold?

def pickscore_flux():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config

def pickscore_flux_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-group24-8gpu'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def geneval_flux_fast():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/geneval")

    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 14 # This bs is a special design, the test set has a total of 2212, to make gpu_num*bs*n as close as possible to 2212, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/geneval/flux_fast'
    config.reward_fn = {
        "geneval": 1.0,
    }
    
    config.prompt_fn = "geneval"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_flux_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "black-forest-labs/FLUX.1-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 3.5
    config.sample.eval_guidance_scale = 3.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 24
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 16 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.clip_range = 1e-5
    config.train.beta = 0
    config.sample.global_std = False
    config.sample.same_latent = False
    config.sample.noise_level = 0.8
    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.sample.sde_type = "cps"
    config.train.ema = True
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/flux-fast'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def counting_flux_kontext():
    gpu_number=28
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # sd3.5 medium
    config.pretrained.model = "black-forest-labs/FLUX.1-Kontext-dev"
    config.sample.num_steps = 6
    config.sample.eval_num_steps = 28
    config.sample.guidance_scale = 2.5

    config.resolution = 512
    config.sample.train_batch_size = 3
    config.sample.num_image_per_prompt = 21
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 2 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.timestep_fraction = 0.99
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = True
    config.sample.noise_level = 0.9
    config.mixed_precision = "bf16"
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/counting_edit/flux_kontext'
    config.reward_fn = {
        "image_similarity": 0.5,
        "geneval": 0.5,
    }
    config.per_prompt_stat_tracking = True
    return config

def pickscore_qwenimage():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def pickscore_qwenimage_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.2
    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True
    return config


def counting_qwenimage_edit():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.0
    config.sample.sde_window_size = 0
    # config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def counting_qwenimage_edit_fast():
    gpu_number=32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.5
    config.sample.sde_window_size = 4
    config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def counting_qwenimage_edit_8gpu():
    gpu_number=8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/counting_edit")

    # flux
    config.pretrained.model = "Qwen/Qwen-Image-Edit"
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4

    config.resolution = 512
    config.sample.train_batch_size = 4
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(32/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))
    assert config.sample.num_batches_per_epoch % 2 == 0, "Please set config.sample.num_batches_per_epoch to an even number! This ensures that config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch / 2, so that gradients are updated twice per epoch."
    config.sample.test_batch_size = 4 # This bs is a special design, the test set has a total of 2048, to make gpu_num*bs*n as close as possible to 2048, because when the number of samples cannot be divided evenly by the number of cards, multi-card will fill the last batch to ensure each card has the same number of samples, affecting gradient synchronization.

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2
    config.train.num_inner_epochs = 1
    config.train.beta = 0
    config.sample.global_std = True
    config.sample.same_latent = False
    config.train.ema = False
    config.sample.noise_level = 1.0
    config.sample.sde_window_size = 0
    # config.sample.sde_window_range = (0, config.sample.num_steps//2)
    config.mixed_precision = "bf16"
    config.use_lora = True
    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    config.save_freq = 60 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/qwenimage_edit'
    config.reward_fn = {
        "image_similarity": 0.2,
        "geneval": 0.8,
    }
    config.per_prompt_stat_tracking = True
    return config

def pickscore_bagel():
    gpu_number = 32
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.run_name = "[bagel-pickscore-full]-32gpu"
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.train.cfg = True     # No effect for BAGEL, always use cfg in code.
    config.train.ema = False
    config.use_lora = False

    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))    # =2 for 32 gpus
    config.sample.test_batch_size = 1 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2

    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    config.sample.same_latent = False
    config.sample.global_std = False
    config.sample.noise_level = 1.3

    config.sample.sde_window_size = 3
    config.sample.sde_window_range = (0, config.sample.num_steps//2)

    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/bagel'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    return config


def pickscore_bagel_lora():
    gpu_number = 8
    config = compressibility()
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # sd3.5 medium
    config.run_name = "[bagel-pickscore-lora]-8gpu"
    config.pretrained.model = "ByteDance-Seed/BAGEL-7B-MoT"
    config.sample.num_steps = 15
    config.sample.eval_num_steps = 50
    config.sample.guidance_scale = 4.0
    config.sample.eval_guidance_scale = 4.0
    config.train.cfg = True     # No effect for BAGEL, always use cfg in code.
    config.train.ema = False
    config.use_lora = True

    config.resolution = 512
    config.sample.train_batch_size = 6
    config.sample.num_image_per_prompt = 16
    config.sample.num_batches_per_epoch = int(48/(gpu_number*config.sample.train_batch_size/config.sample.num_image_per_prompt))    # =2 for 32 gpus
    config.sample.test_batch_size = 1 

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = config.sample.num_batches_per_epoch//2

    config.train.num_inner_epochs = 1
    config.train.clip_range_lt = 1e-5
    config.train.clip_range_gt = 1e-5
    config.train.beta = 0
    config.train.learning_rate = 1e-4
    config.mixed_precision = "bf16"

    config.sample.same_latent = False
    config.sample.global_std = False
    config.sample.noise_level = 1.3

    config.sample.sde_window_size = 2
    config.sample.sde_window_range = (0, config.sample.num_steps//2)

    config.save_freq = 30 # epoch
    config.eval_freq = 30
    config.save_dir = 'logs/pickscore/bagel'
    config.reward_fn = {
        "pickscore": 1.0,
    }
    
    config.prompt_fn = "general_ocr"

    config.per_prompt_stat_tracking = True

    config.activation_checkpointing = True
    config.fsdp_optimizer_offload = True
    return config


def noisyclip_sd3_fast():
    # Base settings
    config = compressibility() 
    config.dataset = os.path.join(os.getcwd(), "dataset/pickscore")

    # Lightest Model
    config.pretrained.model = "stabilityai/stable-diffusion-3.5-medium"
    
    # Lower resolution
    config.resolution = 256 
    
    # Fast sampling steps
    config.sample.num_steps = 10
    config.sample.eval_num_steps = 10
    config.sample.guidance_scale = 4.5

    # Batch settings
    config.sample.train_batch_size = 2
    config.sample.num_image_per_prompt = 2 
    config.sample.num_batches_per_epoch = 2 
    config.sample.test_batch_size = 2

    config.train.batch_size = config.sample.train_batch_size
    config.train.gradient_accumulation_steps = 1
    config.train.num_inner_epochs = 1
    config.train.beta = 0.01 
    
    config.train.ema = True
    config.save_freq = 20
    config.eval_freq = 20
    config.save_dir = 'logs/noisyclip/sd3.5-M-fast'
    
    # NoisyCLIP Reward
    config.reward_fn = {
        "noisyclip": 1.0,
    }
    
    config.prompt_fn = "general_ocr"
    config.per_prompt_stat_tracking = True
    
    return config

def get_config(name):
    return globals()[name]()
