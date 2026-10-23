import torch
from diffusers import StableDiffusion3Pipeline

checkpoint_path = "logs/noisyclip/sd3.5-M-fast/checkpoints/checkpoint-1320/lora"

print(f"Loading base model and weights from: {checkpoint_path}")

model_id = "stabilityai/stable-diffusion-3.5-medium"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)

pipe = pipe.to("cuda")

pipe.load_lora_weights(checkpoint_path, weight_name="adapter_model.safetensors")

prompt = "A realistic photograph of a modern highway scene with a large billboard prominently displaying the text 'Gas Next Exit 2 Miles' against a backdrop of rolling hills and clear blue sky."
print(f"Prompt: {prompt}")

image = pipe(
    prompt=prompt,
    num_inference_steps=28,
    guidance_scale=4.5,
).images[0]

output_path = "test_output.png"
image.save(output_path)
print(f"Image saved in: {output_path}")