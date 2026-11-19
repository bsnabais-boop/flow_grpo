import random

input_file = "train.txt"
output_file = "train_complex.txt"

modifiers = [
    " as a glowing neon sign in a rainy cyberpunk city, highly detailed, 8k",
    " engraved on a rusty metallic spaceship hull, cinematic lighting",
    " written on an ancient magical parchment surrounded by glowing aura, fantasy art",
    " displayed on a glitching holographic billboard in a dystopian street",
    " sculpted in marble inside a grand, hyper-realistic museum hall, volumetric lighting"
]

with open(input_file, 'r', encoding='utf-8') as f:
    prompts = f.readlines()

with open(output_file, 'w', encoding='utf-8') as f:
    for prompt in prompts:
        prompt = prompt.strip()
        if not prompt:
            continue
        
        mod = random.choice(modifiers)
        new_prompt = f"{prompt}{mod}\n"
        f.write(new_prompt)
