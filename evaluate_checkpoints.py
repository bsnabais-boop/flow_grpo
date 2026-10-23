import torch
import os
import matplotlib.pyplot as plt
from diffusers import StableDiffusion3Pipeline
from transformers import CLIPProcessor, CLIPModel
from peft import PeftModel

# ==========================================
# CONFIGURAÇÕES DE BOUNDS E INTERVALOS
# ==========================================
MIN_STEP = 400        
MAX_STEP = 13000      
STEP_INTERVAL = 100   
# ==========================================

current_dir = os.getcwd()
checkpoints_dir = os.path.join(current_dir, "logs/noisyclip/sd3.5-M-fast/checkpoints")
output_base_dir = os.path.join(current_dir, "evaluation_results_full")
os.makedirs(output_base_dir, exist_ok=True)

if os.path.exists(checkpoints_dir):
    all_checkpoints = [d for d in os.listdir(checkpoints_dir) if d.startswith("checkpoint-")]
    all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("-")[1]))
else:
    all_checkpoints = []

checkpoints_to_test = []
for ckpt in all_checkpoints:
    step_num = int(ckpt.split("-")[1])
    if MIN_STEP <= step_num <= MAX_STEP and step_num % STEP_INTERVAL == 0:
        checkpoints_to_test.append(ckpt)

print(f"Checkpoints to evaluate: {checkpoints_to_test}")

test_prompts = [
    "A sign that clearly says 'FLOW GRPO' in a futuristic city",
    "A vintage coffee mug with text that reads 'GRPO' printed on the front",
    "A close-up photo of a wooden sign that clearly spells out 'TEXT' in bold black letters",
    "A neon sign on a building at night displaying the word 'SUCCESS'"
]

print("\n=== LOADING BASE MODEL (ONCE) ===")
model_id = "stabilityai/stable-diffusion-3.5-medium"
pipe = StableDiffusion3Pipeline.from_pretrained(
    model_id, 
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

clip_model_id = "openai/clip-vit-large-patch14"
clip_model = CLIPModel.from_pretrained(clip_model_id).to("cuda")
clip_processor = CLIPProcessor.from_pretrained(clip_model_id)

steps_list = []
scores_list = []

print("\n=== STARTING EVALUATION & PLOTTING PIPELINE ===")
for ckpt in checkpoints_to_test:
    step_num = int(ckpt.split("-")[1])
    ckpt_path = os.path.join(checkpoints_dir, ckpt, "lora")
    
    if not os.path.exists(ckpt_path):
        print(f"\n[Warning] Lora folder not found at: {ckpt_path}. Skipping...")
        continue
        
    print(f"\n--------------------------------------------------")
    print(f"Testing checkpoint: {ckpt}")
    print(f"--------------------------------------------------")
    
    # Carregar os pesos LoRA do PEFT diretamente para o transformer usando PeftModel
    try:
        # Se já houver um adapter carregado, descarregamos primeiro para limpar
        if isinstance(pipe.transformer, PeftModel):
            pipe.transformer = pipe.transformer.unload()
            
        pipe.transformer = PeftModel.from_pretrained(pipe.transformer, ckpt_path)
    except Exception as e:
        print(f"Error loading PEFT weights for {ckpt}: {e}")
        continue

    ckpt_output_dir = os.path.join(output_base_dir, ckpt)
    os.makedirs(ckpt_output_dir, exist_ok=True)

    checkpoint_scores = []

    for idx, prompt in enumerate(test_prompts):
        generator = torch.manual_seed(42)  
        image = pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=4.5,
            generator=generator
        ).images[0]

        safe_prompt_name = "".join([c if c.isalnum() else "_" for c in prompt[:15]])
        image_filename = f"p{idx}_{safe_prompt_name}.png"
        image_path = os.path.join(ckpt_output_dir, image_filename)
        image.save(image_path)

        inputs = clip_processor(text=[prompt], images=image, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            outputs = clip_model(**inputs)
            score = outputs.logits_per_image.item()
            checkpoint_scores.append(score)

    avg_checkpoint_score = sum(checkpoint_scores) / len(checkpoint_scores)
    steps_list.append(step_num)
    scores_list.append(avg_checkpoint_score)
    print(f"  -> Step {step_num} | Average CLIP Score (4 prompts): {avg_checkpoint_score:.4f}")

# 2. Generate and Save the Progress Graph
if steps_list:
    print("\n=== GENERATING FINAL PROGRESS GRAPH ===")
    plt.figure(figsize=(10, 5))
    plt.plot(steps_list, scores_list, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    plt.title("Model Average Text-Alignment Score Over Training Steps (Flow-GRPO)", fontsize=14)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Average CLIP Alignment Score", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    graph_path = os.path.join(output_base_dir, "training_progress_score.png")
    plt.savefig(graph_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SUCCESS] Graph saved at: {graph_path}")

print("\n==================================================")
print("Evaluation completed successfully!")
print(f"Results saved inside: {output_base_dir}")
print("==================================================")