import torch
from safetensors.torch import load_file
import os

# Caminho para dois checkpoints diferentes (ex: início e fim)
ckpt_early = "logs/noisyclip/sd3.5-M-fast/checkpoints/checkpoint-400/lora/adapter_model.safetensors"
ckpt_late = "logs/noisyclip/sd3.5-M-fast/checkpoints/checkpoint-13000/lora/adapter_model.safetensors"

if os.path.exists(ckpt_early) and os.path.exists(ckpt_late):
    weights_400 = load_file(ckpt_early)
    weights_13000 = load_file(ckpt_late)
    
    # Escolher uma matriz de peso para comparar (ex: primeira camada lora_A)
    key_to_check = "base_model.model.transformer_blocks.0.attn.to_q.lora_A.weight"
    
    if key_to_check in weights_400 and key_to_check in weights_13000:
        diff = torch.sum(torch.abs(weights_13000[key_to_check] - weights_400[key_to_check]))
        print(f"Diferença absoluta entre o Step 400 e o Step 13000: {diff.item():.6f}")
        
        if diff.item() == 0.0:
            print("[ALERTA] Os pesos são IDÊNTICOS! O treino estagnou completamente ou parou de atualizar.")
        else:
            print("[SUCESSO] Os pesos mudaram! O treino alterou os valores ao longo do tempo.")
    else:
        print("Chave não encontrada, verifique os nomes no dicionário.")
else:
    print("Um dos checkpoints não foi encontrado.")