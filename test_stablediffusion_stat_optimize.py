import torch
from diffusers import StableDiffusionXLPipeline
import time
import numpy as np

# Charger modèle avec optimisations
model1 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16
)
model1.to("cuda:0")

# ===== OPTIMISATIONS =====

# 1. Activer xFormers (gain ~30-40%)
model1.enable_xformers_memory_efficient_attention()

# 2. Compiler le UNet avec torch.compile (gain ~30-50%)
model1.unet = torch.compile(model1.unet, mode="reduce-overhead", fullgraph=True)

# 3. VAE slicing pour réduire l'utilisation RAM
model1.enable_vae_slicing()

# ===========================

prompt = "A fantasy landscape, stunning detail, vibrant colors"
num_images = 10
generation_times = []

print(f"Génération de {num_images} images avec optimisations...")
print("-" * 60)

# WARMUP (important pour torch.compile)
print("Warmup du modèle compilé...")
with torch.cuda.device(0):
    _ = model1(prompt, num_inference_steps=20).images[0]
print("Warmup terminé, démarrage des générations...\n")

start_total = time.time()

for i in range(num_images):
    start_time = time.time()
    
    with torch.cuda.device(0):
        image = model1(
            prompt, 
            num_inference_steps=20,  # 20-30 steps suffisent généralement
            guidance_scale=7.5
        ).images[0]
        image.save(f"output_model1_{i:03d}.png")
    
    end_time = time.time()
    elapsed = end_time - start_time
    generation_times.append(elapsed)
    
    print(f"Image {i+1}/{num_images} - Temps: {elapsed:.2f}s")

end_total = time.time()
total_time = end_total - start_total

# Calculer statistiques (exclure le premier warmup)
times_array = np.array(generation_times)
mean_time = np.mean(times_array)
median_time = np.median(times_array)
std_time = np.std(times_array)
min_time = np.min(times_array)
max_time = np.max(times_array)

print("\n" + "=" * 60)
print("STATISTIQUES DE GÉNÉRATION")
print("=" * 60)
print(f"Nombre d'images générées : {num_images}")
print(f"Temps total             : {total_time:.2f}s ({total_time/60:.2f} min)")
print(f"Temps moyen par image   : {mean_time:.2f}s")
print(f"Temps médian            : {median_time:.2f}s")
print(f"Écart-type              : {std_time:.2f}s")
print(f"Temps minimum           : {min_time:.2f}s")
print(f"Temps maximum           : {max_time:.2f}s")
print(f"Images par minute       : {60/mean_time:.2f}")
print(f"Gain estimé vs 14s      : {((14-mean_time)/14)*100:.1f}%")
print("=" * 60)

with open("generation_stats.txt", "w") as f:
    f.write(f"Statistiques - {num_images} images avec optimisations\n")
    f.write(f"Temps total: {total_time:.2f}s\n")
    f.write(f"Temps moyen: {mean_time:.2f}s\n")
    f.write(f"Temps médian: {median_time:.2f}s\n")
    f.write(f"Gain vs 14s: {((14-mean_time)/14)*100:.1f}%\n")
    f.write(f"\nDétail par image:\n")
    for i, t in enumerate(generation_times):
        f.write(f"Image {i:03d}: {t:.2f}s\n")

