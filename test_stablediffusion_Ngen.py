import torch
from diffusers import StableDiffusionXLPipeline
from datetime import datetime
import time
import numpy as np
import os

# Créer le dossier img s'il n'existe pas
os.makedirs("img", exist_ok=True)

# Charger modèle
print("Chargement du modèle...")
model1 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16
)
model1.to("cuda:0")
print("✓ Modèle chargé\n")

prompt = "shiny color, cartoon style, futurist,  team of developpers (women and men) different race  in the room creating incredible projects using interface as Précrime in minority report "
num_images = 500
generation_times = []

print(f"Génération de {num_images} images...")
print("=" * 70)

start_total = time.time()

for i in range(num_images):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    with torch.cuda.device(0):
        image = model1(prompt, num_inference_steps=20).images[0]
        # MODIFICATION ICI : ajouter img/ devant
        filename = f"img/output_{i+1:04d}_{timestamp}.png"
        image.save(filename)
    
    elapsed = time.time() - start_time
    generation_times.append(elapsed)
    
    if (i + 1) % 10 == 0 or i == 0:
        avg_time = np.mean(generation_times)
        remaining = (num_images - i - 1) * avg_time
        print(f"[{i+1:04d}/{num_images}] Temps: {elapsed:.2f}s | "
              f"Moy: {avg_time:.2f}s | Restant: {remaining/60:.1f}min")

total_time = time.time() - start_total
times_array = np.array(generation_times)

print("\n" + "=" * 70)
print("STATISTIQUES FINALES")
print("=" * 70)
print(f"Images générées         : {num_images}")
print(f"Temps total             : {total_time/60:.2f} min ({total_time:.1f}s)")
print(f"Temps moyen par image   : {times_array.mean():.2f}s")
print(f"Temps médian            : {np.median(times_array):.2f}s")
print(f"Min: {times_array.min():.2f}s | Max: {times_array.max():.2f}s")
print(f"Images par minute       : {60/times_array.mean():.2f}")
print("=" * 70)

print("\n✓ 500 images sauvegardées dans ./img/")

