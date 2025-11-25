import torch
from diffusers import StableDiffusionXLPipeline
import time
import numpy as np

# Charger modèle 1 sur GPU 0
model1 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16
)
model1.to("cuda:0")

prompt = "A fantasy landscape, stunning detail, vibrant colors"

# Paramètres
num_images = 100
generation_times = []

print(f"Génération de {num_images} images...")
print("-" * 60)

start_total = time.time()

# Générer 100 images
for i in range(num_images):
    start_time = time.time()
    
    with torch.cuda.device(0):
        image = model1(prompt).images[0]
        image.save(f"output_model1_{i:03d}.png")
    
    end_time = time.time()
    elapsed = end_time - start_time
    generation_times.append(elapsed)
    
    # Afficher progression
    print(f"Image {i+1}/{num_images} - Temps: {elapsed:.2f}s")

end_total = time.time()
total_time = end_total - start_total

# Calculer statistiques
times_array = np.array(generation_times)
mean_time = np.mean(times_array)
median_time = np.median(times_array)
std_time = np.std(times_array)
min_time = np.min(times_array)
max_time = np.max(times_array)

# Afficher statistiques
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
print("=" * 60)

# Sauvegarder les stats dans un fichier
with open("generation_stats.txt", "w") as f:
    f.write(f"Statistiques de génération - {num_images} images\n")
    f.write(f"Temps total: {total_time:.2f}s\n")
    f.write(f"Temps moyen: {mean_time:.2f}s\n")
    f.write(f"Temps médian: {median_time:.2f}s\n")
    f.write(f"Écart-type: {std_time:.2f}s\n")
    f.write(f"Min: {min_time:.2f}s, Max: {max_time:.2f}s\n")
    f.write(f"\nDétail par image:\n")
    for i, t in enumerate(generation_times):
        f.write(f"Image {i:03d}: {t:.2f}s\n")

print("\nStatistiques sauvegardées dans 'generation_stats.txt'")

