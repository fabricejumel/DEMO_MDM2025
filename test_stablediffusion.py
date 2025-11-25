import torch
from diffusers import StableDiffusionXLPipeline
from datetime import datetime

# Charger modèle 1 sur GPU 0
model1 = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", 
    torch_dtype=torch.float16
)
model1.to("cuda:0")

prompt = "colored cartoon style, i want an engineering at his desk style futurist with augmented reality, creating an incredible projects"

# Générer timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Générer image avec modèle 1 sur GPU 0
with torch.cuda.device(0):
    image1 = model1(prompt).images[0]
    filename = f"output_model1_{timestamp}.png"
    image1.save(filename)
    print(f"Image sauvegardée: {filename}")

print("Génération terminée.")

