import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

# Charger le pipeline img2img SDXL
pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
pipeline.to("cuda:0")

# Charger une image source
# Option 1: Depuis une URL
#init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")

# Option 2: Depuis un fichier local
init_image = Image.open("input_fabrice.jpg").convert("RGB")

# Redimensionner l'image si nécessaire (SDXL préfère 1024x1024 ou multiples de 8)
init_image = init_image.resize((1024, 1024))

# Prompt pour guider la transformation
prompt = "A engineering creates incredible project"
negative_prompt = "blurry, low quality, distorted"

# Générer l'image
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    strength=0.75,  # 0.0 = image identique, 1.0 = totalement nouvelle
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]

# Sauvegarder
image.save("output_img2img.png")
print("Génération terminée!")

