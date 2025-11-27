import cv2
import transformer
import torch
import utils
import os
import glob

WIDTH = 1280
HEIGHT = 720
PRESERVE_COLOR = False

def webcam(width=1280, height=720):
    """
    Captures webcam, perform style transfer with switchable models using A/Z keys.
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    
    # Scan all available models in transforms folder
    model_paths = sorted(glob.glob("transforms/*.pth"))
    
    if len(model_paths) == 0:
        print("ERROR: No models found in transforms/ folder")
        return
    
    print(f"\n=== Found {len(model_paths)} models ===")
    for i, path in enumerate(model_paths):
        print(f"[{i}] {os.path.basename(path)}")
    
    # Preload all models
    print("\n=== Preloading all models ===")
    models = []
    for i, model_path in enumerate(model_paths):
        print(f"Loading {i+1}/{len(model_paths)}: {os.path.basename(model_path)}...")
        net = transformer.TransformerNetwork()
        net.load_state_dict(torch.load(model_path, map_location=device))
        net = net.to(device)
        net.eval()
        models.append(net)
        print(f"  ✓ Loaded")
    
    print("\n=== All models preloaded ===")
    print("Controls: A=Previous | Z=Next | ESC=Quit\n")
    
    # Initialize model index
    current_model_idx = 0

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        while True:
            # Get webcam input
            ret_val, img = cam.read()
            
            if not ret_val:
                print("ERROR: Cannot read from webcam")
                break

            # Mirror 
            img = cv2.flip(img, 1)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Generate image with current model
            content_tensor = utils.itot(img).to(device)
            generated_tensor = models[current_model_idx](content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            
            if PRESERVE_COLOR:
                generated_image = utils.transfer_color(img, generated_image)

            generated_image = generated_image / 255

            # Show webcam
            cv2.imshow('Style Transfer Webcam', generated_image)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == 27:  # ESC to quit
                break
            elif key == ord('a') or key == ord('A'):  # Previous model
                current_model_idx = (current_model_idx - 1) % len(models)
                print(f"Switched to: {os.path.basename(model_paths[current_model_idx])}")
            elif key == ord('z') or key == ord('Z'):  # Next model
                current_model_idx = (current_model_idx + 1) % len(models)
                print(f"Switched to: {os.path.basename(model_paths[current_model_idx])}")
            
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

# Run
webcam(WIDTH, HEIGHT)

# ============================================================================
# USAGE / UTILISATION
# ============================================================================
# 
# 1. INSTALLATION :
#    git clone https://github.com/rrmina/fast-neural-style-pytorch.git
#    cd fast-neural-style-pytorch
#    pip install torch torchvision opencv-python pillow
#
# 2. LANCEMENT :
#    python webcam_mdm.py
#
# 3. CONTRÔLES EN TEMPS RÉEL :
#    - Touche A : passer au modèle précédent
#    - Touche Z : passer au modèle suivant
#    - Touche ESC : quitter l'application
#
# 4. MODÈLES DISPONIBLES :
#    Le script scanne automatiquement tous les fichiers .pth dans transforms/
#    Tous les modèles sont préchargés en VRAM au démarrage
#
# 5. AJOUTER VOS PROPRES STYLES :
#    python train.py --dataset path/to/coco \
#                    --style-image styles/mon_image_style.jpg \
#                    --save-model-path transforms/mon_style.pth \
#                    --epochs 1 --batch-size 8 --cuda 1
#
# 6. CONFIGURATION :
#    WIDTH = 1280          # Largeur webcam
#    HEIGHT = 720          # Hauteur webcam
#    PRESERVE_COLOR = False # True pour conserver couleurs originales
#
# 7. PERFORMANCES RTX 3090 (24GB VRAM) :
#    - Préchargement : tous les modèles en VRAM
#    - Switching : instantané (pas de rechargement)
#    - 720p : 60+ FPS
#    - 1080p : 30-45 FPS
#
# 8. STYLES SANS COPYRIGHT :
#    - Unsplash/Pexels (CC0)
#    - Wikimedia Commons (Public Domain)
#    - Vos propres créations
#
# ============================================================================

