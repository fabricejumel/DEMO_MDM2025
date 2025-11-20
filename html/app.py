#!/usr/bin/env python3

from flask import Flask, jsonify, send_from_directory, request
import os
import subprocess

app = Flask(__name__)

# Chemins absolus
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGES_FOLDER = os.path.join(BASE_DIR, 'img')  # Changé en 'img'

# Configuration de l'imprimante (votre commande exacte)
PRINTER_NAME = "HP-Color-LaserJet-5700"
PRINTER_OPTIONS = [
    "-o", "media=A6",
    "-o", "InputSlot=Tray2",
    "-o", "mediaType=HP Brochure Glossy 200g",
    "-o", "orientation-requested=3",  # 3=portrait
]

@app.route('/')
def index():
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/api/images')
def list_images():
    print(f"Recherche d'images dans: {IMAGES_FOLDER}")
    
    if not os.path.exists(IMAGES_FOLDER):
        print("Le dossier img n'existe pas!")
        os.makedirs(IMAGES_FOLDER)
        return jsonify([])
    
    files = os.listdir(IMAGES_FOLDER)
    print(f"Fichiers trouvés: {files}")
    
    image_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))])
    print(f"Images filtrées: {image_files}")
    
    return jsonify(image_files)

@app.route('/images/<path:filename>')
def serve_image(filename):
    print(f"Demande d'image: {filename}")
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route('/api/print/<int:image_index>', methods=['POST'])
def print_image(image_index):
    """Route pour imprimer une image via la commande lp"""
    try:
        # Récupérer la liste des images
        files = os.listdir(IMAGES_FOLDER)
        image_files = sorted([f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'))])
        
        if image_index < 0 or image_index >= len(image_files):
            return jsonify({"error": "Index d'image invalide"}), 400
        
        # Chemin complet de l'image à imprimer
        image_path = os.path.join(IMAGES_FOLDER, image_files[image_index])
        
        print(f"\n{'='*60}")
        print(f"Impression de l'image #{image_index + 1}: {image_files[image_index]}")
        print(f"Chemin: {image_path}")
        
        # Construire la commande lp
        cmd = ["lp", "-d", PRINTER_NAME] + PRINTER_OPTIONS + [image_path]
        
        print(f"Commande complète:")
        print(f"  {' '.join(cmd)}")
        print(f"{'='*60}\n")
        
        # Exécuter la commande d'impression
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ Impression réussie: {result.stdout.strip()}")
            return jsonify({
                "success": True, 
                "message": f"Impression lancée pour l'image {image_index + 1}",
                "filename": image_files[image_index],
                "output": result.stdout.strip()
            })
        else:
            print(f"✗ Erreur d'impression: {result.stderr}")
            return jsonify({
                "success": False, 
                "error": result.stderr
            }), 500
            
    except Exception as e:
        print(f"✗ Exception lors de l'impression: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    print(f"\n{'='*60}")
    print(f"Configuration du serveur d'impression")
    print(f"{'='*60}")
    print(f"Dossier de base: {BASE_DIR}")
    print(f"Dossier images: {IMAGES_FOLDER}")
    print(f"Index.html existe: {os.path.exists(os.path.join(BASE_DIR, 'index.html'))}")
    print(f"Dossier img existe: {os.path.exists(IMAGES_FOLDER)}")
    print(f"\nImprimante: {PRINTER_NAME}")
    print(f"Options d'impression:")
    for i in range(0, len(PRINTER_OPTIONS), 2):
        print(f"  {PRINTER_OPTIONS[i]} {PRINTER_OPTIONS[i+1]}")
    print(f"{'='*60}\n")
    
    app.run(host='0.0.0.0', port=8000, debug=True)

