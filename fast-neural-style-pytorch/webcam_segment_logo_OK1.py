import cv2
import transformer
import torch
import utils
import os
import glob
import numpy as np
import torchvision.transforms as transforms
from torchvision.models.segmentation import deeplabv3_resnet101
import threading
import time
import subprocess
import pygame
import mediapipe as mp

# Paramètres globaux
WIDTH = 1748
HEIGHT = 1240 

PRESERVE_COLOR = False
ENABLE_SEGMENTATION = True
SHOW_DEBUG = False

WEBCAM_SCALE = 1.2
WEBCAM_POSITION = "bottom_center"

BACKGROUND_COLOR = (255, 255, 255)

TEMPLATE_PATH = "../logo/MDM2025_template_A6_paysage_small.png"
template = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_UNCHANGED)
if template is None or template.shape[1] != WIDTH or template.shape[0] != HEIGHT:
    raise ValueError("Template PNG introuvable ou résolution ne correspond pas à WIDTH et HEIGHT")

def overlay_transparent(background, overlay):
    overlay_img = overlay[:, :, :3]
    overlay_mask = overlay[:, :, 3:] / 255.0
    for c in range(3):
        background[:, :, c] = overlay_mask[:, :, 0]*overlay_img[:, :, c] + (1 - overlay_mask[:, :, 0])*background[:, :, c]
    return background

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

seg_model = deeplabv3_resnet101(pretrained=True).to(device)
seg_model.eval()

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((520, 520)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

model_paths = sorted(glob.glob("transforms/*.pth"))
if not model_paths:
    raise RuntimeError("No style models found in transforms/ folder")
models = []
for p in model_paths:
    net = transformer.TransformerNetwork()
    net.load_state_dict(torch.load(p, map_location=device))
    net.to(device)
    net.eval()
    models.append(net)
print(f"Loaded {len(models)} style models.")

current_model_idx = 0

WEBCAM_WIDTH = 1280
WEBCAM_HEIGHT = 720

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_HEIGHT)

cv2.namedWindow("Style Transfer Webcam", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Style Transfer Webcam", 1280, 720)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=2,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.5)

SAVE_DIR = "./img_shoot"
os.makedirs(SAVE_DIR, exist_ok=True)

pygame.mixer.init()
click_path = "click.wav"
click_sound = pygame.mixer.Sound(click_path) if os.path.isfile(click_path) else None

PRINTER_NAME = "HP_Color_LaserJet_5700_USB"
PRINTER_OPTIONS = [
    "-o", "media=A6",
    "-o", "InputSlot=Tray2",
    "-o", "mediaType=HP Brochure Glossy 200g",
    "-o", "orientation-requested=4"
]

PRINT_COOLDOWN = 0
last_print_time = 0
global gesture_active, gesture_countdown, start_time_countdown, gesture_history
    
gesture_active = False
gesture_countdown = 0
start_time_countdown = None

GESTURE_WINDOW = 4
gesture_history = []

def is_victory(landmarks):
    if not landmarks:
        return False
    idx_tip = landmarks.landmark[8]
    idx_pip = landmarks.landmark[6]
    mid_tip = landmarks.landmark[12]
    mid_pip = landmarks.landmark[10]
    rng_tip = landmarks.landmark[16]
    rng_pip = landmarks.landmark[14]
    pinky_tip = landmarks.landmark[20]
    pinky_pip = landmarks.landmark[18]
    thumb_tip = landmarks.landmark[4]
    thumb_mcp = landmarks.landmark[2]

    index_up = idx_tip.y < idx_pip.y
    middle_up = mid_tip.y < mid_pip.y
    ring_down = rng_tip.y > rng_pip.y
    pinky_down = pinky_tip.y > pinky_pip.y
    thumb_down = thumb_tip.x < thumb_mcp.x  # main droite

    result = index_up and middle_up and ring_down and pinky_down and thumb_down
    if result:
        print("Geste victoire détecté par landmarks")
    return result

def trigger_print(image):
    global last_print_time
    now = time.time()
    if now - last_print_time < PRINT_COOLDOWN:
        print("Cooldown d'impression actif, attente...")
        return
    last_print_time = now

    filename = os.path.join(SAVE_DIR, f"print_{int(now)}.png")
    cv2.imwrite(filename, image)
    print(f"Image sauvegardée localement : {filename}")

    def print_thread():
        try:
            cmd = ["lp", "-d", PRINTER_NAME] + PRINTER_OPTIONS + [filename]
            subprocess.run(cmd, check=True)
            if click_sound:
                pygame.mixer.Sound.play(click_sound)
            print("Impression lancée avec succès.")
        except Exception as e:
            print(f"Erreur lors de l'impression: {e}")

    threading.Thread(target=print_thread, daemon=True).start()

print("Début du traitement webcam...")

while True:
    ret, img = cam.read()
    if not ret:
        print("ERROR: Cannot read from webcam")
        break

    img = cv2.flip(img, 1)
    img_original = img.copy()

    if ENABLE_SEGMENTATION:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = preprocess(img_rgb).unsqueeze(0).to(device)
        with torch.no_grad():
            output = seg_model(input_tensor)['out'][0]
        mask = output.argmax(0).byte().cpu().numpy()
        mask = cv2.resize(mask, (WEBCAM_WIDTH, WEBCAM_HEIGHT), interpolation=cv2.INTER_NEAREST)
        person_mask = (mask == 15).astype(np.uint8)
    else:
        person_mask = np.zeros((WEBCAM_HEIGHT, WEBCAM_WIDTH), dtype=np.uint8)

    person_mask_3d = np.stack([person_mask] * 3, axis=2)

    with torch.no_grad():
        input_t = utils.itot(img).to(device)
        generated_t = models[current_model_idx](input_t)
        generated_image = utils.ttoi(generated_t.detach())

    if PRESERVE_COLOR:
        generated_image = utils.transfercolor(img, generated_image)

    generated_image = np.clip(generated_image, 0, 255).astype(np.uint8)

    composed = generated_image * person_mask_3d + (1 - person_mask_3d) * np.array(BACKGROUND_COLOR, dtype=np.uint8)

    scaled_w = int(WEBCAM_WIDTH * WEBCAM_SCALE)
    scaled_h = int(WEBCAM_HEIGHT * WEBCAM_SCALE)
    resized = cv2.resize(composed, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((HEIGHT, WIDTH, 3), BACKGROUND_COLOR, dtype=np.uint8)

    if WEBCAM_POSITION == "center":
        x_off = (WIDTH - scaled_w) // 2
        y_off = (HEIGHT - scaled_h) // 2
    elif WEBCAM_POSITION == "bottom_center":
        x_off = (WIDTH - scaled_w) // 2
        y_off = HEIGHT - scaled_h
    elif WEBCAM_POSITION == "center_left":
        x_off = 0
        y_off = (HEIGHT - scaled_h) // 2
    elif WEBCAM_POSITION == "center_right":
        x_off = WIDTH - scaled_w
        y_off = (HEIGHT - scaled_h) // 2
    else:
        x_off = (WIDTH - scaled_w) // 2
        y_off = (HEIGHT - scaled_h) // 2

    canvas[y_off:y_off + scaled_h, x_off:x_off + scaled_w] = resized

    final_img = overlay_transparent(canvas, template)

    # MediaPipe detection multiple mains
    rgb_hands = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_hands)



    detected_this_frame = False

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if is_victory(hand_landmarks):
                detected_this_frame = True
                break

    # Mise à jour de l'historique pour tolérance (fenêtre glissante)
    gesture_history.append(1 if detected_this_frame else 0)
    if len(gesture_history) > GESTURE_WINDOW:
        gesture_history.pop(0)

    gesture_detected = sum(gesture_history) >= (GESTURE_WINDOW - 1)  # tolérance 1 frame

    now = time.time()

    if gesture_detected and not gesture_active:
        gesture_active = True
        gesture_countdown = 3
        start_time_countdown = now
        print("Geste victoire détecté ! Lancement compte à rebours...")

    if gesture_active:
        elapsed = now - start_time_countdown
        if elapsed >= 1:
            gesture_countdown -= 1
            start_time_countdown = now
        if gesture_countdown > 0:
            cv2.putText(final_img, f"Impression dans {gesture_countdown}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 6)
        else:
            gesture_active = False
            trigger_print(final_img)

    cv2.imshow("Style Transfer Webcam", final_img)

    if SHOW_DEBUG:
        debug_w, debug_h = WEBCAM_WIDTH // 2, WEBCAM_HEIGHT // 2
        img_dbg = cv2.resize(img_original, (debug_w, debug_h))
        mask_dbg = cv2.resize(person_mask * 255, (debug_w, debug_h))
        mask_dbg_colored = cv2.cvtColor(mask_dbg, cv2.COLOR_GRAY2BGR)
        debug_view = np.hstack((img_dbg, mask_dbg_colored))
        cv2.imshow("Debug View - Segmentation", debug_view)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key in [ord('a'), ord('A')]:
        current_model_idx = (current_model_idx - 1) % len(models)
        print("Switched to", os.path.basename(model_paths[current_model_idx]))
    elif key in [ord('z'), ord('Z')]:
        current_model_idx = (current_model_idx + 1) % len(models)
        print("Switched to", os.path.basename(model_paths[current_model_idx]))
    elif key in [ord('s'), ord('S')]:
        ENABLE_SEGMENTATION = not ENABLE_SEGMENTATION
        print(f"Segmentation {'ON' if ENABLE_SEGMENTATION else 'OFF'}")
    elif key in [ord('d'), ord('D')]:
        SHOW_DEBUG = not SHOW_DEBUG
        if not SHOW_DEBUG:
            cv2.destroyWindow("Debug View - Segmentation")
        print(f"Debug view {'ON' if SHOW_DEBUG else 'OFF'}")

cam.release()
cv2.destroyAllWindows()

