import cv2
import numpy as np
import mediapipe as mp
import os
from .utils import ensure_dir

mp_hands = mp.solutions.hands

def extract_hand_region(image: np.ndarray, target_size=(128, 128)):
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            return None # no hand detected
        
        # Get Bounding Box from landmarks
        h, w, _ = image.shape
        x_min = w
        y_min = h
        x_max = y_max = 0

        for landmark in result.multi_hand_landmarks[0].landmark:
            x, y = int(landmark.x * w), int(landmark.y * h)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        # Add padding
        padding = 20
        x_min = max(0, x_min - padding)
        y_min = max(0, y_min - padding)
        x_max = min(w, x_max + padding)
        y_max = min(h, y_max + padding)

        # Crop the hand and resize to the target_size
        hand_crop = image[y_min:y_max, x_min:x_max]
        hand_resized = cv2.resize(hand_crop, target_size)

        # Return the properly sized cropped image of the hand
        return hand_resized
    

def preprocess_all_images(raw_dir="data/raw", out_dir="data/processed"):
    for label in os.listdir(raw_dir):
        src_folder = os.path.join(raw_dir, label)
        # Create the destination folder if it does not exist
        dst_folder = os.path.join(out_dir, label)
        ensure_dir(dst_folder)

        for fname in os.listdir(src_folder):
            if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            # Read the current image file
            img_path = os.path.join(src_folder, fname)
            img = cv2.imread(img_path)

            # Extract the hand region
            cropped = extract_hand_region(img)
            if cropped is None:
                print(f"[!] Skipping {fname} (no hand detected)")
                continue

            save_path = os.path.join(dst_folder, fname)
            cv2.imwrite(save_path, cropped)
            print(f"[✓] Saved preprocessed {save_path}")
