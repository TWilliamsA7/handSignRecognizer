import cv2
import numpy as np
import mediapipe as mp
import os
from .utils import ensure_dir

mp_hands = mp.solutions.hands

def extract_hand_region(image, target_size=(128,128), pad_ratio=0.3):
    h, w = image.shape[:2]
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        res = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not res.multi_hand_landmarks:
        return None

    lm = res.multi_hand_landmarks[0].landmark
    xs = [int(l.x * w) for l in lm]
    ys = [int(l.y * h) for l in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # add padding proportional to bbox size
    bw = x_max - x_min
    bh = y_max - y_min
    pad = int(max(bw, bh) * pad_ratio)
    x1 = max(0, x_min - pad)
    y1 = max(0, y_min - pad)
    x2 = min(w, x_max + pad)
    y2 = min(h, y_max + pad)

    crop = image[y1:y2, x1:x2]

    # ensure square crop by padding borders if needed
    ch, cw = crop.shape[:2]
    if ch == 0 or cw == 0:
        return None
    size = max(ch, cw)
    square = np.zeros((size, size, 3), dtype=crop.dtype)
    y_off = (size - ch) // 2
    x_off = (size - cw) // 2
    square[y_off:y_off+ch, x_off:x_off+cw] = crop

    resized = cv2.resize(square, target_size)
    # convert to RGB for training pipeline if needed
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    

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


def preprocess_label(label, raw_dir="data/raw", out_dir="data/processed"):
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



    



    

    
