# src/evaluate.py
import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import shutil
from typing import Tuple

# Ensure project root is on path so we can import src.preprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
try:
    from DataCapture.preprocess import extract_hand_region
except Exception as e:
    raise ImportError(f"Couldn't import extract_hand_region from src.preprocess: {e}")

def load_and_preprocess(test_dir: Path, img_size: int, skip_nohand: bool = True) -> Tuple[np.ndarray, np.ndarray, dict, list]:
    """
    Loads images from test_dir/<label>/*.jpg, uses extract_hand_region to crop,
    resizes to img_size and returns X (normalized), y, label_map, and list of skipped files.
    """
    X, y = [], []
    labels = sorted([p.name for p in test_dir.iterdir() if p.is_dir()])
    label_map = {i: label for i, label in enumerate(labels)}
    skipped = []

    for i, label in label_map.items():
        folder = test_dir / label
        for img_path in sorted(folder.glob("*")):
            if not img_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                skipped.append((str(img_path), "imread_failed"))
                continue

            cropped = extract_hand_region(img, target_size=(img_size, img_size))
            if cropped is None:
                skipped.append((str(img_path), "no_hand"))
                if skip_nohand:
                    continue
                # fallback: use full frame resized
                cropped = cv2.resize(img, (img_size, img_size))
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            else:
                # extract_hand_region is expected to return BGR crop -> convert to RGB
                cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

            X.append(cropped.astype(np.float32) / 255.0)
            y.append(i)

    if len(X) == 0:
        raise RuntimeError("No images loaded after preprocessing. Check your test_dir and cropping function.")

    return np.array(X), np.array(y), label_map, skipped

def save_misclassified(X, y_true, y_pred, label_map, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    inv_map = {v: k for k, v in label_map.items()}
    for idx in np.where(y_true != y_pred)[0]:
        true = label_map[y_true[idx]]
        pred = label_map[y_pred[idx]]
        filename = f"true_{true}_pred_{pred}_{idx}.jpg"
        img = (X[idx] * 255.0).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / filename), img)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="models/hand_sign_mobilenetv2.h5")
    p.add_argument("--test_dir", required=True, help="Folder with subfolders per label")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--skip_nohand", action="store_true", help="Skip test images where no hand was detected")
    p.add_argument("--save_mis", help="Directory to save misclassified crops (optional)")
    args = p.parse_args()

    model_path = Path(args.model_path)
    test_dir = Path(args.test_dir)
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not test_dir.exists():
        raise FileNotFoundError(test_dir)

    print("Loading model:", model_path)
    model = tf.keras.models.load_model(str(model_path))

    print("Loading and preprocessing test images...")
    X, y, label_map, skipped = load_and_preprocess(test_dir, args.img_size, skip_nohand=args.skip_nohand)
    print(f"Loaded {len(X)} images across {len(label_map)} labels. Skipped {len(skipped)} files.")

    # Predict
    y_probs = model.predict(X, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Metrics
    acc = accuracy_score(y, y_pred)
    print(f"\nOverall accuracy: {acc:.4f}\n")
    print("Classification report:\n")
    labels_order = [label_map[i] for i in sorted(label_map.keys())]
    print(classification_report(y, y_pred, target_names=labels_order, digits=4))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_order, yticklabels=labels_order)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

    if args.save_mis:
        save_dir = Path(args.save_mis)
        if save_dir.exists():
            # optionally clear previous
            shutil.rmtree(save_dir)
        save_misclassified(X, y, y_pred, label_map, save_dir)
        print(f"Saved misclassified images to {save_dir}")

    if skipped:
        print("\nSample skipped files (reason):")
        for s in skipped[:10]:
            print(s)

if __name__ == "__main__":
    main()
