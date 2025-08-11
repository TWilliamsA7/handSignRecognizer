# src/evaluate.py
import argparse
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from DataCapture import preprocess
import matplotlib.pyplot as plt

def load_test_images(test_dir, img_size):
    X, y, labels = [], [], []
    label_map = {i: d.name for i, d in enumerate(sorted(Path(test_dir).iterdir()))}
    for i, folder in label_map.items():
        for img_path in (Path(test_dir) / folder).glob("*.jpg"):
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cropped = preprocess.extract_hand_region(img, target_size=(img_size, img_size))
            if cropped is None:
                continue
            X.append(cropped)
            y.append(i)
    return np.array(X) / 255.0, np.array(y), label_map

def main(args):
    # Load model
    model = tf.keras.models.load_model(args.model_path)
    # Load test images
    X_test, y_test, label_map = load_test_images(args.test_dir, args.img_size)
    # Predict
    y_probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_probs, axis=1)

    # Report
    print("Classification Report:\n")
    print(classification_report(y_test, y_pred,
          target_names=[label_map[i] for i in sorted(label_map)]))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[label_map[i] for i in sorted(label_map)],
                yticklabels=[label_map[i] for i in sorted(label_map)])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix on New Test Set")
    plt.show()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="models/hand_sign_mobilenetv2.h5")
    p.add_argument("--test_dir", required=True, help="folder with subdirs per label")
    p.add_argument("--img_size", type=int, default=128)
    args = p.parse_args()
    main(args)
