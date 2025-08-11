# src/live_infer.py
import sys
from pathlib import Path
import argparse
import time
import cv2
import numpy as np
import tensorflow as tf
from collections import deque

# Ensure project root is on path so we can import src.preprocess
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
try:
    from DataCapture.preprocess import extract_hand_region
except Exception as e:
    raise ImportError(f"Couldn't import extract_hand_region from src.preprocess: {e}")

def get_label_list_from_dir(test_dir: Path):
    # common helper: infer labels from train/test folders if available
    labels = sorted([p.name for p in test_dir.iterdir() if p.is_dir()]) if test_dir.exists() else []
    return labels

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", default="models/hand_sign_mobilenetv2.h5")
    p.add_argument("--img_size", type=int, default=128)
    p.add_argument("--confidence_thresh", type=float, default=0.6,
                   help="Minimum confidence to accept prediction")
    p.add_argument("--smooth_window", type=int, default=5,
                   help="Temporal smoothing window (frames)")
    p.add_argument("--labels_dir", default="data/processed",
                   help="Optional directory to infer label order from (subfolders)")
    p.add_argument("--save_snapshots", default=None,
                   help="Optional folder to save snapshots when pressing 's'")
    p.add_argument("--flip", action="store_true",
                   help="Flip (mirror) webcam frames before processing")
    args = p.parse_args()

    model = tf.keras.models.load_model(str(args.model_path))
    labels = get_label_list_from_dir(Path(args.labels_dir))
    if not labels:
        # fallback: require user to pass explicit labels (or edit code)
        raise RuntimeError("Couldn't infer labels from --labels_dir. Ensure it exists with subfolders for labels.")

    print("Labels:", labels)
    IMG_SIZE = args.img_size
    cap = cv2.VideoCapture(0)
    pred_buffer = deque(maxlen=args.smooth_window)
    conf_buffer = deque(maxlen=args.smooth_window)

    save_dir = Path(args.save_snapshots) if args.save_snapshots else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    print("Press 'q' to quit, 's' to save a snapshot (for debugging).")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if args.flip:
            frame = cv2.flip(frame, 1)

        # Optionally draw an informative ROI guide (not required if using landmarks)
        h, w = frame.shape[:2]
        # we'll crop using extract_hand_region which returns an appropriately sized crop
        crop = extract_hand_region(frame, target_size=(IMG_SIZE, IMG_SIZE))

        display_label = ""
        display_conf = 0.0

        if crop is not None:
            # crop is BGR -> convert to RGB
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            inp = (crop_rgb.astype(np.float32) / 255.0)[None, ...]  # shape (1,H,W,3)
            probs = model.predict(inp, verbose=0)[0]
            idx = int(np.argmax(probs))
            conf = float(np.max(probs))
            pred_label = labels[idx]

            pred_buffer.append(pred_label)
            conf_buffer.append(conf)

            # smoothing + thresholding
            # compute most common label among buffer with conf > threshold
            good_preds = [p for p, c in zip(pred_buffer, conf_buffer) if c >= args.confidence_thresh]
            if good_preds:
                # choose mode
                display_label = max(set(good_preds), key=good_preds.count)
                display_conf = np.mean([c for p, c in zip(pred_buffer, conf_buffer) if p == display_label])
            else:
                display_label = "..."
                display_conf = np.mean(conf_buffer) if conf_buffer else 0.0

            # show the crop in a small window for debugging
            small = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
            cv2.imshow("Crop (debug)", small)

        else:
            # no crop found
            pred_buffer.append("no_hand")
            conf_buffer.append(0.0)

        # overlay label on frame
        text = f"{display_label} ({display_conf:.2f})"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        # show frame
        cv2.imshow("Live Hand Sign", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s') and save_dir and crop is not None:
            # save the current crop for debugging (timestamped)
            t = int(time.time())
            fname = f"snapshot_{display_label}_{t}.jpg"
            out = (cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) * 255.0).astype(np.uint8)
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(save_dir / fname), out)
            print("Saved snapshot:", fname)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
