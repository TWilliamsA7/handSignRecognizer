import cv2
import os
import time
from .utils import ensure_dir

def capture_images(label: str, num_samples: int = 100, auto_capture: bool = False):
    # Create a directory to save the image data/raw/label
    save_dir = os.path.join("data", "raw", label)
    ensure_dir(save_dir)

    # Setup Video Capture with OpenCV
    cap = cv2.VideoCapture(0)
    count = 0

    # Info logging
    print(f"[INFO] Capturing {num_samples} images for label: '{label}'")
    print(f"[SPACE] to capture | [ESC] to exit")

    # For each sample
    while count < num_samples:
        ret, frame = cap.read()

        # If there is nothing returned
        if not ret:
            break

        # Place Text on Window to Log Sample Number
        cv2.putText(frame, f"{label}: {count}/{num_samples}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Capture", frame)

        key = cv2.waitKey(1)
        if key == 27: # ESC
            break
        elif key == 32 or auto_capture: # SPACE
            # Create a file using the captured frame
            filepath = os.path.join(save_dir, f"{label}_{count:03d}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"[✓] Saved {filepath}")
            count += 1
            if (auto_capture):
                time.sleep(0.10)

    # Remove the OpenCV instance
    cap.release()
    cv2.destroyAllWindows()

