import cv2
import os
import time
from .utils import ensure_dir
from . import labels

def capture_images(label: str, num_samples: int, lighting: bool):
    # Create a directory to save the image data/raw/label
    save_dir = os.path.join("data", "raw", label)
    ensure_dir(save_dir)

    # Setup Video Capture with OpenCV
    cap = cv2.VideoCapture(0)
    numLightSamples =  int(num_samples / 3)
    count = 0

    # Info logging
    print(f"[INFO] Capturing {num_samples} images for label: '{label}'")
    print(f"[SPACE] to capture | [ESC] to exit")

    # For each sample
    while count < num_samples:
        ret, frame = cap.read()

        # If lighting is turned on, signal to change lighting
        if (lighting):
            if count == numLightSamples:
                print(f"Please Switch Lighting Level: Press [SPACE] To Continue")
                while cv2.waitKey(1) != 32:
                    continue
                numLightSamples += int(num_samples / 3)

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
        # Create a file using the captured frame
        filepath = os.path.join(save_dir, f"{label}_{count:04d}.jpg")
        cv2.imwrite(filepath, frame)
        print(f"[✓] Saved {filepath}")
        count += 1
        time.sleep(0.05)

    # Remove the OpenCV instance
    cap.release()
    cv2.destroyAllWindows()


def capture_all_images(num_samples: int, lighting: bool):
    cap = cv2.VideoCapture(0)
    numLightSamples = int(num_samples / 3)

    for label in labels:
        # Create a directory to save the image data/raw/label
        save_dir = os.path.join("data", "raw", label)
        ensure_dir(save_dir)

        # Setup Video Capture with OpenCV
        count = 0
        numLightSamples = int(num_samples / 3)

        # Info logging
        print(f"[INFO] Capturing {num_samples} images for label: '{label}'")
        print(f"Press [SPACE] to start capturing")

        ret, frame = cap.read()
        cv2.imshow("Capture", frame)
        while cv2.waitKey(1) != 32:
            continue

        # For each sample
        while count < num_samples:
            ret, frame = cap.read()

            # If lighting is turned on, signal to change lighting
            if (lighting):
                if count == numLightSamples:
                    print(f"Please Switch Lighting Level: Press [SPACE] To Continue")
                    while cv2.waitKey(1) != 32:
                        continue
                    numLightSamples += int(num_samples / 3)


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

            # Create a file using the captured frame
            filepath = os.path.join(save_dir, f"{label}_{count:04d}.jpg")
            cv2.imwrite(filepath, frame)
            print(f"[✓] Saved {filepath}")
            count += 1
            time.sleep(0.10)

    # Remove the OpenCV instance
    cap.release()
    cv2.destroyAllWindows()