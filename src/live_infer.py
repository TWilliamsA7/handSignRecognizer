import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from collections import deque
pred_buffer = deque(maxlen=5)

# Load model and constants
MODEL_PATH = Path("") / "models" / "cnn_hand_sign_model_02.h5"
IMG_SIZE = 128
LABELS = ['A', 'B', 'OK']
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_roi(frame, roi_coords):
    x1, y1, x2, y2 = roi_coords
    roi = frame[y1:y2, x1:x2]
    roi = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    roi = roi / 255.0
    return roi

# Webcam capture
cap = cv2.VideoCapture(0)

# Define ROI (Region of Interest)
roi_coords = (100, 100, 300, 300)
print("🟢 Running live hand sign detection. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Frame Flip
    frame = cv2.flip(frame, 1)

    # Draw ROI rectangle
    x1, y1, x2, y2 = roi_coords
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

    # Extract ROI
    roi = preprocess_roi(frame, roi_coords=roi_coords)
    input_tensor = np.expand_dims(roi, axis=0)

    # Predict
    predictions = model.predict(input_tensor, verbose=0)
    pred_label_idx = np.argmax(predictions)
    confidence = np.max(predictions)
    pred_label = LABELS[pred_label_idx]

    pred_buffer.append(pred_label)
    most_common = max(set(pred_buffer), key=pred_buffer.count)

    # Display prediction
    label_text = f"{most_common} ({confidence:.2f})"
    cv2.putText(frame, label_text, (x1, y1 - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow("Hand Sign Detection", frame)

    # Quit Key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
