import cv2
import mediapipe as mp
import csv
import os
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_PATH = "data/raw_landmarks.csv"
os.makedirs("data", exist_ok=True)

if not os.path.exists(DATA_PATH):
    with open(DATA_PATH, mode='w', newline='') as f:
        writer = csv.writer(f)
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}", f"z{i}"]
        header.append("label")
        writer.writerow(header)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

print("\nControls:")
print(" S → Save sample")
print(" L → Change label")
print(" Q → Quit\n")

current_label = input("Enter gesture label: ")
coords_flat = None

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    coords_flat = None

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            h, w, _ = frame.shape
            coords = []
            points = []

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append([lm.x, lm.y, lm.z])
                points.append((cx, cy))
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            for c in HAND_CONNECTIONS:
                cv2.line(frame, points[c[0]], points[c[1]], (0, 255, 0), 2)

            coords_flat = np.array(coords).flatten()

    cv2.putText(frame, f"Label: {current_label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture Data Collection", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if coords_flat is not None:
            with open(DATA_PATH, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(list(coords_flat) + [current_label])
            print("Sample saved!")
        else:
            print("No hand detected")

    if key == ord('l'):
        current_label = input("Enter new label: ")

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
