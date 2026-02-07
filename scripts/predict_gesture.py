import cv2
import numpy as np
import tensorflow as tf
import joblib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model = tf.keras.models.load_model("models/gesture_landmark_model.keras")
le = joblib.load("models/label_encoder.pkl")
scaler = joblib.load("models/scaler.pkl")

base_options = python.BaseOptions(model_asset_path="models/hand_landmarker.task")
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),  
    (0,5),(5,6),(6,7),(7,8),     
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            h, w, _ = frame.shape
            coords = []

            for lm in hand_landmarks:
                cx, cy = int(lm.x * w), int(lm.y * h)
                coords.append([lm.x, lm.y, lm.z])
                cv2.circle(frame, (cx, cy), 6, (0, 0, 255), -1)

            for c in HAND_CONNECTIONS:
                x1 = int(hand_landmarks[c[0]].x * w)
                y1 = int(hand_landmarks[c[0]].y * h)
                x2 = int(hand_landmarks[c[1]].x * w)
                y2 = int(hand_landmarks[c[1]].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            row = np.array(coords).flatten().reshape(1, -1)
            row = scaler.transform(row)

            pred = model.predict(row, verbose=0)
            confidence = np.max(pred)

            if confidence > 0.80:
                gesture = le.inverse_transform([np.argmax(pred)])[0]
            else:
                gesture = "Unknown"

            cv2.putText(frame, f"{gesture} ({confidence:.2f})",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2)

    cv2.imshow("Gesture Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
