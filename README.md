# Drone_Gesture_Detection

<b>Hand Gesture Recognition using MediaPipe & Deep Learning</b>
1. A real-time hand gesture recognition system built using MediaPipe Hand Landmarks and a Neural Network classifier.
2. This project captures hand landmark coordinates from a webcam, trains a deep learning model, and predicts gestures live.
<br>
<hr>
<br>
<b>Features</b>
1. Real-time hand tracking using MediaPipe
2. Deep learning gesture classification (TensorFlow)
3. Custom gesture dataset collection
4. Live prediction from webcam feed
5. Model + label encoder + scaler saving
6. Easy to add new gestures
<br>
<hr>
<br>
<b>Tech Stack</b>
<pre>Python
OpenCV
MediaPipe
TensorFlow / Keras
NumPy, Pandas, Scikit-learn</pre>
<br>
<hr>
<br>
<bProject Structure</b>
<img width="209" height="495" alt="image" src="https://github.com/user-attachments/assets/e69a9591-31c7-43e0-bf2b-de177460f74d" />
<br>
<hr>
<br>
<b>Note:-</b>
<pre>
Large files are excluded from this repository:
Collected dataset (data/)
Trained models (models/)
MediaPipe model file
You must download the MediaPipe model manually.
</pre>
<br>
<hr>
<br>
<b>Setup Instructions</b>
<pre>
Clone the repository:
git clone https://github.com/YOUR_USERNAME/gesture_bot_ann.git
cd gesture_bot_ann

Create virtual environment:
python -m venv gesture_env
gesture_env\Scripts\activate #windows

Install dependencies:
pip install -r requirements.txt

Download MediaPipe Model:
Download hand_landmarker.task from MediaPipe official release and place it in:

models/hand_landmarker.task

Step 1: Collect Gesture Data
python scripts/collect_data.py

Controls:
Key	Action
S	Save sample
L	Change gesture label
Q	Quit

Step 2: Train Model
python scripts/train_model.py

This will create:
gesture_landmark_model.keras
label_encoder.pkl
scaler.pkl

Step 3: Run Live Prediction
python scripts/predict_gesture.py

The webcam will display the detected gesture in real time.
</pre>

<b>Supported Features:</b>

<b>Future Improvements:</b>
1. More gesture classes
2. Data augmentation
3. CNN / LSTM based temporal model
4. Deploy as web app
5. Mobile integration

Author:
<b>Shahzeb Ali</b>

