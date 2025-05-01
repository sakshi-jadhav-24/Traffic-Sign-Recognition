import streamlit as st
import cv2
import torch
import numpy as np
from src.model import load_model
from src.preprocessing import preprocess_image_pytorch

# Constants
IMG_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 70

# Traffic Sign Classes
classes = {
    0: 'Speed limit (20km/h)', 1: 'Speed limit (30km/h)', 2: 'Speed limit (50km/h)',
    3: 'Speed limit (60km/h)', 4: 'Speed limit (70km/h)', 5: 'Speed limit (80km/h)',
    6: 'End of speed limit (80km/h)', 7: 'Speed limit (100km/h)', 8: 'Speed limit (120km/h)',
    9: 'No passing', 10: 'No passing for vehicles over 3.5 metric tons',
    11: 'Right-of-way at the next intersection', 12: 'Priority road', 13: 'Yield', 14: 'Stop',
    15: 'No vehicles', 16: 'Vehicles over 3.5 metric tons prohibited', 17: 'No entry',
    18: 'General caution', 19: 'Dangerous curve to the left', 20: 'Dangerous curve to the right',
    21: 'Double curve', 22: 'Bumpy road', 23: 'Slippery road', 24: 'Road narrows on the right',
    25: 'Road work', 26: 'Traffic signals', 27: 'Pedestrians', 28: 'Children crossing',
    29: 'Bicycles crossing', 30: 'Beware of ice/snow', 31: 'Wild animals crossing',
    32: 'End of all speed and passing limits', 33: 'Turn right ahead', 34: 'Turn left ahead',
    35: 'Ahead only', 36: 'Go straight or right', 37: 'Go straight or left', 38: 'Keep right',
    39: 'Keep left', 40: 'Roundabout mandatory', 41: 'End of no passing',
    42: 'End of no passing by vehicles over 3.5 metric tons'
}

def run():
    st.header("🚦 Real-Time Traffic Sign Recognition")
    st.warning("Live webcam may not work on Streamlit Cloud. Test locally.")

    start = st.button("Start Webcam")
    stop = st.checkbox("Stop Webcam")

    if start:
        stframe = st.empty()
        model_path = 'models/traffic_model_pytorch.pth'
        model = load_model(model_path)
        model = model.to(DEVICE)

        cap = cv2.VideoCapture(0)

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to access webcam.")
                break

            input_tensor = preprocess_image_pytorch(frame).to(DEVICE)

            with torch.no_grad():
                output = model(input_tensor)
                _, predicted = torch.max(output, 1)
                confidence = torch.softmax(output, dim=1).max().item() * 100

            class_id = predicted.item()
            label = f"{classes[class_id]} ({confidence:.2f}%)"

            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2, cv2.LINE_AA)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)

        cap.release()
        st.success("✅ Webcam stopped.")
