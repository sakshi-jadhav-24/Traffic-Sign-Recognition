import streamlit as st
import cv2
import torch
import numpy as np
from src.model import load_model  # Your PyTorch model loader
from src.preprocessing import preprocess_image_pytorch  # Ensure preprocessing is correctly imported

# Constants
IMG_SIZE = 32  # Define the image size expected by the model
CONFIDENCE_THRESHOLD = 70
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class labels (traffic sign classes)
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
    st.header("🎥 Real-Time Traffic Sign Recognition")
    st.warning("Live webcam access works best in local Streamlit run. Not supported on Streamlit Cloud.")
    
    # Button to start webcam
    start = st.button("Start Webcam")
    stop = st.checkbox("Stop Webcam")  # Option to stop webcam feed

    if start:
        stframe = st.empty()  # Create a placeholder to display webcam feed
        model_path = 'models/traffic_model_pytorch.pth'  # Provide the correct path to your trained model file
        model = load_model(model_path)  # Load the model using load_model function
        model = model.to(DEVICE)  # Move model to appropriate device (GPU/CPU)

        cap = cv2.VideoCapture(0)  # Start webcam feed

        while cap.isOpened() and not stop:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Failed to capture from webcam.")
                break

            # Preprocess the frame from the webcam
            img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))  # Resize image to the size expected by the model
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB format
            img = np.expand_dims(img, axis=0)  # Add batch dimension (1, IMG_SIZE, IMG_SIZE, 3)
            img = np.transpose(img, (0, 3, 1, 2))  # Rearrange dimensions from (1, 32, 32, 3) to (1, 3, 32, 32)
            img_tensor = torch.tensor(img, dtype=torch.float32).to(DEVICE)  # Convert to tensor and move to GPU/CPU

            # Model prediction
            output = model(img_tensor)  # Get model output
            _, predicted = torch.max(output, 1)  # Get predicted class index
            class_id = predicted.item()  # Convert the tensor to a Python integer
            confidence = torch.max(torch.nn.functional.softmax(output, dim=1)).item() * 100  # Confidence level
            label = f"{classes[class_id]} ({confidence:.2f}%)"  # Prepare label with class name and confidence

            # Draw the label on the frame
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB for display
            stframe.image(frame_rgb, channels="RGB", use_column_width=True)  # Display the frame in Streamlit

        cap.release()  # Release webcam when done
        st.success("✅ Webcam stopped.")
