import streamlit as st
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from torch import nn

# Constants
IMG_SIZE = 32
MODEL_PATH = "models/traffic_model_pytorch.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names
CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)',
    'Speed limit (60km/h)', 'Speed limit (70km/h)', 'Speed limit (80km/h)',
    'End of speed limit (80km/h)', 'Speed limit (100km/h)', 'Speed limit (120km/h)',
    'No passing', 'No passing for vehicles over 3.5 metric tons', 'Right-of-way at the next intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Vehicles over 3.5 metric tons prohibited',
    'No entry', 'General caution', 'Dangerous curve to the left', 'Dangerous curve to the right',
    'Double curve', 'Bumpy road', 'Slippery road', 'Road narrows on the right', 'Road work',
    'Traffic signals', 'Pedestrians', 'Children crossing', 'Bicycles crossing', 'Beware of ice/snow',
    'Wild animals crossing', 'End of all speed and passing limits', 'Turn right ahead',
    'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left', 'Keep right',
    'Keep left', 'Roundabout mandatory', 'End of no passing', 'End of no passing by vehicles over 3.5 metric tons'
]

# Model class
class TrafficCNN(nn.Module):
    def __init__(self):
        super(TrafficCNN, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(512 * 4 * 4, 1024), nn.ReLU(),
            nn.Linear(1024, 43)
        )

    def forward(self, x):
        return self.net(x)

# Load model
model = TrafficCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Main app
def run():
    st.title("🚦 Traffic Sign Classifier")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        img = transform(image).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]

            class_id = int(np.argmax(probs))
            confidence = probs[class_id] * 100
            class_name = CLASS_NAMES[class_id]

        st.subheader("🎯 Prediction Result")
        st.success(f"**Class ID:** {class_id} \n\n**Class Name:** {class_name}")
        st.info(f"**Confidence Score:** {confidence:.2f}%")

        # Show Top-5 predictions
        top5 = np.argsort(probs)[-5:][::-1]
        st.subheader("📊 Top 5 Predictions:")
        for i in top5:
            st.write(f"**{i}** → {CLASS_NAMES[i]} — **{probs[i]*100:.2f}%**")

if __name__ == "__main__":
    run()
