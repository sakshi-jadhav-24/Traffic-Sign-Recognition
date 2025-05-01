from torchvision import transforms
from PIL import Image
import cv2

def preprocess_image_pytorch(cv2_img):
    rgb_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_img)

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    tensor_img = transform(pil_img).unsqueeze(0)
    return tensor_img