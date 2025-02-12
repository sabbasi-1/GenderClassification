import torch
import cv2
import numpy as np
import torchvision.transforms as transforms
from ultralytics import YOLO
import torchvision.models as models
import torch.nn as nn
from PIL import Image

yolo_model = YOLO("last.pt")  # Change to your YOLO model path
class GenderClassifier(nn.Module):
    def __init__(self):
        super(GenderClassifier, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)  # Binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.backbone(x)
        return self.sigmoid(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gender_model = GenderClassifier().to(device)
gender_model.load_state_dict(torch.load("gender_classifier_resnet18.pth", map_location=device))
gender_model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def detect_faces_and_classify(image_path):
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return

    results = yolo_model(image)
    
    for result in results:
        for box in result.boxes.xyxy.cpu().numpy():  # Extract bounding boxes
            x1, y1, x2, y2 = map(int, box[:4])  # Get face coordinates
            face = image[y1:y2, x1:x2]
            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                output = gender_model(face_tensor).item()
            gender = "Male" if output > 0.5 else "Female"
            confidence = output if output > 0.5 else 1 - output
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{gender} ({confidence:.2f})"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            print(f"Predicted Gender: {gender}, Confidence: {confidence:.2f}")
    output_path = "output.jpg"
    cv2.imwrite(output_path, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
detect_faces_and_classify("pic5.jpg")
