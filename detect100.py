import torch
import cv2
import os
import pandas as pd
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO("last (8).pt")  # Ensure this model is trained for both detection and classification

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model.to(device)

def detect_and_classify_yolo(directory_path, output_csv="results.csv"):
    results_list = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read {filename}!")
                continue
            
            # Resize image to 128x128
            image = cv2.resize(image, (128, 128))
            
            # Perform YOLO inference with enforced image size
            results = yolo_model.predict(image, imgsz=128)
            
            for result in results:
                for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    gender = "Female" if cls == 0 else "Male"  # Adjust based on YOLO class mapping
                    results_list.append([filename, gender])
                    print(f"{filename}: {gender}")
            
    # Save results to CSV
    df = pd.DataFrame(results_list, columns=["Image Name", "Predicted Gender"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Example Usage
detect_and_classify_yolo("testing")
