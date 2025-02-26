import torch
import cv2
import os
import pandas as pd
from ultralytics import YOLO

yolo_model = YOLO("model.pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model.to(device)

def detect_and_classify_yolo(directory_path, output_csv="results_2.csv"):
    results_list = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read {filename}!")
                continue
            
            results = yolo_model.predict(image, imgsz=128,conf=0.35)
            
            for result in results:
                for box, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box[:4])
                    gender = "Female" if cls == 0 else "Male"  
                    results_list.append([filename, gender])
                    print(f"{filename}: {gender}")
            
    df = pd.DataFrame(results_list, columns=["Image Name", "Predicted Gender"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
detect_and_classify_yolo("folder_name")
