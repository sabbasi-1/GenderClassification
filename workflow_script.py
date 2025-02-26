import torch
import cv2
import os
import pandas as pd
from ultralytics import YOLO

yolo_model = YOLO("last (8).pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model.to(device)

def detect_and_classify_females(directory_path, output_csv="results_female.csv"):
    results_list = []
    
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(directory_path, filename)
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error: Could not read {filename}!")
                continue
            
            results = yolo_model.predict(image, imgsz=128,conf=0.6)
            
            highest_confidence = 0  
            best_result = None
            contains_male = False
            
            for result in results:
                for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                    if cls == 1:  
                        contains_male = True
                        break
                    if cls == 0 and conf > highest_confidence:  
                        highest_confidence = conf
                        best_result = (filename, "Female")
                if contains_male:
                    break  
            
            if best_result is not None and not contains_male:
                results_list.append(best_result)
                print(f"{best_result[0]}: {best_result[1]}")
    
    df = pd.DataFrame(results_list, columns=["Image Name", "Predicted Gender"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
detect_and_classify_females("testing")
