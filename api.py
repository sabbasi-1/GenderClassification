from fastapi import FastAPI, File, UploadFile
import uvicorn
import torch
import cv2
import numpy as np
from ultralytics import YOLO
import os

yolo_model = YOLO("last (8).pt")  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model.to(device)
app = FastAPI()
@app.get("/")
def read_root():
    return {"message": "FastAPI Gender Detection API is running!"}
@app.post("/detect")
async def detect_gender(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)        
        if image is None:
            return {"error": "Invalid image file"}

        results = yolo_model.predict(image, imgsz=128, conf=0.6)

        highest_confidence = 0  
        best_result = None
        contains_male = False
        
        for result in results:
            for box, cls, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()):
                if cls == 1:  # Male class detected
                    contains_male = True
                    break
                if cls == 0 and conf > highest_confidence:  # Female class detected
                    highest_confidence = conf
                    best_result = {"filename": file.filename, "predicted_gender": "Female", "confidence": float(conf)}
            
            if contains_male:
                return {"filename": file.filename, "predicted_gender": "Male", "confidence": float(conf)}
        if best_result:
            return best_result

        return {"filename": file.filename, "predicted_gender": "Unknown", "confidence": 0}

    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
