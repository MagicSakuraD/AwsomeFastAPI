from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
from ultralytics import YOLO
import asyncio

app = FastAPI()

# Load YOLOv8n model
model = YOLO("yolov8n.pt")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Get video stream URL from client
    video_url = await websocket.receive_text()
    
    # Open video stream
    cap = cv2.VideoCapture(video_url)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model(frame_rgb)
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode processed frame to base64
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Send processed frame back to client
        await websocket.send_text(f"data:image/jpeg;base64,{frame_base64}")
        
        # Add a small delay to control frame rate
        await asyncio.sleep(0.03)  # Adjust this value to change frame rate

    cap.release()

@app.get("/")
async def get():
    return {"message": "Video streaming server is running"}