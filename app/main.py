from fastapi import FastAPI, WebSocket
import cv2
import numpy as np
import base64
from ultralytics import YOLO

app = FastAPI()


# Load YOLOv10n model
model = YOLO("yolov8n.pt")
# Display model information (optional)
model.info()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        
        # Decode base64 image
        img_data = base64.b64decode(data.split(',')[1])
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Perform object detection
        results = model(img_rgb)
        
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Get class and confidence
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Draw label
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Encode processed image to base64
        _, buffer = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        # Send processed image back to client
        await websocket.send_text(f"data:image/jpeg;base64,{img_base64}")

@app.get("/")
async def get():
    return {"message": "WebSocket server is running"}