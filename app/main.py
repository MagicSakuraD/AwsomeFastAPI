import cv2
import numpy as np
import base64
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO
import asyncio

app = FastAPI()

# 加载 YOLO 模型
model = YOLO("yolov8n.pt")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            
            # 解码 Base64 图像
            header, encoded = data.split(',', 1)
            img_data = base64.b64decode(encoded)
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # 转换 BGR 到 RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 使用 YOLO 进行目标检测
            results = model(img_rgb)

            # 处理检测结果并绘制边界框
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    label = f"{model.names[cls]} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # 编码处理后的图像为 Base64
            _, buffer = cv2.imencode('.jpg', img)
            processed_image = base64.b64encode(buffer.tobytes()).decode('utf-8')
            processed_image = f"data:image/jpeg;base64,{processed_image}"

            # 发送处理后的图像回前端
            await websocket.send_text(processed_image)
    except WebSocketDisconnect:
        print("Client disconnected")

@app.get("/")
async def get():
    return {"message": "WebSocket server is running"}
