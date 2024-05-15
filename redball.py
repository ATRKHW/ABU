import cv2
import torch
import numpy as np

# โหลดโมเดล YOLOv8
model = torch.hub.load('ultralytics/yolov8', 'yolov8s')

# กำหนดคลาสเป้าหมาย (ลูกฟุตบอล)
classes = ['football']

# ฟังก์ชันสำหรับระบูลูกฟุตบอล
def detect_football(image):
  # แปลงภาพเป็น RGB
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  # ปรับขนาดภาพ
  image = cv2.resize(image, (640, 640))

  # แปลงภาพเป็น Tensor
  image = torch.from_numpy(image).unsqueeze(0)
  image = image.float() / 255.0

  # ทำนายผล
  results = model(image)

  # แปลงผลลัพธ์เป็น bounding boxes
  boxes = results.xyxy[0].cpu().numpy()

  # วนซ้ำ bounding boxes
  for box in boxes:
    # กรองเฉพาะคลาสเป้าหมาย
    if box[5] == classes.index('football'):
      # คำนวณค่าความแม่นยำ
      confidence = box[6]

      # แสดง bounding box และค่าความแม่นยำ
      x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
      cv2.putText(image, f"Football: {confidence:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  # แสดงผล
  cv2.imshow('Football Detection', image)

# อ่านภาพ
image = cv2.imread('football.jpg')

# ระบูลูกฟุตบอล
detect_football(image)

# รอการกดปุ่ม
cv2.waitKey(0)

# ปิดหน้าต่าง
cv2.destroyAllWindows()
