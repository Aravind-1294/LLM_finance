from ultralyticsplus import YOLO, render_result
import cv2

model = YOLO('foduucom/stockmarket-pattern-detection-yolov8')

model.overrides['conf'] = 0.25 
model.overrides['iou'] = 0.45  
model.overrides['agnostic_nms'] = False 
model.overrides['max_det'] = 1000 


video_path = "RELIANCE 2887.50 ▲ +1.3% Unnamed - Google Chrome 2024-03-20 23-27-39.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
