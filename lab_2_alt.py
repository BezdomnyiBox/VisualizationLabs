from ultralytics import YOLO
import cv2
import glob

model = YOLO("yolov8x.pt")



images = glob.glob('./lab_2_images/*.jpg')
for fname in images:
    img = cv2.imread(fname)
    results = model(img)

    annotated_frame = results[0].plot()
    cv2.imshow("Detection", annotated_frame)
    cv2.imwrite(f"./lab_2_images/output/detection_{fname}.jpg", annotated_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]
            print(f"Обнаружено: {label} ({conf:.2%})")


