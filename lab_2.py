from ultralytics import YOLO
import cv2
import glob
import os

model = YOLO("yolov8n.pt")

def train_model():
    model.train(
        data='./data.yaml',
        epochs=20,         # сначала меньше, чтобы протестировать
        imgsz=640,         # размер изображений
        batch=8,           # размер батча
        device='cpu'           
    )

def test_model():
    test_images = glob.glob('./archive/test/*.jpg')
    output_dir = './output'
    os.makedirs(output_dir, exist_ok=True)
    for fname in test_images:
        img = cv2.imread(fname)
        results = model(img)
        annotated_frame = results[0].plot()
        out_path = os.path.join(output_dir, f'detection_{os.path.basename(fname)}')
        cv2.imwrite(out_path, annotated_frame)
        print(f"Saved: {out_path}")

        # Выводим найденные объекты
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls_id]
                print(f"Обнаружено: {label} ({conf:.2%}) на {fname}")



def main():
    #train_model()
    test_model()

if __name__ == "__main__":
    main()