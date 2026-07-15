from pathlib import Path
import numpy as np
import cv2
import supervision as sv
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent
IMAGE_PATH = ROOT_DIR / "data" / "bus.jpg"
VIDEO_PATH = 0
MODEL_PATH = ROOT_DIR.parent / "model" / "drinks.pt"

model = YOLO(str(MODEL_PATH))
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()


# def detect_image(path: Path):
#     image = cv2.imread(str(path))
#     if image is None:
#         raise FileNotFoundError(f"Failed to load image: {path}")

#     results = model(image, verbose=False)[0]
#     detections = sv.Detections.from_ultralytics(results)

#     annotated_image = box_annotator.annotate(scene=image, detections=detections)
#     annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

#     cv2.imshow("image", annotated_image)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


def detect_video(path: Path):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {path}")

    while True:
        ret, image = cap.read()
        if not ret:
            break

        # 1. 模型推論與轉換
        results = model(image, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        # 2. 建立遮罩 (置信度 > 0.85 且是 'person')
        mask = (detections.confidence > 0.85) 
        # & (
        #     np.array([results.names[class_id] for class_id in detections.class_id]) == 'person'
        # )

        # 3. 【核心修正】直接用 mask 過濾 detections 物件
        # 這樣過濾後，框和標籤就會同步只剩下符合條件的對象
        detections = detections[mask]

        # 4. 根據過濾後的 detections 產生對應的標籤
        # 改用 results.names[class_id] 確保不會因為找不到 'class_name' 鍵值而報錯
        labels = [
            f"{results.names[class_id]} {confidence:.2f}"
            for class_id, confidence 
            in zip(detections.class_id, detections.confidence)
        ]

        # 5. 繪製畫面（把拿掉的 mask 參數移除）
        annotated_image = box_annotator.annotate(scene=image, detections=detections)
        annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

        # 6. 顯示影像
        cv2.imshow("video", annotated_image)
        if cv2.waitKey(1) == 27:  # 按 ESC 鍵退出
            break

if __name__ == "__main__":
    # 偵測單張影像
    # detect_image(IMAGE_PATH)

    # 偵測影片檔案（請確認 cv/src/data/vtest.avi 存在）
    detect_video(VIDEO_PATH)
