from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
YOLO_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "ultralytics"
MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
YOLO_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(YOLO_CONFIG_DIR))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))
MODEL_PATHS = [
    PROJECT_ROOT / "model" / "best.pt",
    PROJECT_ROOT / "model" / "runs" / "classify" / "train" / "weights" / "best.pt",
]


@dataclass
class DetectionResult:
    success: bool
    class_name: str = ""
    confidence: float = 0.0
    message: str = ""
    model_path: str = ""


def find_model_path() -> Path | None:
    for path in MODEL_PATHS:
        if path.exists():
            return path
    return None


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV 載入失敗: {exc}"


def _load_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"Ultralytics YOLO 載入失敗: {exc}"


def capture_product_image() -> tuple[Path | None, str]:
    cv2, error = _load_cv2()
    if error:
        return None, error

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return None, "攝影機無法開啟，請確認鏡頭是否被其他程式占用。"

    temp_file = Path(tempfile.gettempdir()) / "unmanned_store_product_scan.jpg"
    start = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                return None, "攝影機讀取失敗。"
            frame = cv2.flip(frame, 1)
            cv2.putText(
                frame,
                "Put item in view. SPACE scan, ESC cancel",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )
            cv2.imshow("掃描商品", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None, "使用者取消商品掃描。"
            if key == 32 or time.time() - start > 8:
                cv2.imwrite(str(temp_file), frame)
                return temp_file, "已拍攝商品影像。"
    except Exception as exc:
        return None, f"掃描商品時發生錯誤: {exc}"
    finally:
        camera.release()
        cv2.destroyAllWindows()


def detect_from_image(image_path: Path) -> DetectionResult:
    model_path = find_model_path()
    if model_path is None:
        return DetectionResult(False, message="找不到 YOLO 模型，請確認 best.pt 路徑。")

    YOLO, error = _load_yolo()
    if error:
        return DetectionResult(False, message=error, model_path=str(model_path))

    try:
        model = YOLO(str(model_path))
        results = model(str(image_path), verbose=False)
        if not results:
            return DetectionResult(False, message="商品辨識失敗，請重新掃描。")

        result = results[0]
        names = result.names or {}

        # Classify model
        if getattr(result, "probs", None) is not None and result.probs is not None:
            top_index = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            class_name = str(names.get(top_index, top_index))
            return DetectionResult(
                True,
                class_name=class_name,
                confidence=confidence,
                message="YOLO 分類辨識成功。",
                model_path=str(model_path),
            )

        # Detect model
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            best_box = max(boxes, key=lambda box: float(box.conf[0]))
            class_index = int(best_box.cls[0])
            confidence = float(best_box.conf[0])
            class_name = str(names.get(class_index, class_index))
            return DetectionResult(
                True,
                class_name=class_name,
                confidence=confidence,
                message="YOLO 偵測辨識成功。",
                model_path=str(model_path),
            )

        return DetectionResult(False, message="商品辨識失敗，請重新掃描。")
    except Exception as exc:
        return DetectionResult(False, message=f"YOLO 辨識失敗: {exc}", model_path=str(model_path))


def scan_item() -> DetectionResult:
    image_path, message = capture_product_image()
    if image_path is None:
        return DetectionResult(False, message=message)
    return detect_from_image(image_path)
