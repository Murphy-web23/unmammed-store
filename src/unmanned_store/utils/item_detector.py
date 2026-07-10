from __future__ import annotations

import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ITEM_ROOT = PROJECT_ROOT / "src" / "item"
MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OPENCV_MATCH_THRESHOLD = 0.34
ITEM_SCAN_WINDOW = "item_scan"


@dataclass
class DetectionResult:
    success: bool
    class_name: str = ""
    confidence: float = 0.0
    message: str = ""
    model_path: str = ""


@dataclass
class ItemFeature:
    class_name: str
    image_path: Path
    color_hist: Any
    descriptors: Any


def find_item_reference_root() -> Path | None:
    if ITEM_ROOT.exists() and any(folder.is_dir() for folder in ITEM_ROOT.iterdir()):
        return ITEM_ROOT
    return None


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV 載入失敗: {exc}"


def _resize_for_feature(cv2: Any, image: Any, max_size: int = 700) -> Any:
    height, width = image.shape[:2]
    scale = min(1.0, max_size / max(height, width))
    if scale >= 1.0:
        return image
    return cv2.resize(image, (int(width * scale), int(height * scale)))


def _extract_color_hist(cv2: Any, image: Any) -> Any:
    resized = _resize_for_feature(cv2, image, max_size=360)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _extract_orb_descriptors(cv2: Any, image: Any) -> Any:
    resized = _resize_for_feature(cv2, image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    orb = cv2.ORB_create(nfeatures=900)
    _, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def _extract_item_feature(cv2: Any, image_path: Path, class_name: str) -> ItemFeature | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    return ItemFeature(
        class_name=class_name,
        image_path=image_path,
        color_hist=_extract_color_hist(cv2, image),
        descriptors=_extract_orb_descriptors(cv2, image),
    )


def _reference_image_paths() -> list[tuple[str, Path]]:
    if not ITEM_ROOT.exists():
        return []
    paths: list[tuple[str, Path]] = []
    for folder in sorted(ITEM_ROOT.iterdir()):
        if not folder.is_dir():
            continue
        for image_path in sorted(folder.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append((folder.name, image_path))
    return paths


def _load_reference_features(cv2: Any) -> list[ItemFeature]:
    features: list[ItemFeature] = []
    for class_name, image_path in _reference_image_paths():
        feature = _extract_item_feature(cv2, image_path, class_name)
        if feature is not None:
            features.append(feature)
    return features


def _orb_similarity(cv2: Any, left: Any, right: Any) -> float:
    if left is None or right is None or len(left) < 2 or len(right) < 2:
        return 0.0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(left, right, k=2)
    good = 0
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.75 * second.distance:
            good += 1
    return min(1.0, good / 35)


def _compare_features(cv2: Any, query: ItemFeature, reference: ItemFeature) -> float:
    color_corr = cv2.compareHist(query.color_hist, reference.color_hist, cv2.HISTCMP_CORREL)
    color_score = max(0.0, min(1.0, (float(color_corr) + 1.0) / 2.0))
    orb_score = _orb_similarity(cv2, query.descriptors, reference.descriptors)
    if orb_score == 0.0:
        return color_score * 0.55
    return (color_score * 0.42) + (orb_score * 0.58)


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
            cv2.imshow(ITEM_SCAN_WINDOW, frame)
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
    cv2, error = _load_cv2()
    if error:
        return DetectionResult(False, message=error)

    if find_item_reference_root() is None:
        return DetectionResult(False, message="找不到商品參考資料夾，請確認 src/item/ 是否存在。")

    query = _extract_item_feature(cv2, image_path, "__query__")
    if query is None:
        return DetectionResult(False, message="商品影像讀取失敗，請重新掃描。")

    references = _load_reference_features(cv2)
    if not references:
        return DetectionResult(False, message="src/item/ 沒有可用的商品參考照片。")

    scores_by_class: dict[str, list[tuple[float, Path]]] = {}
    for reference in references:
        score = _compare_features(cv2, query, reference)
        scores_by_class.setdefault(reference.class_name, []).append((score, reference.image_path))

    best_class = ""
    best_score = 0.0
    best_image = ""
    for class_name, scores in scores_by_class.items():
        ordered = sorted(scores, key=lambda item: item[0], reverse=True)
        top_scores = ordered[:3]
        average_top = sum(score for score, _ in top_scores) / len(top_scores)
        class_score = (ordered[0][0] * 0.72) + (average_top * 0.28)
        if class_score > best_score:
            best_class = class_name
            best_score = class_score
            best_image = str(ordered[0][1])

    if not best_class or best_score < OPENCV_MATCH_THRESHOLD:
        return DetectionResult(
            False,
            message=f"商品辨識失敗，最高相似度 {best_score:.2f}，請重新掃描。",
        )

    return DetectionResult(
        True,
        class_name=best_class,
        confidence=best_score,
        message=f"OpenCV 商品辨識成功，參考圖: {best_image}",
        model_path=str(ITEM_ROOT),
    )


def scan_item() -> DetectionResult:
    image_path, message = capture_product_image()
    if image_path is None:
        return DetectionResult(False, message=message)
    return detect_from_image(image_path)
