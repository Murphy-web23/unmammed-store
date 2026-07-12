from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ITEM_ROOT = PROJECT_ROOT / "src" / "item"
MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
OPENCV_MATCH_THRESHOLD = 0.4
OPENCV_MIN_MARGIN = 0.06
OPENCV_TOP_MATCHES = 3
COLOR_WEIGHT = 0.3
EDGE_WEIGHT = 0.22
SHAPE_WEIGHT = 0.18
APPEARANCE_WEIGHT = 0.2
ORB_WEIGHT = 0.1
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
    green_hist: Any
    edge_vector: Any
    shape_vector: Any
    shape_label: str
    appearance_vector: Any
    descriptors: Any


@dataclass
class MatchScore:
    total: float
    color: float
    edge: float
    shape: float
    appearance: float
    orb: float


@dataclass
class ClassMatch:
    class_name: str
    score: float
    reference_path: Path
    detail: MatchScore


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


def _center_crop(image: Any, ratio: float = 0.9) -> Any:
    height, width = image.shape[:2]
    crop_width = max(1, int(width * ratio))
    crop_height = max(1, int(height * ratio))
    left = max(0, (width - crop_width) // 2)
    top = max(0, (height - crop_height) // 2)
    return image[top : top + crop_height, left : left + crop_width]


def _foreground_mask(cv2: Any, image: Any) -> Any:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    color_mask = cv2.inRange(saturation, 35, 255)
    dark_mask = cv2.inRange(value, 0, 95)
    mask = cv2.bitwise_or(color_mask, dark_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def _foreground_crop(cv2: Any, image: Any) -> Any:
    mask = _foreground_mask(cv2, image)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return _center_crop(image)

    height, width = image.shape[:2]
    image_area = height * width
    boxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.002:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width * box_height > image_area * 0.82:
            continue
        boxes.append((x, y, x + box_width, y + box_height))

    if not boxes:
        return _center_crop(image)

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    padding = int(max(right - left, bottom - top) * 0.16)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if (right - left) * (bottom - top) < image_area * 0.01:
        return _center_crop(image)
    return image[top:bottom, left:right]


def _prepared_image(cv2: Any, image: Any, max_size: int = 700) -> Any:
    resized = _resize_for_feature(cv2, image, max_size=max_size)
    enhanced = _preprocess_lighting_and_color(cv2, resized)
    return _foreground_crop(cv2, enhanced)


def _dynamic_color_balance(cv2: Any, image: Any) -> Any:
    import numpy as np

    # Dynamic Color Balancing: 用灰世界假設校正 BGR 三通道，降低偏色光源影響。
    bgr = image.astype(np.float32)
    channel_means = bgr.reshape(-1, 3).mean(axis=0)
    target_mean = float(channel_means.mean())
    scales = np.where(channel_means > 1e-6, target_mean / channel_means, 1.0)
    balanced = bgr * scales.reshape(1, 1, 3)
    return np.clip(balanced, 0, 255).astype(np.uint8)


def _apply_clahe_lighting(cv2: Any, image: Any) -> Any:
    # CLAHE: 在 LAB 的 L 通道做局部對比增強，提升暗光下可辨識細節。
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_channel)
    merged = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def _enhance_hsv_green_separation(cv2: Any, image: Any) -> Any:
    import numpy as np

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    sat = np.clip((sat - 18.0) * 1.18, 0, 255)
    val = np.clip((val - 10.0) * 1.08, 0, 255)

    green_mask = (hue >= 35.0) & (hue <= 95.0)
    hue_adjust = np.zeros_like(hue)
    hue_adjust[(hue >= 40.0) & (hue < 60.0)] = -2.0
    hue_adjust[(hue >= 75.0) & (hue <= 95.0)] = 2.0
    hue = np.where(green_mask, np.clip(hue + hue_adjust, 0, 179), hue)

    hsv[:, :, 0] = hue
    hsv[:, :, 1] = sat
    hsv[:, :, 2] = val
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _preprocess_lighting_and_color(cv2: Any, image: Any) -> Any:
    balanced = _dynamic_color_balance(cv2, image)
    clahe_image = _apply_clahe_lighting(cv2, balanced)
    return _enhance_hsv_green_separation(cv2, clahe_image)


def _extract_color_hist(cv2: Any, image: Any) -> Any:
    resized = _prepared_image(cv2, image, max_size=420)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [18, 16, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    green_mask = cv2.inRange(hsv, (35, 30, 30), (95, 255, 255))
    green_hist = cv2.calcHist([hsv], [0, 1], green_mask, [24, 16], [0, 180, 0, 256])
    cv2.normalize(green_hist, green_hist)
    return hist, green_hist


def _extract_edge_vector(cv2: Any, image: Any) -> Any:
    import numpy as np

    resized = cv2.resize(_prepared_image(cv2, image, max_size=520), (192, 192))
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 55, 155)
    grid_scores = []
    for row in range(4):
        for col in range(4):
            cell = edges[row * 48 : (row + 1) * 48, col * 48 : (col + 1) * 48]
            grid_scores.append(float(cell.mean() / 255.0))
    horizontal = edges.reshape(12, 16, 192).mean(axis=(1, 2)) / 255.0
    vertical = edges.reshape(192, 12, 16).mean(axis=(0, 2)) / 255.0
    vector = np.concatenate(
        [
            np.array(grid_scores, dtype=np.float32),
            horizontal.astype(np.float32),
            vertical.astype(np.float32),
        ]
    )
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _extract_shape_features(cv2: Any, image: Any) -> tuple[Any, str]:
    import math
    import numpy as np

    prepared = _prepared_image(cv2, image, max_size=520)
    resized = cv2.resize(prepared, (240, 240))
    mask = _foreground_mask(cv2, resized)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(9, dtype=np.float32), "形狀不明"

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    if area <= 1.0 or perimeter <= 1.0:
        return np.zeros(9, dtype=np.float32), "形狀不明"

    x, y, width, height = cv2.boundingRect(contour)
    box_area = float(max(1, width * height))
    long_side = float(max(width, height))
    short_side = float(max(1, min(width, height)))
    aspect_ratio = min(1.0, long_side / short_side / 8.0)
    extent = min(1.0, area / box_area)

    hull = cv2.convexHull(contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = min(1.0, area / hull_area) if hull_area > 1.0 else 0.0
    circularity = min(1.0, (4.0 * math.pi * area) / (perimeter * perimeter))

    epsilon = 0.035 * perimeter
    approx = cv2.approxPolyDP(contour, epsilon, True)
    vertices = len(approx)
    corner_score = max(0.0, min(1.0, vertices / 10.0))
    rectangle_score = max(0.0, 1.0 - abs(vertices - 4) / 6.0) * extent

    moments = cv2.moments(contour)
    if moments["m00"] == 0:
        pointiness = 0.0
    else:
        center_x = moments["m10"] / moments["m00"]
        center_y = moments["m01"] / moments["m00"]
        points = contour.reshape(-1, 2).astype(np.float32)
        distances = np.sqrt(((points[:, 0] - center_x) ** 2) + ((points[:, 1] - center_y) ** 2))
        mean_distance = float(distances.mean()) if len(distances) else 0.0
        max_distance = float(distances.max()) if len(distances) else 0.0
        pointiness = max(0.0, min(1.0, (max_distance - mean_distance) / max(mean_distance, 1.0)))

    long_score = max(0.0, min(1.0, (long_side / short_side - 1.0) / 5.0))
    square_score = max(0.0, 1.0 - abs(width - height) / max(width, height, 1)) * rectangle_score

    vector = np.array(
        [
            aspect_ratio,
            extent,
            solidity,
            circularity,
            corner_score,
            rectangle_score,
            pointiness,
            long_score,
            square_score,
        ],
        dtype=np.float32,
    )
    norm = float(np.linalg.norm(vector))

    labels = []
    if long_score >= 0.42:
        labels.append("長形")
    if rectangle_score >= 0.42:
        labels.append("方形/矩形")
    if corner_score >= 0.35:
        labels.append("有稜角")
    if pointiness >= 0.28:
        labels.append("有尖角")
    if not labels:
        labels.append("圓滑/不規則")
    return (vector if norm == 0.0 else vector / norm), "、".join(labels)


def _extract_appearance_vector(cv2: Any, image: Any) -> Any:
    import numpy as np

    resized = cv2.resize(_prepared_image(cv2, image, max_size=520), (96, 96))
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    small_hsv = cv2.resize(hsv, (32, 32)).astype(np.float32).reshape(-1) / 255.0
    small_gray = cv2.resize(gray, (48, 48)).astype(np.float32).reshape(-1) / 255.0
    vector = np.concatenate([small_gray, small_hsv])
    vector -= float(vector.mean())
    norm = float(np.linalg.norm(vector))
    return vector if norm == 0.0 else vector / norm


def _extract_orb_descriptors(cv2: Any, image: Any) -> Any:
    resized = _prepared_image(cv2, image)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    orb = cv2.ORB_create(nfeatures=1200, scaleFactor=1.2, nlevels=8)
    _, descriptors = orb.detectAndCompute(gray, None)
    return descriptors


def _extract_item_feature(cv2: Any, image_path: Path, class_name: str) -> ItemFeature | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None
    shape_vector, shape_label = _extract_shape_features(cv2, image)
    color_hist, green_hist = _extract_color_hist(cv2, image)
    return ItemFeature(
        class_name=class_name,
        image_path=image_path,
        color_hist=color_hist,
        green_hist=green_hist,
        edge_vector=_extract_edge_vector(cv2, image),
        shape_vector=shape_vector,
        shape_label=shape_label,
        appearance_vector=_extract_appearance_vector(cv2, image),
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
    distance_total = 0.0
    for pair in matches:
        if len(pair) < 2:
            continue
        first, second = pair
        if first.distance < 0.72 * second.distance:
            good += 1
            distance_total += float(first.distance)
    if good == 0:
        return 0.0
    match_count_score = min(1.0, good / 38)
    distance_score = max(0.0, 1.0 - ((distance_total / good) / 80.0))
    return (match_count_score * 0.72) + (distance_score * 0.28)


def _cosine_similarity(left: Any, right: Any) -> float:
    import numpy as np

    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    raw = float(np.dot(left, right) / (left_norm * right_norm))
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def _vector_shape_similarity(left: Any, right: Any) -> float:
    import numpy as np

    diff = float(np.mean(np.abs(left - right)))
    return max(0.0, min(1.0, 1.0 - (diff * 3.2)))


def _compare_features(cv2: Any, query: ItemFeature, reference: ItemFeature) -> MatchScore:
    color_corr = cv2.compareHist(query.color_hist, reference.color_hist, cv2.HISTCMP_CORREL)
    color_corr_score = max(0.0, min(1.0, (float(color_corr) + 1.0) / 2.0))
    color_distance = cv2.compareHist(query.color_hist, reference.color_hist, cv2.HISTCMP_BHATTACHARYYA)
    color_distance_score = max(0.0, min(1.0, 1.0 - float(color_distance)))
    green_corr = cv2.compareHist(query.green_hist, reference.green_hist, cv2.HISTCMP_CORREL)
    green_corr_score = max(0.0, min(1.0, (float(green_corr) + 1.0) / 2.0))
    green_distance = cv2.compareHist(query.green_hist, reference.green_hist, cv2.HISTCMP_BHATTACHARYYA)
    green_distance_score = max(0.0, min(1.0, 1.0 - float(green_distance)))
    green_score = (green_corr_score * 0.45) + (green_distance_score * 0.55)
    color_score = (color_corr_score * 0.27) + (color_distance_score * 0.43) + (green_score * 0.30)
    edge_score = _vector_shape_similarity(query.edge_vector, reference.edge_vector)
    shape_score = _cosine_similarity(query.shape_vector, reference.shape_vector)
    appearance_score = _cosine_similarity(query.appearance_vector, reference.appearance_vector)
    orb_score = _orb_similarity(cv2, query.descriptors, reference.descriptors)
    total = (
        (color_score * COLOR_WEIGHT)
        + (edge_score * EDGE_WEIGHT)
        + (shape_score * SHAPE_WEIGHT)
        + (appearance_score * APPEARANCE_WEIGHT)
        + (orb_score * ORB_WEIGHT)
    )
    if orb_score == 0.0:
        total *= 0.9
    return MatchScore(
        total=max(0.0, min(1.0, total)),
        color=color_score,
        edge=edge_score,
        shape=shape_score,
        appearance=appearance_score,
        orb=orb_score,
    )


def _rank_classes(cv2: Any, query: ItemFeature, references: list[ItemFeature]) -> list[ClassMatch]:
    scores_by_class: dict[str, list[tuple[MatchScore, Path]]] = {}
    for reference in references:
        score = _compare_features(cv2, query, reference)
        scores_by_class.setdefault(reference.class_name, []).append((score, reference.image_path))

    ranked: list[ClassMatch] = []
    for class_name, scores in scores_by_class.items():
        ordered = sorted(scores, key=lambda item: item[0].total, reverse=True)
        top_scores = ordered[:OPENCV_TOP_MATCHES]
        average_top = sum(score.total for score, _ in top_scores) / len(top_scores)
        class_score = (ordered[0][0].total * 0.68) + (average_top * 0.32)
        ranked.append(
            ClassMatch(
                class_name=class_name,
                score=class_score,
                reference_path=ordered[0][1],
                detail=ordered[0][0],
            )
        )
    return sorted(ranked, key=lambda item: item.score, reverse=True)


def _format_detail(detail: MatchScore) -> str:
    return (
        f"顏色 {detail.color:.2f}, 邊緣 {detail.edge:.2f}, "
        f"形狀 {detail.shape:.2f}, "
        f"外觀 {detail.appearance:.2f}, 細節 {detail.orb:.2f}"
    )


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
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                return None, "攝影機讀取失敗。"
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
            if key == 32:
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

    ranked = _rank_classes(cv2, query, references)
    if not ranked:
        return DetectionResult(False, message="src/item/ 沒有可用的商品參考照片。")

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = best.score - (runner_up.score if runner_up else 0.0)

    if best.score < OPENCV_MATCH_THRESHOLD:
        return DetectionResult(
            False,
            message=(
                f"商品辨識失敗，最高相似度 {best.score:.2f}，低於門檻 "
                f"{OPENCV_MATCH_THRESHOLD:.2f}。\n"
                f"第一名: {best.class_name} ({_format_detail(best.detail)})\n"
                "請讓商品置中、靠近鏡頭，背景保持單純後重新掃描。"
            ),
        )

    if runner_up and margin < OPENCV_MIN_MARGIN:
        return DetectionResult(
            False,
            class_name=best.class_name,
            confidence=best.score,
            message=(
                f"商品辨識不確定，第一名 {best.class_name} {best.score:.2f}，"
                f"第二名 {runner_up.class_name} {runner_up.score:.2f}，差距 {margin:.2f}。\n"
                f"第一名特徵值: {_format_detail(best.detail)}\n"
                f"第二名特徵值: {_format_detail(runner_up.detail)}\n"
                "這通常代表參考照片太少、角度太像，或背景干擾太高；請補拍更多角度後再掃描。"
            ),
        )

    return DetectionResult(
        True,
        class_name=best.class_name,
        confidence=best.score,
        message=(
            f"OpenCV 商品辨識成功，分數 {best.score:.2f}。\n"
            f"{_format_detail(best.detail)}\n"
            f"參考圖: {best.reference_path}"
        ),
        model_path=str(ITEM_ROOT),
    )


def scan_item() -> DetectionResult:
    image_path, message = capture_product_image()
    if image_path is None:
        return DetectionResult(False, message=message)
    return detect_from_image(image_path)


def capture_product_reference_photos(class_name: str, count: int = 6) -> tuple[bool, str, Path | None]:
    cv2, error = _load_cv2()
    if error:
        return False, error, None

    safe_class_name = class_name.strip()
    if not safe_class_name:
        return False, "class_name 不可為空。", None

    target_folder = ITEM_ROOT / safe_class_name
    target_folder.mkdir(parents=True, exist_ok=True)

    prompts = [
        "正面",
        "左前 45 度",
        "右前 45 度",
        "左側",
        "右側",
        "俯視或傾斜",
    ]
    if count != 6:
        prompts = [f"角度 {index + 1}" for index in range(count)]

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return False, "攝影機無法開啟，請確認鏡頭是否被其他程式占用。", None

    try:
        for index in range(count):
            while True:
                ok, frame = camera.read()
                if not ok:
                    return False, "攝影機讀取失敗。", None
                cv2.putText(
                    frame,
                    f"{prompts[index]}  SPACE拍照 ({index + 1}/{count})",
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "ESC 取消",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow("商品建檔拍照 - ESC 取消", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return False, "使用者取消商品拍照。", None
                if key == 32:
                    cv2.imwrite(str(target_folder / f"{index + 1}.jpg"), frame)
                    break

        return True, f"已完成 {count} 張商品參考照片拍攝。", target_folder
    except Exception as exc:
        return False, f"商品拍照流程發生錯誤: {exc}", None
    finally:
        camera.release()
        cv2.destroyAllWindows()
