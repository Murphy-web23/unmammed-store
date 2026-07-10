from __future__ import annotations

import math
import os
import tempfile
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
ITEM_ROOT = PROJECT_ROOT / "src" / "item"
MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
MEDIAPIPE_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "mediapipe"
IMAGE_EMBEDDER_MODEL = MEDIAPIPE_DIR / "image_embedder.tflite"
IMAGE_EMBEDDER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_embedder/"
    "mobilenet_v3_small/image_embedder/float16/latest/image_embedder.tflite"
)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MEDIAPIPE_MATCH_THRESHOLD = 0.25
MEDIAPIPE_MIN_MARGIN = 0.04
TOP_MATCHES = 3
EMBEDDING_WEIGHT = 0.82
COLOR_WEIGHT = 0.18
UMBRELLA_COLOR_WEIGHT = 0.22
ITEM_SCAN_WINDOW = "item_scan"
UMBRELLA_GREEN_CLASS = "umbrella_green"
UMBRELLA_TIFFANY_CLASS = "umbrella_tiffany"
OWALA_CLASS = "owala"
UMBRELLA_GREEN_HUE_CENTER = 58.0  #調綠綠閥值
UMBRELLA_TIFFANY_HUE_CENTER = 92.0
UMBRELLA_HUE_BOUNDARY_MARGIN = 6.0
GREEN_BIAS_TRIGGER = 0.14
GREEN_BIAS_REORDER_MARGIN = 0.06
GREEN_BIAS_BOOST = 0.14


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
    embedding: list[float]
    color_hist: Any
    hue_center: float
    green_ratio: float
    cyan_ratio: float
    dark_green_ratio: float


@dataclass
class MatchScore:
    total: float
    embedding: float
    color: float
    umbrella_tone: float


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


def _load_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
        from mediapipe.tasks import python  # type: ignore
        from mediapipe.tasks.python import vision  # type: ignore

        return mp, python, vision, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, None, None, f"MediaPipe 載入失敗: {exc}"


def ensure_image_embedder_model() -> str | None:
    if IMAGE_EMBEDDER_MODEL.exists():
        return None
    try:
        urllib.request.urlretrieve(IMAGE_EMBEDDER_MODEL_URL, IMAGE_EMBEDDER_MODEL)
        return None
    except Exception as exc:
        return (
            "找不到 MediaPipe ImageEmbedder 模型，且自動下載失敗。\n"
            f"請手動下載到: {IMAGE_EMBEDDER_MODEL}\n"
            f"下載網址: {IMAGE_EMBEDDER_MODEL_URL}\n"
            f"錯誤: {exc}"
        )


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
    color_mask = cv2.inRange(saturation, 25, 255)
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
        if box_width * box_height > image_area * 0.9:
            continue
        boxes.append((x, y, x + box_width, y + box_height))

    if not boxes:
        return _center_crop(image)

    left = min(box[0] for box in boxes)
    top = min(box[1] for box in boxes)
    right = max(box[2] for box in boxes)
    bottom = max(box[3] for box in boxes)
    padding = int(max(right - left, bottom - top) * 0.14)
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(width, right + padding)
    bottom = min(height, bottom + padding)
    if (right - left) * (bottom - top) < image_area * 0.01:
        return _center_crop(image)
    return image[top:bottom, left:right]


def _prepared_image(cv2: Any, image: Any, max_size: int = 700) -> Any:
    resized = _resize_for_feature(cv2, image, max_size=max_size)
    return _foreground_crop(cv2, resized)


def _extract_color_hist(cv2: Any, image: Any) -> Any:
    target = cv2.resize(_prepared_image(cv2, image, max_size=420), (192, 192))
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 16, 8], [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist


def _circular_hue_distance(left: float, right: float) -> float:
    delta = (right - left + 90.0) % 180.0 - 90.0
    return delta


def _circular_midpoint(left: float, right: float) -> float:
    midpoint = left + (_circular_hue_distance(left, right) / 2.0)
    return midpoint % 180.0


def _extract_hue_center(cv2: Any, image: Any) -> float:
    target = _prepared_image(cv2, image, max_size=420)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].astype(float)
    saturation = hsv[:, :, 1].astype(float)
    value = hsv[:, :, 2].astype(float)
    weight = saturation * value
    valid = weight > 0
    if not valid.any():
        return 0.0

    radians = (hue[valid] / 180.0) * (math.pi * 2.0)
    weights = weight[valid]
    x = float(sum(math.cos(angle) * weight for angle, weight in zip(radians, weights)))
    y = float(sum(math.sin(angle) * weight for angle, weight in zip(radians, weights)))
    if x == 0.0 and y == 0.0:
        return float(hue[valid].mean())
    angle = math.atan2(y, x)
    if angle < 0.0:
        angle += math.pi * 2.0
    return (angle / (math.pi * 2.0)) * 180.0


def _extract_green_cyan_ratio(cv2: Any, image: Any) -> tuple[float, float]:
    target = _prepared_image(cv2, image, max_size=420)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    valid_mask = (saturation >= 25) & (value >= 18)
    valid_count = int(valid_mask.sum())
    if valid_count <= 0:
        return 0.0, 0.0

    green_mask = valid_mask & (hue >= 30) & (hue <= 92)
    cyan_mask = valid_mask & (hue >= 76) & (hue <= 108)
    return float(green_mask.sum() / valid_count), float(cyan_mask.sum() / valid_count)


def _extract_dark_green_ratio(cv2: Any, image: Any) -> float:
    target = _prepared_image(cv2, image, max_size=420)
    hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    dark_green_mask = (hue >= 32) & (hue <= 88) & (saturation >= 28) & (value >= 12)
    total_mask = (saturation >= 18) & (value >= 10)
    total_count = int(total_mask.sum())
    if total_count <= 0:
        return 0.0
    return float(dark_green_mask.sum() / total_count)


def _color_similarity(cv2: Any, left: Any, right: Any) -> float:
    corr = cv2.compareHist(left, right, cv2.HISTCMP_CORREL)
    corr_score = max(0.0, min(1.0, (float(corr) + 1.0) / 2.0))
    distance = cv2.compareHist(left, right, cv2.HISTCMP_BHATTACHARYYA)
    distance_score = max(0.0, min(1.0, 1.0 - float(distance)))
    return (corr_score * 0.42) + (distance_score * 0.58)


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def _extract_embedding(
    cv2: Any,
    mp: Any,
    embedder: Any,
    image: Any,
) -> list[float] | None:
    try:
        import numpy as np
    except Exception:
        return None

    target = _prepared_image(cv2, image)
    crops = [
        target,
        _center_crop(target, ratio=0.84),
        _center_crop(target, ratio=0.66),
    ]

    vectors: list[Any] = []
    for crop in crops:
        if crop is None or crop.size == 0:
            continue
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = embedder.embed(mp_image)
        if not result.embeddings:
            continue
        vector = result.embeddings[0].embedding
        if vector is None:
            continue
        vector_array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector_array.size == 0:
            continue
        vectors.append(vector_array)

    if not vectors:
        return None

    average = sum(vectors) / float(len(vectors))
    norm = float(np.linalg.norm(average))
    if norm == 0.0:
        return None
    return (average / norm).tolist()


def _extract_item_feature(
    cv2: Any,
    mp: Any,
    embedder: Any,
    image_path: Path,
    class_name: str,
) -> ItemFeature | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    embedding = _extract_embedding(cv2, mp, embedder, image)
    if embedding is None:
        return None

    hue_center = _extract_hue_center(cv2, image)
    green_ratio, cyan_ratio = _extract_green_cyan_ratio(cv2, image)
    dark_green_ratio = _extract_dark_green_ratio(cv2, image)

    return ItemFeature(
        class_name=class_name,
        image_path=image_path,
        embedding=embedding,
        color_hist=_extract_color_hist(cv2, image),
        hue_center=hue_center,
        green_ratio=green_ratio,
        cyan_ratio=cyan_ratio,
        dark_green_ratio=dark_green_ratio,
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


def _load_reference_features(cv2: Any, mp: Any, embedder: Any) -> list[ItemFeature]:
    features: list[ItemFeature] = []
    for class_name, image_path in _reference_image_paths():
        feature = _extract_item_feature(cv2, mp, embedder, image_path, class_name)
        if feature is not None:
            features.append(feature)
    return features


def _class_hue_centers(references: list[ItemFeature]) -> dict[str, float]:
    grouped: dict[str, list[float]] = {}
    for feature in references:
        grouped.setdefault(feature.class_name, []).append(feature.hue_center)

    centers: dict[str, float] = {}
    for class_name, values in grouped.items():
        if not values:
            continue
        radians = [(value / 180.0) * (math.pi * 2.0) for value in values]
        x = sum(math.cos(angle) for angle in radians)
        y = sum(math.sin(angle) for angle in radians)
        if x == 0.0 and y == 0.0:
            centers[class_name] = sum(values) / len(values)
            continue
        angle = math.atan2(y, x)
        if angle < 0.0:
            angle += math.pi * 2.0
        centers[class_name] = (angle / (math.pi * 2.0)) * 180.0
    return centers


def _umbrella_hue_bias(query_hue: float, green_hue: float, tiffany_hue: float) -> tuple[float, float]:
    green_hue = green_hue or UMBRELLA_GREEN_HUE_CENTER
    tiffany_hue = tiffany_hue or UMBRELLA_TIFFANY_HUE_CENTER

    distance_to_green = abs(_circular_hue_distance(query_hue, green_hue))
    distance_to_tiffany = abs(_circular_hue_distance(query_hue, tiffany_hue))
    boundary = _circular_midpoint(green_hue, tiffany_hue)
    distance_to_boundary = abs(_circular_hue_distance(query_hue, boundary))

    if distance_to_boundary <= UMBRELLA_HUE_BOUNDARY_MARGIN:
        blend = max(0.0, min(1.0, distance_to_boundary / max(UMBRELLA_HUE_BOUNDARY_MARGIN, 1.0)))
        if distance_to_green < distance_to_tiffany:
            return 0.65 + (0.35 * (1.0 - blend)), 0.35 * blend
        return 0.35 * blend, 0.65 + (0.35 * (1.0 - blend))

    gap = abs(distance_to_green - distance_to_tiffany)
    if gap < 1.0:
        return 0.5, 0.5
    if distance_to_green < distance_to_tiffany:
        return 1.0, 0.0
    return 0.0, 1.0


def _compare_features(cv2: Any, query: ItemFeature, reference: ItemFeature) -> MatchScore:
    embedding_score = _cosine_similarity(query.embedding, reference.embedding)
    color_score = _color_similarity(cv2, query.color_hist, reference.color_hist)
    umbrella_tone_score = 0.5
    if reference.class_name in {UMBRELLA_GREEN_CLASS, UMBRELLA_TIFFANY_CLASS}:
        query_bias = (query.green_ratio + query.dark_green_ratio) - (query.cyan_ratio * 1.15)
        reference_bias = (reference.green_ratio + reference.dark_green_ratio) - (reference.cyan_ratio * 1.15)
        umbrella_tone_score = max(0.0, min(1.0, 1.0 - (abs(query_bias - reference_bias) * 2.0)))
    total = (embedding_score * EMBEDDING_WEIGHT) + (color_score * COLOR_WEIGHT)
    if reference.class_name in {UMBRELLA_GREEN_CLASS, UMBRELLA_TIFFANY_CLASS}:
        total = (total * (1.0 - UMBRELLA_COLOR_WEIGHT)) + (umbrella_tone_score * UMBRELLA_COLOR_WEIGHT)
    return MatchScore(
        total=max(0.0, min(1.0, total)),
        embedding=embedding_score,
        color=color_score,
        umbrella_tone=umbrella_tone_score,
    )


def _apply_green_bias_rerank(
    query: ItemFeature,
    ranked: list[ClassMatch],
    hue_centers: dict[str, float],
) -> list[ClassMatch]:
    if len(ranked) < 2:
        return ranked

    green_match = next((match for match in ranked if match.class_name == UMBRELLA_GREEN_CLASS), None)
    tiffany_match = next((match for match in ranked if match.class_name == UMBRELLA_TIFFANY_CLASS), None)
    if green_match is None or tiffany_match is None:
        return ranked

    query_hue = query.hue_center
    if query_hue == 0.0:
        return ranked

    green_hue = hue_centers.get(UMBRELLA_GREEN_CLASS)
    tiffany_hue = hue_centers.get(UMBRELLA_TIFFANY_CLASS)
    if green_hue is None or tiffany_hue is None:
        return ranked

    green_bias, tiffany_bias = _umbrella_hue_bias(query_hue, green_hue, tiffany_hue)
    if green_bias == tiffany_bias == 0.5:
        return ranked

    if query.dark_green_ratio >= 0.18:
        green_bias = min(1.0, green_bias + 0.25)
        tiffany_bias = max(0.0, tiffany_bias - 0.25)

    boosted: list[ClassMatch] = []
    for match in ranked:
        if match.class_name == UMBRELLA_GREEN_CLASS:
            boosted_score = match.score + (green_bias * GREEN_BIAS_BOOST)
            boosted.append(
                ClassMatch(
                    class_name=match.class_name,
                    score=max(0.0, min(1.0, boosted_score)),
                    reference_path=match.reference_path,
                    detail=match.detail,
                )
            )
        elif match.class_name == UMBRELLA_TIFFANY_CLASS:
            boosted_score = match.score + (tiffany_bias * GREEN_BIAS_BOOST)
            boosted.append(
                ClassMatch(
                    class_name=match.class_name,
                    score=max(0.0, min(1.0, boosted_score)),
                    reference_path=match.reference_path,
                    detail=match.detail,
                )
            )
        else:
            boosted.append(match)
    return sorted(boosted, key=lambda item: item.score, reverse=True)


def _rank_classes(cv2: Any, query: ItemFeature, references: list[ItemFeature]) -> list[ClassMatch]:
    scores_by_class: dict[str, list[tuple[MatchScore, Path]]] = {}
    for reference in references:
        score = _compare_features(cv2, query, reference)
        scores_by_class.setdefault(reference.class_name, []).append((score, reference.image_path))

    ranked: list[ClassMatch] = []
    for class_name, scores in scores_by_class.items():
        ordered = sorted(scores, key=lambda item: item[0].total, reverse=True)
        top_scores = ordered[:TOP_MATCHES]
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
    return f"語意特徵 {detail.embedding:.2f}, 色彩 {detail.color:.2f}"


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
    cv2, cv2_error = _load_cv2()
    if cv2_error:
        return DetectionResult(False, message=cv2_error)

    if find_item_reference_root() is None:
        return DetectionResult(False, message="找不到商品參考資料夾，請確認 src/item/ 是否存在。")

    model_error = ensure_image_embedder_model()
    if model_error:
        return DetectionResult(False, message=model_error)

    mp, mp_tasks, vision, mp_error = _load_mediapipe()
    if mp_error:
        return DetectionResult(False, message=mp_error)

    image = cv2.imread(str(image_path))
    if image is None:
        return DetectionResult(False, message="商品影像讀取失敗，請重新掃描。")

    try:
        base_options = mp_tasks.BaseOptions(model_asset_path=str(IMAGE_EMBEDDER_MODEL))
        options = vision.ImageEmbedderOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            l2_normalize=True,
            quantize=False,
        )
        with vision.ImageEmbedder.create_from_options(options) as embedder:
            query_embedding = _extract_embedding(cv2, mp, embedder, image)
            if query_embedding is None:
                return DetectionResult(False, message="無法從掃描畫面擷取商品特徵，請重新掃描。")
            query_green_ratio, query_cyan_ratio = _extract_green_cyan_ratio(cv2, image)
            query_dark_green_ratio = _extract_dark_green_ratio(cv2, image)
            query_hue_center = _extract_hue_center(cv2, image)

            query = ItemFeature(
                class_name="__query__",
                image_path=image_path,
                embedding=query_embedding,
                color_hist=_extract_color_hist(cv2, image),
                hue_center=query_hue_center,
                green_ratio=query_green_ratio,
                cyan_ratio=query_cyan_ratio,
                dark_green_ratio=query_dark_green_ratio,
            )
            references = _load_reference_features(cv2, mp, embedder)
    except Exception as exc:
        return DetectionResult(False, message=f"MediaPipe 商品特徵擷取失敗: {exc}")

    if not references:
        return DetectionResult(False, message="src/item/ 沒有可用的商品參考照片。")

    hue_centers = _class_hue_centers(references)
    ranked = _rank_classes(cv2, query, references)
    ranked = _apply_green_bias_rerank(query, ranked, hue_centers)
    if not ranked:
        return DetectionResult(False, message="src/item/ 沒有可用的商品參考照片。")

    best = ranked[0]
    runner_up = ranked[1] if len(ranked) > 1 else None
    margin = best.score - (runner_up.score if runner_up else 0.0)

    if best.score < MEDIAPIPE_MATCH_THRESHOLD:
        return DetectionResult(
            False,
            message="商品辨識失敗，請讓商品置中、靠近鏡頭後重新掃描。",
        )

    if runner_up and margin < MEDIAPIPE_MIN_MARGIN:
        return DetectionResult(
            False,
            message="商品辨識不夠明確，請調整角度或距離後再掃描。",
        )

    return DetectionResult(
        True,
        class_name=best.class_name,
        confidence=best.score,
        message=f"MediaPipe 商品辨識成功，商品: {best.class_name}，分數 {best.score:.2f}。",
        model_path=str(IMAGE_EMBEDDER_MODEL),
    )


def scan_item() -> DetectionResult:
    image_path, message = capture_product_image()
    if image_path is None:
        return DetectionResult(False, message=message)
    return detect_from_image(image_path)
