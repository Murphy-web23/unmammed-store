from __future__ import annotations

import tempfile
import time
import os
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .member_manager import FACE_ROOT, PROJECT_ROOT, Member, read_members


MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
MEDIAPIPE_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "mediapipe"
FACE_LANDMARKER_MODEL = MEDIAPIPE_DIR / "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
MEDIAPIPE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))

LANDMARK_MATCH_THRESHOLD = 0.09
APPEARANCE_MATCH_THRESHOLD = 0.78
COMBINED_MATCH_THRESHOLD = 0.65
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass
class FaceRecognitionResult:
    success: bool
    member: Member | None = None
    identity_path: str = ""
    message: str = ""


@dataclass
class FaceFeature:
    landmarks: list[float]
    appearance: list[float]


def _safe_mirror_frame(cv2: Any, frame: Any) -> Any:
    return cv2.flip(frame, 1)


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


def ensure_face_landmarker_model() -> str | None:
    if FACE_LANDMARKER_MODEL.exists():
        return None

    try:
        urllib.request.urlretrieve(FACE_LANDMARKER_MODEL_URL, FACE_LANDMARKER_MODEL)
        return None
    except Exception as exc:
        return (
            "找不到 MediaPipe FaceLandmarker 模型，且自動下載失敗。\n"
            f"請手動下載到: {FACE_LANDMARKER_MODEL}\n"
            f"下載網址: {FACE_LANDMARKER_MODEL_URL}\n"
            f"錯誤: {exc}"
        )


def capture_face_image() -> tuple[Path | None, str]:
    cv2, error = _load_cv2()
    if error:
        return None, error

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return None, "攝影機無法開啟，請確認鏡頭是否被其他程式占用。"

    temp_file = Path(tempfile.gettempdir()) / "unmanned_store_face_scan.jpg"
    start = time.time()
    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                return None, "攝影機讀取失敗。"
            frame = _safe_mirror_frame(cv2, frame)
            cv2.putText(
                frame,
                "Press SPACE to scan, ESC to cancel",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )
            cv2.imshow("掃描會員", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                return None, "使用者取消掃描。"
            if key == 32 or time.time() - start > 8:
                cv2.imwrite(str(temp_file), frame)
                return temp_file, "已拍攝人臉影像。"
    except Exception as exc:
        return None, f"掃描會員時發生錯誤: {exc}"
    finally:
        camera.release()
        cv2.destroyAllWindows()


def _normalize_landmarks(landmarks: list[Any]) -> list[float] | None:
    if not landmarks:
        return None

    points = [(float(point.x), float(point.y), float(point.z)) for point in landmarks]
    center_x = sum(point[0] for point in points) / len(points)
    center_y = sum(point[1] for point in points) / len(points)
    center_z = sum(point[2] for point in points) / len(points)
    centered = [
        (point[0] - center_x, point[1] - center_y, point[2] - center_z)
        for point in points
    ]

    min_x = min(point[0] for point in centered)
    max_x = max(point[0] for point in centered)
    min_y = min(point[1] for point in centered)
    max_y = max(point[1] for point in centered)
    scale = math.hypot(max_x - min_x, max_y - min_y)
    if scale <= 0:
        return None

    vector: list[float] = []
    for x, y, z in centered:
        vector.extend([x / scale, y / scale, z / scale])
    return vector


def _mirror_landmark_vector(vector: list[float]) -> list[float]:
    mirrored = vector.copy()
    for index in range(0, len(mirrored), 3):
        mirrored[index] = -mirrored[index]
    return mirrored


def _landmark_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)) / len(left))


def _crop_face_by_landmarks(image: Any, landmarks: list[Any]) -> Any | None:
    height, width = image.shape[:2]
    xs = [float(point.x) * width for point in landmarks]
    ys = [float(point.y) * height for point in landmarks]
    min_x, max_x = max(0, min(xs)), min(width - 1, max(xs))
    min_y, max_y = max(0, min(ys)), min(height - 1, max(ys))

    face_width = max_x - min_x
    face_height = max_y - min_y
    if face_width <= 0 or face_height <= 0:
        return None

    pad_x = face_width * 0.18
    pad_y = face_height * 0.18
    left = max(0, int(min_x - pad_x))
    right = min(width, int(max_x + pad_x))
    top = max(0, int(min_y - pad_y))
    bottom = min(height, int(max_y + pad_y))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right]


def _extract_appearance_vector(cv2: Any, face_image: Any) -> list[float] | None:
    try:
        import numpy as np  # type: ignore
    except Exception:
        return None

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64))
    gray = cv2.equalizeHist(gray).astype("float32") / 255.0
    gray = (gray - float(gray.mean())) / (float(gray.std()) + 1e-6)
    return gray.flatten().tolist()


def _appearance_distance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        return float("inf")
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    return 1.0 - (dot / ((left_norm * right_norm) + 1e-8))


def extract_face_feature(image_path: Path) -> tuple[FaceFeature | None, str]:
    cv2, cv2_error = _load_cv2()
    if cv2_error:
        return None, cv2_error

    model_error = ensure_face_landmarker_model()
    if model_error:
        return None, model_error

    mp, mp_tasks, vision, mp_error = _load_mediapipe()
    if mp_error:
        return None, mp_error

    image = cv2.imread(str(image_path))
    if image is None:
        return None, f"無法讀取圖片: {image_path}"

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    try:
        base_options = mp_tasks.BaseOptions(model_asset_path=str(FACE_LANDMARKER_MODEL))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
        )
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
        with vision.FaceLandmarker.create_from_options(options) as landmarker:
            result = landmarker.detect(mp_image)
    except Exception as exc:
        return None, f"MediaPipe landmark 偵測失敗: {exc}"

    if not result.face_landmarks:
        return None, f"沒有偵測到臉部 landmarks: {image_path.name}"

    landmarks = result.face_landmarks[0]
    landmark_vector = _normalize_landmarks(landmarks)
    if landmark_vector is None:
        return None, f"臉部 landmarks 正規化失敗: {image_path.name}"

    face_crop = _crop_face_by_landmarks(image, landmarks)
    if face_crop is None:
        return None, f"臉部裁切失敗: {image_path.name}"

    appearance = _extract_appearance_vector(cv2, face_crop)
    if appearance is None:
        return None, f"臉部影像特徵擷取失敗: {image_path.name}"

    return FaceFeature(landmarks=landmark_vector, appearance=appearance), "landmark + appearance 擷取成功"


def extract_landmark_vector(image_path: Path) -> tuple[list[float] | None, str]:
    feature, message = extract_face_feature(image_path)
    if feature is None:
        return None, message
    return feature.landmarks, message


def _member_face_images(member: Member) -> list[Path]:
    if not member.face_folder:
        return []
    folder = Path(member.face_folder)
    if not folder.is_absolute():
        folder = PROJECT_ROOT / folder
    if not folder.exists():
        return []
    return [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]


def recognize_from_image(image_path: Path) -> FaceRecognitionResult:
    if not FACE_ROOT.exists():
        return FaceRecognitionResult(False, message="src/face/ 不存在，無法辨識會員。")

    query_feature, error = extract_face_feature(image_path)
    if query_feature is None:
        return FaceRecognitionResult(False, message=error)

    query_landmarks = [query_feature.landmarks, _mirror_landmark_vector(query_feature.landmarks)]
    best_member: Member | None = None
    best_image = ""
    best_combined_distance = float("inf")
    best_landmark_distance = float("inf")
    best_appearance_distance = float("inf")
    checked_images = 0

    for member in read_members():
        for face_image in _member_face_images(member):
            reference_feature, _ = extract_face_feature(face_image)
            if reference_feature is None:
                continue
            checked_images += 1
            landmark_distance = min(
                _landmark_distance(candidate, reference_feature.landmarks)
                for candidate in query_landmarks
            )
            appearance_distance = _appearance_distance(
                query_feature.appearance,
                reference_feature.appearance,
            )
            combined_distance = (landmark_distance * 0.2) + (appearance_distance * 0.8)
            if combined_distance < best_combined_distance:
                best_combined_distance = combined_distance
                best_landmark_distance = landmark_distance
                best_appearance_distance = appearance_distance
                best_member = member
                best_image = str(face_image)

    if checked_images == 0:
        return FaceRecognitionResult(False, message="沒有可用的會員臉部 landmark 照片。")

    if (
        best_member is None
        or best_landmark_distance > LANDMARK_MATCH_THRESHOLD
        or best_appearance_distance > APPEARANCE_MATCH_THRESHOLD
        or best_combined_distance > COMBINED_MATCH_THRESHOLD
    ):
        return FaceRecognitionResult(
            False,
            identity_path=best_image,
            message=(
                "查無會員資料。"
                f"landmark: {best_landmark_distance:.3f}, "
                f"appearance: {best_appearance_distance:.3f}, "
                f"combined: {best_combined_distance:.3f}"
            ),
        )

    return FaceRecognitionResult(
        True,
        member=best_member,
        identity_path=best_image,
        message=(
            "會員辨識成功，"
            f"landmark: {best_landmark_distance:.3f}, "
            f"appearance: {best_appearance_distance:.3f}, "
            f"combined: {best_combined_distance:.3f}"
        ),
    )


def scan_member() -> FaceRecognitionResult:
    image_path, message = capture_face_image()
    if image_path is None:
        return FaceRecognitionResult(False, message=message)
    return recognize_from_image(image_path)
