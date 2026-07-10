from __future__ import annotations

import tempfile
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .member_manager import FACE_ROOT, PROJECT_ROOT, Member, get_member_by_face_folder


DEEPFACE_HOME = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "deepface"
MPL_CONFIG_DIR = PROJECT_ROOT / "src" / "unmanned_store" / "temp" / "matplotlib"
DEEPFACE_HOME.mkdir(parents=True, exist_ok=True)
MPL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("DEEPFACE_HOME", str(DEEPFACE_HOME))
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CONFIG_DIR))


@dataclass
class FaceRecognitionResult:
    success: bool
    member: Member | None = None
    identity_path: str = ""
    message: str = ""


def _safe_mirror_frame(cv2: Any, frame: Any) -> Any:
    return cv2.flip(frame, 1)


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV 載入失敗: {exc}"


def _load_deepface():
    try:
        from deepface import DeepFace  # type: ignore

        return DeepFace, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"DeepFace 載入失敗: {exc}"


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


def recognize_from_image(image_path: Path) -> FaceRecognitionResult:
    DeepFace, error = _load_deepface()
    if error:
        return FaceRecognitionResult(False, message=error)

    if not FACE_ROOT.exists():
        return FaceRecognitionResult(False, message="src/face/ 不存在，無法辨識會員。")

    try:
        find_kwargs = {
            "img_path": str(image_path),
            "db_path": str(FACE_ROOT),
            "model_name": "VGG-Face",
            "distance_metric": "cosine",
            "detector_backend": "opencv",
            "enforce_detection": False,
            "align": True,
            "silent": True,
            "refresh_database": True,
        }
        try:
            results = DeepFace.find(**find_kwargs)
        except TypeError:
            find_kwargs.pop("refresh_database", None)
            results = DeepFace.find(**find_kwargs)
        frames = results if isinstance(results, list) else [results]
        best_identity = ""
        best_distance = float("inf")
        for frame in frames:
            if frame is not None and not frame.empty and "identity" in frame.columns:
                distance_columns = [
                    column
                    for column in frame.columns
                    if column.endswith("_cosine")
                    or column.endswith("_euclidean")
                    or column == "distance"
                ]
                sorted_frame = frame
                if distance_columns:
                    sorted_frame = frame.sort_values(distance_columns[0], ascending=True)
                row = sorted_frame.iloc[0]
                distance = float(row[distance_columns[0]]) if distance_columns else 0.0
                threshold = float(row["threshold"]) if "threshold" in row and row["threshold"] else None
                if threshold is not None and distance > threshold:
                    continue
                if distance < best_distance:
                    best_distance = distance
                    best_identity = str(row["identity"])
        if not best_identity:
            return FaceRecognitionResult(False, message="查無會員資料。")

        identity_path = Path(best_identity)
        face_folder = identity_path.parent
        try:
            face_folder.relative_to(PROJECT_ROOT)
        except ValueError:
            pass

        member = get_member_by_face_folder(face_folder)
        if not member:
            return FaceRecognitionResult(
                False,
                identity_path=str(identity_path),
                message="找到相似照片，但 members.csv 沒有對應會員資料。",
            )
        return FaceRecognitionResult(
            True,
            member=member,
            identity_path=str(identity_path),
            message="會員辨識成功。",
        )
    except Exception as exc:
        return FaceRecognitionResult(False, message=f"DeepFace 辨識失敗: {exc}")


def scan_member() -> FaceRecognitionResult:
    image_path, message = capture_face_image()
    if image_path is None:
        return FaceRecognitionResult(False, message=message)
    return recognize_from_image(image_path)
