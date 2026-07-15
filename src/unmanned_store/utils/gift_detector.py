from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = PROJECT_ROOT / "model" / "drinks.pt"
DEFAULT_CONFIDENCE = 0.65
GIFT_SCAN_WINDOW = "yolo_gift_scan"


@dataclass(frozen=True)
class GiftRule:
    minimum_total: int
    kind: str
    display_name: str


@dataclass
class GiftDetectionResult:
    success: bool
    required_kind: str = ""
    required_display_name: str = ""
    detected_class: str = ""
    confidence: float = 0.0
    tea_confidence: float = 0.0
    milk_confidence: float = 0.0
    message: str = ""
    model_path: str = ""


GIFT_RULES = (
    GiftRule(minimum_total=150, kind="milk", display_name="牛奶"),
    GiftRule(minimum_total=100, kind="tea", display_name="茶"),
)

GIFT_ALIASES = {
    "tea": ("tea", "green tea", "black tea", "oolong", "cha", "drink tea", "bottle tea", "tea bottle", "茶"),
    "milk": ("milk", "milk bottle", "bottle milk", "cow milk", "牛奶", "鮮奶"),
}


def required_gift_for_total(total: int | float) -> GiftRule | None:
    amount = int(total)
    for rule in GIFT_RULES:
        if amount >= rule.minimum_total:
            return rule
    return None


def _load_cv2() -> tuple[Any | None, str]:
    try:
        import cv2  # type: ignore

        return cv2, ""
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV is not available: {exc}"


def _load_yolo() -> tuple[Any | None, str]:
    if not MODEL_PATH.exists():
        return None, f"YOLO model not found: {MODEL_PATH}"
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO(str(MODEL_PATH)), ""
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"Ultralytics YOLO is not available: {exc}"


def _normalise_label(label: str) -> str:
    return label.strip().replace("_", " ").replace("-", " ").lower()


def _matches_gift_kind(label: str, required_kind: str) -> bool:
    normalized = _normalise_label(label)
    aliases = GIFT_ALIASES.get(required_kind, (required_kind,))
    return any(alias.lower() in normalized for alias in aliases)


def _best_detection(results: Any, required_kind: str, min_confidence: float) -> tuple[str, float, bool]:
    best_label = ""
    best_confidence = 0.0
    best_is_required = False

    names = getattr(results, "names", {}) or {}
    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return best_label, best_confidence, best_is_required

    for box in boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = str(names.get(class_id, class_id))
        if confidence < min_confidence:
            continue
        is_required = _matches_gift_kind(label, required_kind)
        if is_required and (not best_is_required or confidence > best_confidence):
            best_label = label
            best_confidence = confidence
            best_is_required = True
        elif not best_is_required and confidence > best_confidence:
            best_label = label
            best_confidence = confidence

    return best_label, best_confidence, best_is_required


def _confidence_by_gift_kind(results: Any) -> dict[str, float]:
    scores = {"tea": 0.0, "milk": 0.0}
    names = getattr(results, "names", {}) or {}
    boxes = getattr(results, "boxes", None)
    if boxes is None:
        return scores

    for box in boxes:
        confidence = float(box.conf[0])
        class_id = int(box.cls[0])
        label = str(names.get(class_id, class_id))
        for kind in scores:
            if _matches_gift_kind(label, kind):
                scores[kind] = max(scores[kind], confidence)
    return scores


def _result_from_yolo_results(results: Any, required_kind: str, min_confidence: float) -> GiftDetectionResult:
    label, confidence, is_required = _best_detection(results, required_kind, min_confidence)
    scores = _confidence_by_gift_kind(results)
    required_name = required_kind

    if is_required:
        return GiftDetectionResult(
            True,
            required_kind=required_kind,
            required_display_name=required_name,
            detected_class=label,
            confidence=confidence,
            tea_confidence=scores["tea"],
            milk_confidence=scores["milk"],
            message=(
                f"Detected required gift: {label} ({confidence:.2f}); "
                f"tea={scores['tea']:.2f}, milk={scores['milk']:.2f}"
            ),
            model_path=str(MODEL_PATH),
        )

    if label:
        message = (
            f"Detected {label} ({confidence:.2f}), but required gift is {required_name}. "
            f"tea={scores['tea']:.2f}, milk={scores['milk']:.2f}"
        )
    else:
        message = (
            f"No {required_name} gift detected above confidence {min_confidence:.2f}. "
            f"tea={scores['tea']:.2f}, milk={scores['milk']:.2f}"
        )
    return GiftDetectionResult(
        False,
        required_kind=required_kind,
        required_display_name=required_name,
        detected_class=label,
        confidence=confidence,
        tea_confidence=scores["tea"],
        milk_confidence=scores["milk"],
        message=message,
        model_path=str(MODEL_PATH),
    )


def _draw_confidence_bar(cv2: Any, image: Any, label: str, confidence: float, x: int, y: int, color: tuple[int, int, int]) -> None:
    width = 230
    height = 18
    filled = int(width * max(0.0, min(1.0, confidence)))
    cv2.putText(image, f"{label}: {confidence:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.62, color, 2)
    cv2.rectangle(image, (x, y), (x + width, y + height), (220, 220, 220), 1)
    if filled > 0:
        cv2.rectangle(image, (x, y), (x + filled, y + height), color, -1)


def _draw_demo_overlay(
    cv2: Any,
    image: Any,
    result: GiftDetectionResult,
    required_display_name: str,
    waiting_for_confirm: bool = False,
) -> Any:
    overlay = image.copy()
    cv2.rectangle(overlay, (12, 12), (430, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, image, 0.38, 0, image)

    is_matched = waiting_for_confirm or result.success
    status = "MATCHED - live confidence comparison" if is_matched else "Scanning gift confidence comparison"
    cv2.putText(image, status, (24, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 0) if is_matched else (0, 220, 255), 2)
    cv2.putText(image, f"Required gift: {required_display_name}", (24, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)
    _draw_confidence_bar(cv2, image, "Tea", result.tea_confidence, 24, 104, (0, 220, 80))
    _draw_confidence_bar(cv2, image, "Milk", result.milk_confidence, 24, 146, (240, 240, 240))
    if waiting_for_confirm:
        cv2.putText(image, "Press ENTER or SPACE to finish", (24, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)
    return image


def detect_gift_in_image(
    image: Any,
    required_kind: str,
    min_confidence: float = DEFAULT_CONFIDENCE,
    model: Any | None = None,
) -> GiftDetectionResult:
    if model is None:
        model, error = _load_yolo()
        if error:
            return GiftDetectionResult(False, required_kind=required_kind, message=error, model_path=str(MODEL_PATH))

    results = model(image, verbose=False)[0]
    return _result_from_yolo_results(results, required_kind, min_confidence)


def scan_gift_camera(
    total: int | float,
    camera_index: int = 0,
    min_confidence: float = DEFAULT_CONFIDENCE,
    timeout_seconds: int = 25,
) -> GiftDetectionResult:
    rule = required_gift_for_total(total)
    if rule is None:
        return GiftDetectionResult(True, message="No gift scan is required for totals below 100.")

    cv2, error = _load_cv2()
    if error:
        return GiftDetectionResult(False, required_kind=rule.kind, required_display_name=rule.display_name, message=error)

    model, error = _load_yolo()
    if error:
        return GiftDetectionResult(False, required_kind=rule.kind, required_display_name=rule.display_name, message=error)

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return GiftDetectionResult(
            False,
            required_kind=rule.kind,
            required_display_name=rule.display_name,
            message=f"Failed to open camera {camera_index}.",
            model_path=str(MODEL_PATH),
        )

    started_at = time.monotonic()
    confirmed_result: GiftDetectionResult | None = None
    last_result = GiftDetectionResult(
        False,
        required_kind=rule.kind,
        required_display_name=rule.display_name,
        message=f"Waiting for {rule.display_name}.",
        model_path=str(MODEL_PATH),
    )

    try:
        while True:
            if confirmed_result is None and time.monotonic() - started_at >= timeout_seconds:
                break

            ok, frame = camera.read()
            if not ok:
                last_result.message = "Failed to read camera frame."
                break

            yolo_results = model(frame, verbose=False)[0]
            result = _result_from_yolo_results(yolo_results, rule.kind, min_confidence)
            last_result = result
            annotated = yolo_results.plot()
            if result.success:
                confirmed_result = result
            _draw_demo_overlay(
                cv2,
                annotated,
                result,
                rule.display_name,
                waiting_for_confirm=confirmed_result is not None,
            )
            cv2.imshow(GIFT_SCAN_WINDOW, annotated)

            key = cv2.waitKey(1 if confirmed_result is None else 80) & 0xFF
            if confirmed_result is not None and key in (13, 32):
                return confirmed_result
            if key == 27:
                if confirmed_result is not None:
                    confirmed_result.message = "Gift scan confirmed and closed with ESC."
                    return confirmed_result
                result.message = "Gift scan cancelled."
                return result

        last_result.message = f"Timed out before detecting required gift: {rule.display_name}."
        return last_result
    finally:
        camera.release()
        cv2.destroyAllWindows()
