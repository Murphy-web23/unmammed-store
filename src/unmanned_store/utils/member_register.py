from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .member_manager import LEVEL_DISCOUNTS, add_member, next_general_face_folder


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV 載入失敗: {exc}"


def _load_cjk_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    font_candidates = [
        Path("C:/Windows/Fonts/msjh.ttc"),
        Path("C:/Windows/Fonts/msjhbd.ttc"),
        Path("C:/Windows/Fonts/simsun.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
        Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    ]
    for font_path in font_candidates:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_text(
    cv2,
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    size: int,
) -> np.ndarray:
    # OpenCV putText 不支援中文，改用 Pillow 確保提示文字可正確顯示。
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(image)
    font = _load_cjk_font(size)
    draw.text(position, text, font=font, fill=(color[2], color[1], color[0]))
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _show_completion(cv2, frame, text: str, duration: float = 1.2) -> None:
    start = time.time()
    while time.time() - start < duration:
        display = frame.copy()
        display = _draw_text(cv2, display, text, (30, 85), (0, 200, 255), 36)
        cv2.imshow("加入會員拍照 - ESC 取消", display)
        if cv2.waitKey(1) & 0xFF == 27:
            break


def capture_member_photos(target_folder: Path, count: int = 3) -> tuple[bool, str]:
    cv2, error = _load_cv2()
    if error:
        return False, error

    target_folder.mkdir(parents=True, exist_ok=True)
    prompts = ["請看正前方", "請稍微左轉", "請稍微右轉"]
    completed_prompts = ["正面拍攝完成", "左轉拍攝完成", "右轉拍攝完成"]
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return False, "攝影機無法開啟，請確認鏡頭是否被其他程式占用。"

    try:
        for index in range(count):
            while True:
                ok, frame = camera.read()
                if not ok:
                    return False, "攝影機讀取失敗。"
                frame = cv2.flip(frame, 1)
                text = f"{prompts[index]}  SPACE拍照 ({index + 1}/{count})"
                frame = _draw_text(cv2, frame, text, (30, 30), (0, 255, 0), 30)
                frame = _draw_text(cv2, frame, "ESC 取消", (30, 70), (0, 255, 255), 28)
                cv2.imshow("加入會員拍照 - ESC 取消", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    return False, "使用者取消拍照。"
                if key == 32:
                    cv2.imwrite(str(target_folder / f"{index + 1}.jpg"), frame)
                    break
            _show_completion(cv2, frame, completed_prompts[index])
        return True, "會員照片拍攝完成。"
    except Exception as exc:
        return False, f"拍照流程發生錯誤: {exc}"
    finally:
        camera.release()
        cv2.destroyAllWindows()


class MemberRegisterWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, on_success) -> None:
        super().__init__(master)
        self.on_success = on_success
        self.title("加入會員")
        self.geometry("360x220")
        self.resizable(False, False)

        self.name_var = tk.StringVar()
        self.level_var = tk.StringVar(value="一般會員")
        self.status_var = tk.StringVar(value="請輸入姓名並選擇會員等級")

        container = ttk.Frame(self, padding=16)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="會員姓名").grid(row=0, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.name_var).grid(
            row=0, column=1, sticky="ew", pady=6
        )

        ttk.Label(container, text="會員等級").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Combobox(
            container,
            textvariable=self.level_var,
            values=list(LEVEL_DISCOUNTS.keys()),
            state="readonly",
        ).grid(row=1, column=1, sticky="ew", pady=6)

        ttk.Label(container, textvariable=self.status_var, foreground="#555").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=10
        )

        button_row = ttk.Frame(container)
        button_row.grid(row=3, column=0, columnspan=2, sticky="e", pady=8)
        ttk.Button(button_row, text="開始拍照", command=self.start_capture).pack(
            side="left", padx=4
        )
        ttk.Button(button_row, text="取消", command=self.destroy).pack(side="left", padx=4)
        container.columnconfigure(1, weight=1)

    def start_capture(self) -> None:
        name = self.name_var.get().strip()
        level = self.level_var.get().strip()
        if not name:
            messagebox.showwarning("資料不足", "請輸入會員姓名。", parent=self)
            return

        target_folder = next_general_face_folder()
        self.status_var.set("即將開啟攝影機，請依畫面提示拍三張照片。")
        self.update_idletasks()
        success, message = capture_member_photos(target_folder)
        if not success:
            messagebox.showerror("加入會員失敗", message, parent=self)
            self.status_var.set(message)
            return

        member = add_member(name=name, level=level, face_folder=target_folder)
        messagebox.showinfo(
            "會員建立成功",
            f"{member.name} 已建立成功，請重新掃描會員。",
            parent=self,
        )
        self.on_success(member)
        self.destroy()
