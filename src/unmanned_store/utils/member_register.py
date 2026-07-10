from __future__ import annotations

import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

from .member_manager import LEVEL_DISCOUNTS, add_member, next_general_face_folder


def _load_cv2():
    try:
        import cv2  # type: ignore

        return cv2, None
    except Exception as exc:  # pragma: no cover - depends on local install
        return None, f"OpenCV 載入失敗: {exc}"


def capture_member_photos(target_folder: Path, count: int = 3) -> tuple[bool, str]:
    cv2, error = _load_cv2()
    if error:
        return False, error

    target_folder.mkdir(parents=True, exist_ok=True)
    prompts = ["請看正前方", "請稍微左轉", "請稍微右轉"]
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        cv2.destroyAllWindows()
        return False, "攝影機無法開啟，請確認鏡頭是否被其他程式占用。"

    try:
        for index in range(count):
            for number in (3, 2, 1):
                start = time.time()
                while time.time() - start < 1:
                    ok, frame = camera.read()
                    if not ok:
                        return False, "攝影機讀取失敗。"
                    text = f"{prompts[index]}  {number}"
                    cv2.putText(
                        frame,
                        text,
                        (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("加入會員拍照 - ESC 取消", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        return False, "使用者取消拍照。"

            ok, frame = camera.read()
            if not ok:
                return False, "攝影機讀取失敗。"
            cv2.imwrite(str(target_folder / f"{index + 1}.jpg"), frame)
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
