from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from unmanned_store.utils.checkout import Cart
from unmanned_store.utils.face_recognizer import scan_member
from unmanned_store.utils.item_detector import find_model_path, scan_item
from unmanned_store.utils.member_manager import (
    FACE_ROOT,
    MEMBERS_CSV,
    PROJECT_ROOT,
    Member,
    discount_label,
    ensure_member_files,
    get_student_member,
    read_members,
    reset_demo_members,
)
from unmanned_store.utils.member_register import MemberRegisterWindow
from unmanned_store.utils.product_manager import (
    PRODUCTS_CSV,
    get_product_by_class,
    list_class_names,
    read_products,
)


class UnmannedStoreApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("無人商店 AI 自助結帳系統")
        self.geometry("920x640")
        self.minsize(820, 560)

        ensure_member_files()
        read_products()

        self.current_member = Member.non_member()
        self.cart = Cart()
        self.product_index = 0

        self.name_var = tk.StringVar()
        self.level_var = tk.StringVar()
        self.discount_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.original_total_var = tk.StringVar()
        self.discounted_total_var = tk.StringVar()

        self._build_ui()
        self.set_member(Member.non_member(), "尚未掃描會員")
        self.refresh_cart()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 16, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Microsoft JhengHei UI", 11, "bold"))

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        ttk.Label(root, text="無人商店 AI 自助結帳系統", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 12)
        )

        member_box = ttk.LabelFrame(root, text="會員資訊", padding=12, style="Section.TLabelframe")
        member_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        member_box.columnconfigure(1, weight=1)

        labels = [
            ("姓名", self.name_var),
            ("會員等級", self.level_var),
            ("折扣", self.discount_var),
            ("狀態", self.status_var),
        ]
        for row, (label, variable) in enumerate(labels):
            ttk.Label(member_box, text=f"{label}:").grid(row=row, column=0, sticky="nw", pady=5)
            ttk.Label(member_box, textvariable=variable, wraplength=330).grid(
                row=row, column=1, sticky="w", pady=5
            )

        action_box = ttk.LabelFrame(member_box, text="正式操作", padding=10)
        action_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 8))
        for index in range(2):
            action_box.columnconfigure(index, weight=1)
        ttk.Button(action_box, text="掃描會員", command=self.scan_member_flow).grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="加入會員", command=self.open_register_window).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="掃描商品", command=self.scan_item_flow).grid(
            row=1, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="掃描完成 / 結帳", command=self.checkout_flow).grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="清空購物車", command=self.clear_cart).grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=4
        )

        test_box = ttk.LabelFrame(member_box, text="測試模式", padding=10)
        test_box.grid(row=6, column=0, columnspan=2, sticky="ew", pady=8)
        for index in range(2):
            test_box.columnconfigure(index, weight=1)
        test_buttons = [
            ("測試非會員", self.test_non_member),
            ("測試學生會員", self.test_student_member),
            ("測試一般會員", self.test_general_member),
            ("測試加入商品", self.test_add_product),
            ("檢查資料庫", self.check_database),
            ("測試 Landmark 單次辨識", self.test_landmark_once),
            ("測試 YOLO 單次辨識", self.test_yolo_once),
            ("重置現場會員", self.reset_demo_member_data),
        ]
        for index, (text, command) in enumerate(test_buttons):
            ttk.Button(test_box, text=text, command=command).grid(
                row=index // 2, column=index % 2, sticky="ew", padx=4, pady=4
            )

        cart_box = ttk.LabelFrame(root, text="購物車", padding=12, style="Section.TLabelframe")
        cart_box.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        cart_box.rowconfigure(0, weight=1)
        cart_box.columnconfigure(0, weight=1)

        self.cart_list = tk.Listbox(cart_box, font=("Microsoft JhengHei UI", 11), height=14)
        self.cart_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(cart_box, orient="vertical", command=self.cart_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.cart_list.configure(yscrollcommand=scrollbar.set)

        totals = ttk.Frame(cart_box)
        totals.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        totals.columnconfigure(1, weight=1)
        ttk.Label(totals, text="原價總金額:").grid(row=0, column=0, sticky="w", pady=4)
        ttk.Label(totals, textvariable=self.original_total_var).grid(row=0, column=1, sticky="e")
        ttk.Label(totals, text="折扣後總金額:").grid(row=1, column=0, sticky="w", pady=4)
        ttk.Label(totals, textvariable=self.discounted_total_var).grid(row=1, column=1, sticky="e")

    def set_member(self, member: Member, status: str) -> None:
        self.current_member = member
        self.name_var.set(member.name)
        self.level_var.set(member.level)
        self.discount_var.set(discount_label(member.discount))
        self.status_var.set(status)
        self.refresh_cart()

    def refresh_cart(self) -> None:
        self.cart_list.delete(0, tk.END)
        for product in self.cart.items:
            self.cart_list.insert(tk.END, f"{product.name}  ${product.price}")
        self.original_total_var.set(f"{self.cart.original_total()} 元")
        self.discounted_total_var.set(f"{self.cart.discounted_total(self.current_member.discount)} 元")

    def scan_member_flow(self) -> None:
        result = scan_member()
        if result.success and result.member:
            self.set_member(result.member, "會員辨識成功")
            messagebox.showinfo("會員辨識成功", f"歡迎 {result.member.name}")
            return

        self.set_member(Member.non_member(), "查無會員資料，目前以非會員身份結帳")
        join = messagebox.askyesno("查無會員資料", "你目前不是會員，是否要加入會員？")
        if join:
            self.open_register_window()

    def open_register_window(self) -> None:
        MemberRegisterWindow(self, on_success=self.on_member_created)

    def on_member_created(self, member: Member) -> None:
        self.status_var.set("會員建立成功，請重新掃描會員")
        messagebox.showinfo("下一步", "會員建立成功，請重新掃描會員後再結帳套用折扣。")

    def scan_item_flow(self) -> None:
        result = scan_item()
        if not result.success:
            messagebox.showwarning("商品辨識失敗", result.message)
            return

        product = get_product_by_class(result.class_name)
        if not product:
            messagebox.showwarning(
                "商品尚未建檔",
                f"YOLO 回傳 class_name: {result.class_name}\n此商品尚未建檔。",
            )
            return

        add = messagebox.askyesno(
            "確認加入購物車",
            f"偵測到: {product.name}\n價格: {product.price} 元\n信心度: {result.confidence:.2f}\n\n是否加入購物車？",
        )
        if add:
            self.cart.add_item(product)
            self.refresh_cart()

    def checkout_flow(self) -> None:
        if not self.cart.items:
            messagebox.showwarning("購物車是空的", "請先加入商品再結帳。")
            return
        messagebox.showinfo("結帳結果", self.cart.checkout_message(self.current_member))

    def clear_cart(self) -> None:
        self.cart.clear()
        self.refresh_cart()

    def test_non_member(self) -> None:
        self.set_member(Member.non_member(), "查無會員資料，目前以非會員身份結帳")

    def test_student_member(self) -> None:
        member = get_student_member()
        if not member:
            messagebox.showwarning("找不到學生會員", "members.csv 找不到學生會員示範者。")
            return
        self.set_member(member, "測試模式: 學生會員")

    def test_general_member(self) -> None:
        member = Member("TEST_GENERAL", "一般會員示範者", "一般會員", 0.9, "")
        self.set_member(member, "測試模式: 一般會員")

    def test_add_product(self) -> None:
        products = read_products()
        if not products:
            messagebox.showwarning("沒有商品資料", "products.csv 沒有可加入的商品。")
            return
        product = products[self.product_index % len(products)]
        self.product_index += 1
        self.cart.add_item(product)
        self.refresh_cart()

    def check_database(self) -> None:
        lines = [
            f"members.csv: {'存在' if MEMBERS_CSV.exists() else '不存在'}",
            f"products.csv: {'存在' if PRODUCTS_CSV.exists() else '不存在'}",
            f"src/face/: {'存在' if FACE_ROOT.exists() else '不存在'}",
            "",
            "會員照片資料:",
        ]
        for member in read_members():
            folder = Path(member.face_folder)
            if not folder.is_absolute():
                folder = (PROJECT_ROOT / member.face_folder).resolve()
            image_count = 0
            if folder.exists():
                image_count = len(list(folder.glob("*.jpg"))) + len(list(folder.glob("*.png")))
            lines.append(f"- {member.member_id} {member.name}: {folder} / {image_count} 張")

        model_primary = PROJECT_ROOT / "model" / "best.pt"
        model_backup = PROJECT_ROOT / "model" / "runs" / "classify" / "train" / "weights" / "best.pt"
        lines.extend(
            [
                "",
                f"model/best.pt: {'存在' if model_primary.resolve().exists() else '不存在'}",
                f"model/runs/classify/train/weights/best.pt: {'存在' if model_backup.resolve().exists() else '不存在'}",
                "",
                "products.csv class_name:",
                ", ".join(list_class_names()) or "沒有商品 class_name",
            ]
        )
        messagebox.showinfo("資料庫檢查", "\n".join(lines))

    def test_landmark_once(self) -> None:
        result = scan_member()
        if result.success and result.member:
            messagebox.showinfo(
                "Landmark 單次辨識",
                "\n".join(
                    [
                        "辨識成功",
                        f"找到圖片: {result.identity_path}",
                        f"會員: {result.member.name}",
                        f"會員等級: {result.member.level}",
                    ]
                ),
            )
        else:
            messagebox.showwarning("Landmark 單次辨識", f"辨識失敗\n原因: {result.message}")

    def test_yolo_once(self) -> None:
        result = scan_item()
        if not result.success:
            messagebox.showwarning("YOLO 單次辨識", result.message)
            return
        product = get_product_by_class(result.class_name)
        if product:
            messagebox.showinfo(
                "YOLO 單次辨識",
                f"class_name: {result.class_name}\nconfidence: {result.confidence:.2f}\n對應商品: {product.name} / {product.price} 元",
            )
        else:
            messagebox.showwarning(
                "YOLO 單次辨識",
                f"class_name: {result.class_name}\nconfidence: {result.confidence:.2f}\n此 class_name 尚未建檔",
            )

    def reset_demo_member_data(self) -> None:
        confirm = messagebox.askyesno(
            "重置現場會員",
            "將刪除 members.csv 中非 M001 的會員，並刪除 src/face/general_member_ 開頭資料夾。\n\n確定要重置嗎？",
        )
        if not confirm:
            return
        removed_members, removed_folders = reset_demo_members()
        self.set_member(Member.non_member(), "已重置現場會員資料")
        messagebox.showinfo(
            "重置完成",
            f"已刪除 {removed_members} 筆現場會員資料與 {removed_folders} 個照片資料夾。",
        )


def main() -> None:
    model_path = find_model_path()
    app = UnmannedStoreApp()
    if model_path is None:
        app.status_var.set("尚未掃描會員；提醒: 找不到 YOLO 模型，掃描商品可先用測試加入商品備援")
    app.mainloop()


if __name__ == "__main__":
    main()
