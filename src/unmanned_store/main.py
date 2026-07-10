from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from unmanned_store.utils.checkout import Cart
from unmanned_store.utils.face_recognizer import scan_member
from unmanned_store.utils.item_detector import find_item_reference_root, scan_item
from unmanned_store.utils.member_manager import (
    Member,
    discount_label,
    ensure_member_files,
)
from unmanned_store.utils.member_register import MemberRegisterWindow
from unmanned_store.utils.product_manager import ProductManagerWindow, get_product_by_class


# 想讓購物車商品欄更矮：把這個數字調小，例如 6、5、4。
CART_LIST_HEIGHT = 7
# 想讓整個主頁面往下移：把這個數字調大，例如 16 或 24。
PAGE_TOP_OFFSET = 18


class UnmannedStoreApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("無人商店 AI 自助結帳系統")
        self.geometry("860x560")
        self.minsize(760, 500)

        ensure_member_files()

        self.current_member = Member.non_member()
        self.cart = Cart()
        self.product_manager_window: ProductManagerWindow | None = None

        self.name_var = tk.StringVar()
        self.level_var = tk.StringVar()
        self.discount_var = tk.StringVar()
        self.status_var = tk.StringVar()
        self.original_total_var = tk.StringVar()
        self.discounted_total_var = tk.StringVar()
        self.cart_selection_var = tk.StringVar(value="選取商品後可按刪除 × 移除單項")

        self._build_ui()
        self.set_member(Member.non_member(), "尚未掃描會員")
        self.refresh_cart()

    def _build_ui(self) -> None:
        style = ttk.Style(self)
        style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 14, "bold"))
        style.configure("Section.TLabelframe.Label", font=("Microsoft JhengHei UI", 10, "bold"))
        style.configure("Action.TButton", padding=(10, 8))

        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True, pady=(PAGE_TOP_OFFSET, 0))
        root.columnconfigure(0, weight=1)
        root.columnconfigure(1, weight=1)
        root.rowconfigure(1, weight=1)

        ttk.Label(root, text="無人商店 AI 自助結帳系統", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8)
        )

        member_box = ttk.LabelFrame(root, text="會員資訊", padding=8, style="Section.TLabelframe")
        member_box.grid(row=1, column=0, sticky="nsew", padx=(0, 8))
        member_box.columnconfigure(1, weight=1)

        labels = [
            ("姓名", self.name_var),
            ("會員等級", self.level_var),
            ("折扣", self.discount_var),
            ("狀態", self.status_var),
        ]
        for row, (label, variable) in enumerate(labels):
            ttk.Label(member_box, text=f"{label}:").grid(row=row, column=0, sticky="nw", pady=3)
            ttk.Label(member_box, textvariable=variable, wraplength=330).grid(
                row=row, column=1, sticky="w", pady=3
            )

        action_box = ttk.LabelFrame(member_box, text="購買", padding=8)
        action_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(10, 6))
        for index in range(2):
            action_box.columnconfigure(index, weight=1)
        ttk.Button(action_box, text="掃描會員", command=self.scan_member_flow).grid(
            row=0, column=0, sticky="ew", padx=3, pady=4
        )
        ttk.Button(action_box, text="加入會員", command=self.open_register_window).grid(
            row=0, column=1, sticky="ew", padx=3, pady=4
        )
        ttk.Button(action_box, text="掃描商品", command=self.scan_item_flow).grid(
            row=1, column=0, sticky="ew", padx=3, pady=4
        )
        ttk.Button(action_box, text="商品管理", command=self.open_product_manager).grid(
            row=1, column=1, sticky="ew", padx=3, pady=4
        )
        ttk.Button(action_box, text="掃描完成 / 結帳", command=self.checkout_flow).grid(
            row=2, column=0, sticky="ew", padx=3, pady=4
        )
        ttk.Button(action_box, text="清空購物車", command=self.clear_cart).grid(
            row=2, column=1, sticky="ew", padx=3, pady=4
        )
        for child in action_box.winfo_children():
            if isinstance(child, ttk.Button):
                child.configure(style="Action.TButton")

        cart_box = ttk.LabelFrame(root, text="購物車", padding=6, style="Section.TLabelframe")
        cart_box.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        cart_box.rowconfigure(0, weight=0)
        cart_box.columnconfigure(0, weight=1)

        self.cart_list = tk.Listbox(cart_box, font=("Microsoft JhengHei UI", 10), height=CART_LIST_HEIGHT)
        self.cart_list.grid(row=0, column=0, sticky="ew")
        scrollbar = ttk.Scrollbar(cart_box, orient="vertical", command=self.cart_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.cart_list.configure(yscrollcommand=scrollbar.set)
        self.cart_list.bind("<<ListboxSelect>>", self.on_cart_select)
        self.cart_list.bind("<Delete>", self.delete_selected_cart_item)

        cart_actions = ttk.Frame(cart_box)
        cart_actions.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        cart_actions.columnconfigure(0, weight=1)
        ttk.Button(cart_actions, text="刪除選取 ×", command=self.delete_selected_cart_item).grid(
            row=0, column=0, sticky="ew"
        )

        ttk.Label(cart_box, textvariable=self.cart_selection_var, foreground="#555").grid(
            row=2, column=0, columnspan=2, sticky="w", pady=(4, 0)
        )

        totals = ttk.Frame(cart_box)
        totals.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(6, 0))
        totals.columnconfigure(1, weight=1)
        ttk.Label(totals, text="原價總金額:").grid(row=0, column=0, sticky="w", pady=1)
        ttk.Label(totals, textvariable=self.original_total_var).grid(row=0, column=1, sticky="e")
        ttk.Label(totals, text="折扣後總金額:").grid(row=1, column=0, sticky="w", pady=1)
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

    def open_product_manager(self) -> None:
        password = simpledialog.askstring("商品管理驗證", "請輸入商品管理密碼", show="*", parent=self)
        if password is None:
            return
        if password != "1234":
            messagebox.showerror("驗證失敗", "密碼錯誤，無法開啟商品管理。", parent=self)
            return
        if self.product_manager_window is not None and self.product_manager_window.winfo_exists():
            self.product_manager_window.lift()
            self.product_manager_window.focus_force()
            return
        self.product_manager_window = ProductManagerWindow(self)
        self.product_manager_window.transient(self)
        self.product_manager_window.focus_force()

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
                f"MediaPipe 回傳 class_name: {result.class_name}\n此商品尚未建檔。",
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

    def on_cart_select(self, _event: tk.Event) -> None:
        selection = self.cart_list.curselection()
        if not selection:
            self.cart_selection_var.set("選取商品後可按刪除 × 移除單項")
            return
        product = self.cart.items[selection[0]]
        self.cart_selection_var.set(f"目前選取: {product.name}")

    def delete_selected_cart_item(self, _event: tk.Event | None = None) -> None:
        selection = self.cart_list.curselection()
        if not selection:
            messagebox.showwarning("尚未選取", "請先選擇要刪除的購物車商品。")
            return

        index = selection[0]
        product = self.cart.items[index]
        if not messagebox.askyesno("確認刪除", f"確定要從購物車移除 {product.name} 嗎？"):
            return

        del self.cart.items[index]
        self.refresh_cart()
        self.cart_selection_var.set("商品已移除")


def main() -> None:
    app = UnmannedStoreApp()
    if find_item_reference_root() is None:
        app.status_var.set("尚未掃描會員；提醒: 找不到 src/item 商品參考照片")
    app.mainloop()


if __name__ == "__main__":
    main()
