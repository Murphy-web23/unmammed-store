from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from unmanned_store.utils.checkout import Cart
from unmanned_store.utils.face_recognizer import scan_member
from unmanned_store.utils.gift_detector import required_gift_for_total, scan_gift_camera
from unmanned_store.utils.item_detector import find_item_reference_root, scan_item
from unmanned_store.utils.admin_panel import AdminPanelWindow
from unmanned_store.utils.member_manager import (
    Member,
    discount_label,
    ensure_member_files,
)
from unmanned_store.utils.member_register import MemberRegisterWindow
from unmanned_store.utils.product_manager import (
    get_product_by_class,
    read_products,
)


class UnmannedStoreApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("自助結帳系統")
        self.geometry("980x660")
        self.minsize(920, 600)

        ensure_member_files()
        read_products()

        self.current_member = Member.non_member()
        self.cart = Cart()
        self.prompted_gift_kind: str | None = None
        self.gift_message = "未領取滿額贈品"
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
        style.configure("Action.TButton", font=("Microsoft JhengHei UI", 12, "bold"), padding=(10, 8))
        style.configure("Danger.TButton", font=("Microsoft JhengHei UI", 10, "bold"), padding=(6, 2))

        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=7)
        root.columnconfigure(1, weight=5)
        root.rowconfigure(1, weight=1)

        ttk.Label(root, text="自助結帳系統", style="Title.TLabel").grid(
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

        action_box = ttk.LabelFrame(member_box, text="操作", padding=12)
        action_box.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(18, 8))
        for index in range(2):
            action_box.columnconfigure(index, weight=1)
        ttk.Button(action_box, text="掃描會員", command=self.scan_member_flow, style="Action.TButton").grid(
            row=0, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="加入會員", command=self.open_register_window, style="Action.TButton").grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="掃描商品", command=self.scan_item_flow, style="Action.TButton").grid(
            row=1, column=0, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="管理員操作", command=self.open_admin_panel, style="Action.TButton").grid(
            row=1, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(action_box, text="掃描完成 / 結帳", command=self.checkout_flow, style="Action.TButton").grid(
            row=2, column=0, columnspan=2, sticky="ew", padx=4, pady=(10, 4)
        )

        cart_box = ttk.LabelFrame(root, text="購物車", padding=12, style="Section.TLabelframe")
        cart_box.grid(row=1, column=1, sticky="nsew", padx=(8, 0))
        cart_box.rowconfigure(0, weight=1)
        cart_box.columnconfigure(0, weight=1)

        self.cart_canvas = tk.Canvas(cart_box, highlightthickness=0)
        self.cart_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(cart_box, orient="vertical", command=self.cart_canvas.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.cart_canvas.configure(yscrollcommand=scrollbar.set)

        self.cart_items_frame = ttk.Frame(self.cart_canvas)
        self.cart_window = self.cart_canvas.create_window((0, 0), window=self.cart_items_frame, anchor="nw")
        self.cart_items_frame.columnconfigure(0, weight=1)
        self.cart_items_frame.bind("<Configure>", self._sync_cart_scroll)
        self.cart_canvas.bind("<Configure>", self._resize_cart_content)

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
        for widget in self.cart_items_frame.winfo_children():
            widget.destroy()

        if not self.cart.items:
            ttk.Label(self.cart_items_frame, text="購物車目前沒有商品", foreground="#666").grid(
                row=0, column=0, sticky="w", padx=6, pady=8
            )
        else:
            for index, product in enumerate(self.cart.items):
                row = ttk.Frame(self.cart_items_frame, padding=(4, 2))
                row.grid(row=index, column=0, sticky="ew", pady=2)
                row.columnconfigure(0, weight=1)
                row.columnconfigure(1, weight=0)
                row.columnconfigure(2, weight=0)
                ttk.Label(row, text=f"{product.name}").grid(row=0, column=0, sticky="w")

                ttk.Label(row, text=f"${product.price}", anchor="e", width=8).grid(
                    row=0, column=1, sticky="e", padx=(8, 10)
                )
                ttk.Button(
                    row,
                    text="刪除",
                    command=lambda item_index=index: self.remove_cart_item(item_index),
                    style="Danger.TButton",
                ).grid(row=0, column=2, sticky="e")

        self.cart_items_frame.update_idletasks()
        self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox("all"))
        self.original_total_var.set(f"{self.cart.original_total()} 元")
        self.discounted_total_var.set(f"{self.cart.discounted_total(self.current_member.discount)} 元")

    def _sync_cart_scroll(self, _event=None) -> None:
        self.cart_canvas.configure(scrollregion=self.cart_canvas.bbox("all"))

    def _resize_cart_content(self, event) -> None:
        self.cart_canvas.itemconfig(self.cart_window, width=event.width)

    def remove_cart_item(self, index: int) -> None:
        self.cart.remove_item(index)
        self.refresh_cart()
        self._reset_gift_state_if_needed()

    def _current_payable_total(self) -> int:
        return self.cart.discounted_total(self.current_member.discount)

    def _reset_gift_state_if_needed(self) -> None:
        gift_rule = required_gift_for_total(self._current_payable_total())
        if gift_rule is None:
            self.prompted_gift_kind = None
            self.gift_message = "未領取滿額贈品"
        elif self.prompted_gift_kind and self.prompted_gift_kind != gift_rule.kind:
            self.prompted_gift_kind = None
            self.gift_message = "未領取滿額贈品"

    def maybe_prompt_gift_flow(self) -> bool:
        payable_total = self._current_payable_total()
        gift_rule = required_gift_for_total(payable_total)
        if gift_rule is None or self.prompted_gift_kind == gift_rule.kind:
            return True

        wants_gift = messagebox.askyesno(
            "滿額贈品",
            (
                f"目前折扣後金額為 {payable_total} 元，符合滿額贈品資格。\n"
                f"可領取贈品：{gift_rule.display_name}\n\n"
                "是否需要領取贈品？"
            ),
        )
        self.prompted_gift_kind = gift_rule.kind
        if not wants_gift:
            self.gift_message = f"已放棄滿額贈品：{gift_rule.display_name}"
            return True

        self.status_var.set(f"正在辨識贈品：{gift_rule.display_name}")
        self.update_idletasks()
        gift_result = scan_gift_camera(payable_total)
        if not gift_result.success:
            self.gift_message = "贈品辨識未通過，未領取贈品"
            self.status_var.set(self.gift_message)
            messagebox.showwarning(
                "贈品辨識未通過",
                (
                    f"{gift_result.message}\n\n"
                    "本次不兌換贈品。"
                ),
            )
            return True

        self.gift_message = (
            f"已領取滿額贈品：{gift_rule.display_name}"
            f"（YOLO: {gift_result.detected_class}, {gift_result.confidence:.2f}；"
            f"茶 {gift_result.tea_confidence:.2f} / 牛奶 {gift_result.milk_confidence:.2f}）"
        )
        self.status_var.set(self.gift_message)
        return True

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
            if result.class_name:
                candidate_product = get_product_by_class(result.class_name)
                if candidate_product:
                    confirmed = messagebox.askyesno(
                        "商品辨識不確定",
                        (
                            f"{result.message}\n\n"
                            f"候選商品: {candidate_product.name}\n"
                            f"候選價格: {candidate_product.price} 元\n"
                            f"候選信心度: {result.confidence:.2f}\n\n"
                            "是否仍要加入購物車？"
                        ),
                    )
                    if confirmed:
                        self.cart.add_item(candidate_product)
                        self.refresh_cart()
                    return
            messagebox.showwarning("商品辨識失敗", result.message)
            return

        product = get_product_by_class(result.class_name)
        if not product:
            messagebox.showwarning(
                "商品尚未建檔",
                f"OpenCV 回傳 class_name: {result.class_name}\n此商品尚未建檔。",
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
        payable_total = self._current_payable_total()
        gift_rule = required_gift_for_total(payable_total)
        checkout_text = "\n".join(
            [
                self.cart.checkout_message(self.current_member),
                "",
                "謝謝光臨，歡迎下次再來！",
            ]
        )
        messagebox.showinfo("結帳結果", checkout_text)
        if gift_rule:
            self.maybe_prompt_gift_flow()
        self.cart.clear()
        self.prompted_gift_kind = None
        self.gift_message = "未領取滿額贈品"
        self.refresh_cart()

    def open_admin_panel(self) -> None:
        password = simpledialog.askstring("管理員驗證", "請輸入管理員密碼", show="*", parent=self)
        if password is None:
            return
        if password != "1234":
            messagebox.showerror("驗證失敗", "密碼錯誤。")
            return
        AdminPanelWindow(self, on_data_changed=self.refresh_cart)


def main() -> None:
    app = UnmannedStoreApp()
    if find_item_reference_root() is None:
        app.status_var.set("尚未掃描會員；提醒: 找不到 src/item 商品參考照片，請先由管理員新增商品並拍攝參考照片")
    app.mainloop()


if __name__ == "__main__":
    main()
