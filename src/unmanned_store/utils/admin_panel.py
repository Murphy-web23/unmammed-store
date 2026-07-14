from __future__ import annotations

import shutil
import tkinter as tk
from tkinter import messagebox, ttk

from .item_detector import ITEM_ROOT, capture_product_reference_photos
from .member_manager import (
    LEVEL_DISCOUNTS,
    Member,
    add_member_row,
    delete_member,
    read_members,
    update_member,
)
from .product_manager import (
    Product,
    add_product,
    delete_product,
    read_products,
    update_product,
)


class AdminPanelWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc, on_data_changed=None) -> None:
        super().__init__(master)
        self.on_data_changed = on_data_changed
        self.title("管理員操作")
        self.geometry("840x560")
        self.minsize(780, 520)

        self.member_id_var = tk.StringVar()
        self.member_name_var = tk.StringVar()
        self.member_level_var = tk.StringVar(value="一般會員")
        self.member_discount_var = tk.StringVar(value="0.9")
        self.member_face_var = tk.StringVar()

        self.product_id_var = tk.StringVar()
        self.product_name_var = tk.StringVar()
        self.product_price_var = tk.StringVar()
        self.product_class_var = tk.StringVar()

        self._build_ui()
        self._refresh_members()
        self._refresh_products()

    def _build_ui(self) -> None:
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        notebook = ttk.Notebook(root)
        notebook.pack(fill="both", expand=True)

        self.member_tab = ttk.Frame(notebook, padding=12)
        self.product_tab = ttk.Frame(notebook, padding=12)
        notebook.add(self.member_tab, text="會員管理")
        notebook.add(self.product_tab, text="商品管理")

        self._build_member_tab()
        self._build_product_tab()

    def _build_member_tab(self) -> None:
        self.member_tab.columnconfigure(0, weight=1, minsize=280)
        self.member_tab.columnconfigure(1, weight=4, minsize=430)
        self.member_tab.rowconfigure(0, weight=1)

        table_frame = ttk.Frame(self.member_tab)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("member_id", "name", "level", "discount", "face_folder")
        self.member_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        headers = {
            "member_id": "會員編號",
            "name": "姓名",
            "level": "等級",
            "discount": "折扣",
            "face_folder": "照片資料夾",
        }
        for key in columns:
            self.member_tree.heading(key, text=headers[key])
        self.member_tree.column("member_id", width=90, anchor="center", stretch=False)
        self.member_tree.column("name", width=120, anchor="w", stretch=False)
        self.member_tree.column("level", width=95, anchor="center", stretch=False)
        self.member_tree.column("discount", width=70, anchor="center", stretch=False)
        self.member_tree.column("face_folder", width=220, anchor="w", stretch=False)
        self.member_tree.grid(row=0, column=0, sticky="nsew")
        self.member_tree.bind("<<TreeviewSelect>>", self._on_member_selected)

        member_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.member_tree.yview)
        member_scroll.grid(row=0, column=1, sticky="ns")
        member_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.member_tree.xview)
        member_scroll_x.grid(row=1, column=0, sticky="ew")
        self.member_tree.configure(yscrollcommand=member_scroll.set, xscrollcommand=member_scroll_x.set)

        form = ttk.LabelFrame(self.member_tab, text="編輯會員", padding=10)
        form.grid(row=0, column=1, sticky="nsew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="會員編號").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.member_id_var, state="readonly", width=26).grid(
            row=0, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="姓名").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.member_name_var, width=26).grid(
            row=1, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="會員等級").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Combobox(
            form,
            textvariable=self.member_level_var,
            values=list(LEVEL_DISCOUNTS.keys()),
            state="readonly",
            width=24,
        ).grid(row=2, column=1, sticky="ew", pady=5)

        ttk.Label(form, text="折扣").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.member_discount_var, width=26).grid(
            row=3, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="照片資料夾").grid(row=4, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.member_face_var, width=26).grid(
            row=4, column=1, sticky="ew", pady=5
        )

        button_row = ttk.Frame(form)
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(12, 2))
        button_row.columnconfigure(0, weight=1)

        ttk.Button(button_row, text="新增會員", command=self._add_member).grid(
            row=0, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="更新會員", command=self._update_member).grid(
            row=1, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="刪除會員", command=self._delete_member).grid(
            row=2, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="重新整理", command=self._refresh_members).grid(
            row=3, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )

    def _build_product_tab(self) -> None:
        self.product_tab.columnconfigure(0, weight=2)
        self.product_tab.columnconfigure(1, weight=3)
        self.product_tab.rowconfigure(0, weight=1)

        table_frame = ttk.Frame(self.product_tab)
        table_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        table_frame.rowconfigure(0, weight=1)
        table_frame.columnconfigure(0, weight=1)

        columns = ("product_id", "name", "price", "class_name")
        self.product_tree = ttk.Treeview(table_frame, columns=columns, show="headings", height=14)
        headers = {
            "product_id": "商品編號",
            "name": "商品名稱",
            "price": "價格",
            "class_name": "class_name",
        }
        for key in columns:
            self.product_tree.heading(key, text=headers[key])
        self.product_tree.column("product_id", width=110, anchor="center", stretch=False)
        self.product_tree.column("name", width=220, anchor="w", stretch=False)
        self.product_tree.column("price", width=90, anchor="e", stretch=False)
        self.product_tree.column("class_name", width=260, anchor="w", stretch=False)
        self.product_tree.grid(row=0, column=0, sticky="nsew")
        self.product_tree.bind("<<TreeviewSelect>>", self._on_product_selected)

        product_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.product_tree.yview)
        product_scroll.grid(row=0, column=1, sticky="ns")
        product_scroll_x = ttk.Scrollbar(table_frame, orient="horizontal", command=self.product_tree.xview)
        product_scroll_x.grid(row=1, column=0, sticky="ew")
        self.product_tree.configure(yscrollcommand=product_scroll.set, xscrollcommand=product_scroll_x.set)

        form = ttk.LabelFrame(self.product_tab, text="編輯商品", padding=10)
        form.grid(row=0, column=1, sticky="nsew")
        form.columnconfigure(1, weight=1)

        ttk.Label(form, text="商品編號").grid(row=0, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.product_id_var, state="readonly", width=30).grid(
            row=0, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="商品名稱").grid(row=1, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.product_name_var, width=30).grid(
            row=1, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="價格").grid(row=2, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.product_price_var, width=30).grid(
            row=2, column=1, sticky="ew", pady=5
        )

        ttk.Label(form, text="class_name").grid(row=3, column=0, sticky="w", pady=5)
        ttk.Entry(form, textvariable=self.product_class_var, width=30).grid(
            row=3, column=1, sticky="ew", pady=5
        )

        tip = ttk.Label(
            form,
            text="新增商品會啟動攝影機並拍攝 6 張不同角度參考照片。",
            wraplength=250,
            foreground="#555",
        )
        tip.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 6))

        button_row = ttk.Frame(form)
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew", pady=(8, 2))
        button_row.columnconfigure(0, weight=1)

        ttk.Button(button_row, text="新增商品", command=self._add_product).grid(
            row=0, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="更新商品", command=self._update_product).grid(
            row=1, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="刪除商品", command=self._delete_product).grid(
            row=2, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )
        ttk.Button(button_row, text="重新整理", command=self._refresh_products).grid(
            row=3, column=0, sticky="ew", padx=3, pady=3, ipady=4
        )

    def _refresh_members(self) -> None:
        for row_id in self.member_tree.get_children():
            self.member_tree.delete(row_id)

        for member in read_members():
            self.member_tree.insert(
                "",
                "end",
                iid=member.member_id,
                values=(
                    member.member_id,
                    member.name,
                    member.level,
                    f"{member.discount:g}",
                    member.face_folder,
                ),
            )

    def _refresh_products(self) -> None:
        for row_id in self.product_tree.get_children():
            self.product_tree.delete(row_id)

        for product in read_products():
            self.product_tree.insert(
                "",
                "end",
                iid=product.product_id,
                values=(product.product_id, product.name, product.price, product.class_name),
            )

    def _on_member_selected(self, _event=None) -> None:
        selected = self.member_tree.selection()
        if not selected:
            return
        member_id = selected[0]
        values = self.member_tree.item(member_id, "values")
        if len(values) < 5:
            return
        self.member_id_var.set(str(values[0]))
        self.member_name_var.set(str(values[1]))
        self.member_level_var.set(str(values[2]))
        self.member_discount_var.set(str(values[3]))
        self.member_face_var.set(str(values[4]))

    def _on_product_selected(self, _event=None) -> None:
        selected = self.product_tree.selection()
        if not selected:
            return
        product_id = selected[0]
        values = self.product_tree.item(product_id, "values")
        if len(values) < 4:
            return
        self.product_id_var.set(str(values[0]))
        self.product_name_var.set(str(values[1]))
        self.product_price_var.set(str(values[2]))
        self.product_class_var.set(str(values[3]))

    def _add_member(self) -> None:
        name = self.member_name_var.get().strip()
        level = self.member_level_var.get().strip()
        discount_text = self.member_discount_var.get().strip()
        face_folder = self.member_face_var.get().strip()

        if not name:
            messagebox.showwarning("資料不足", "請輸入會員姓名。", parent=self)
            return

        try:
            discount = float(discount_text)
        except ValueError:
            messagebox.showwarning("格式錯誤", "折扣需為數字，例如 0.9。", parent=self)
            return

        member = add_member_row(name=name, level=level, discount=discount, face_folder=face_folder)
        messagebox.showinfo("新增完成", f"已新增會員 {member.name}。", parent=self)
        self._refresh_members()
        self.member_id_var.set(member.member_id)
        if self.on_data_changed:
            self.on_data_changed()

    def _update_member(self) -> None:
        member_id = self.member_id_var.get().strip()
        if not member_id:
            messagebox.showwarning("未選取會員", "請先選取要更新的會員。", parent=self)
            return

        try:
            discount = float(self.member_discount_var.get().strip())
        except ValueError:
            messagebox.showwarning("格式錯誤", "折扣需為數字，例如 0.85。", parent=self)
            return

        updated = Member(
            member_id=member_id,
            name=self.member_name_var.get().strip(),
            level=self.member_level_var.get().strip(),
            discount=discount,
            face_folder=self.member_face_var.get().strip(),
        )
        if not updated.name:
            messagebox.showwarning("資料不足", "會員姓名不可為空。", parent=self)
            return

        if not update_member(updated):
            messagebox.showerror("更新失敗", "找不到要更新的會員。", parent=self)
            return

        messagebox.showinfo("更新完成", f"已更新會員 {updated.name}。", parent=self)
        self._refresh_members()
        if self.on_data_changed:
            self.on_data_changed()

    def _delete_member(self) -> None:
        member_id = self.member_id_var.get().strip()
        if not member_id:
            messagebox.showwarning("未選取會員", "請先選取要刪除的會員。", parent=self)
            return

        confirm = messagebox.askyesno(
            "刪除會員",
            f"確定要刪除會員 {member_id} 嗎？",
            parent=self,
        )
        if not confirm:
            return

        if not delete_member(member_id):
            messagebox.showwarning("刪除失敗", "刪除失敗，M001 不可刪除或資料不存在。", parent=self)
            return

        messagebox.showinfo("刪除完成", f"會員 {member_id} 已刪除。", parent=self)
        self._refresh_members()
        self.member_id_var.set("")
        self.member_name_var.set("")
        self.member_face_var.set("")
        if self.on_data_changed:
            self.on_data_changed()

    def _add_product(self) -> None:
        name = self.product_name_var.get().strip()
        class_name = self.product_class_var.get().strip()
        price_text = self.product_price_var.get().strip()

        if not name or not class_name or not price_text:
            messagebox.showwarning("資料不足", "請輸入商品名稱、價格、class_name。", parent=self)
            return

        if any(product.class_name == class_name for product in read_products()):
            messagebox.showwarning(
                "class_name 已存在",
                "此 class_name 已存在，請改用更新商品或使用新的 class_name。",
                parent=self,
            )
            return

        try:
            price = int(float(price_text))
        except ValueError:
            messagebox.showwarning("格式錯誤", "價格需為數字。", parent=self)
            return

        confirm = messagebox.askyesno(
            "開始拍照",
            "將開啟攝影機拍攝 6 張不同角度照片，是否開始？",
            parent=self,
        )
        if not confirm:
            return

        success, message, _folder = capture_product_reference_photos(class_name, count=6)
        if not success:
            messagebox.showerror("新增商品失敗", message, parent=self)
            return

        product = add_product(name=name, price=price, class_name=class_name)
        messagebox.showinfo(
            "新增完成",
            f"已新增商品 {product.name}，並完成 6 張商品參考照片拍攝。",
            parent=self,
        )
        self._refresh_products()
        self.product_id_var.set(product.product_id)
        if self.on_data_changed:
            self.on_data_changed()

    def _update_product(self) -> None:
        product_id = self.product_id_var.get().strip()
        if not product_id:
            messagebox.showwarning("未選取商品", "請先選取要更新的商品。", parent=self)
            return

        name = self.product_name_var.get().strip()
        class_name = self.product_class_var.get().strip()
        if not name or not class_name:
            messagebox.showwarning("資料不足", "商品名稱與 class_name 不可為空。", parent=self)
            return

        try:
            price = int(float(self.product_price_var.get().strip()))
        except ValueError:
            messagebox.showwarning("格式錯誤", "價格需為數字。", parent=self)
            return

        old_product = next((product for product in read_products() if product.product_id == product_id), None)
        if not old_product:
            messagebox.showerror("更新失敗", "找不到要更新的商品。", parent=self)
            return

        old_folder = ITEM_ROOT / old_product.class_name
        new_folder = ITEM_ROOT / class_name
        if old_product.class_name != class_name and old_folder.exists() and not new_folder.exists():
            old_folder.rename(new_folder)

        updated = Product(product_id=product_id, name=name, price=price, class_name=class_name)
        if not update_product(updated):
            messagebox.showerror("更新失敗", "商品更新失敗。", parent=self)
            return

        messagebox.showinfo("更新完成", f"已更新商品 {updated.name}。", parent=self)
        self._refresh_products()
        if self.on_data_changed:
            self.on_data_changed()

    def _delete_product(self) -> None:
        product_id = self.product_id_var.get().strip()
        if not product_id:
            messagebox.showwarning("未選取商品", "請先選取要刪除的商品。", parent=self)
            return

        product = next((item for item in read_products() if item.product_id == product_id), None)
        if not product:
            messagebox.showerror("刪除失敗", "找不到要刪除的商品。", parent=self)
            return

        confirm = messagebox.askyesno(
            "刪除商品",
            f"確定要刪除商品 {product.name} 嗎？",
            parent=self,
        )
        if not confirm:
            return

        if not delete_product(product_id):
            messagebox.showerror("刪除失敗", "商品刪除失敗。", parent=self)
            return

        remove_images = messagebox.askyesno(
            "刪除參考照片",
            "是否一併刪除該商品 class_name 對應的參考照片資料夾？",
            parent=self,
        )
        if remove_images:
            folder = ITEM_ROOT / product.class_name
            if folder.exists() and folder.is_dir():
                shutil.rmtree(folder, ignore_errors=True)

        messagebox.showinfo("刪除完成", f"已刪除商品 {product.name}。", parent=self)
        self._refresh_products()
        self.product_id_var.set("")
        self.product_name_var.set("")
        self.product_price_var.set("")
        self.product_class_var.set("")
        if self.on_data_changed:
            self.on_data_changed()
