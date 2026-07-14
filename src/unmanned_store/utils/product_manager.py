from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCTS_CSV = PROJECT_ROOT / "src" / "unmanned_store" / "data" / "products.csv"

DEFAULT_PRODUCTS = [
    {"product_id": "P001", "name": "Asahi 啤酒", "price": "49", "class_name": "asahi"},
    {"product_id": "P002", "name": "Owala 水壺", "price": "899", "class_name": "owala"},
    {"product_id": "P003", "name": "藍色原子筆", "price": "15", "class_name": "pen_blue"},
    {"product_id": "P004", "name": "Tiffany 色原子筆", "price": "15", "class_name": "pen_tiffany"},
    {"product_id": "P005", "name": "綠色雨傘", "price": "199", "class_name": "umbrella_green"},
    {"product_id": "P006", "name": "Tiffany 色雨傘", "price": "199", "class_name": "umbrella_tiffany"},
]


@dataclass
class Product:
    product_id: str
    name: str
    price: int
    class_name: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Product":
        return cls(
            product_id=row.get("product_id", "").strip(),
            name=row.get("name", "").strip(),
            price=int(float(row.get("price", "0") or 0)),
            class_name=row.get("class_name", "").strip(),
        )


def ensure_product_file() -> None:
    PRODUCTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not PRODUCTS_CSV.exists():
        with PRODUCTS_CSV.open("w", encoding="utf-8-sig", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=["product_id", "name", "price", "class_name"],
            )
            writer.writeheader()
            writer.writerows(DEFAULT_PRODUCTS)


def read_products() -> list[Product]:
    ensure_product_file()
    products: list[Product] = []
    with PRODUCTS_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("product_id"):
                products.append(Product.from_row(row))
    return products


def write_products(products: list[Product]) -> None:
    ensure_product_file()
    with PRODUCTS_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["product_id", "name", "price", "class_name"],
        )
        writer.writeheader()
        for product in products:
            writer.writerow(
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "price": str(product.price),
                    "class_name": product.class_name,
                }
            )


def next_product_id() -> str:
    max_number = 0
    for product in read_products():
        if product.product_id.startswith("P") and product.product_id[1:].isdigit():
            max_number = max(max_number, int(product.product_id[1:]))
    return f"P{max_number + 1:03d}"


def add_product(name: str, price: int, class_name: str) -> Product:
    products = read_products()
    product = Product(
        product_id=next_product_id(),
        name=name.strip(),
        price=int(price),
        class_name=class_name.strip(),
    )
    products.append(product)
    write_products(products)
    return product


def update_product(updated: Product) -> bool:
    products = read_products()
    for index, product in enumerate(products):
        if product.product_id == updated.product_id:
            products[index] = updated
            write_products(products)
            return True
    return False


def delete_product(product_id: str) -> bool:
    products = read_products()
    kept = [product for product in products if product.product_id != product_id]
    if len(kept) == len(products):
        return False
    write_products(kept)
    return True


def get_product_by_class(class_name: str) -> Product | None:
    target = class_name.strip()
    for product in read_products():
        if product.class_name == target:
            return product
    return None


def list_class_names() -> list[str]:
    return [product.class_name for product in read_products()]


def _write_products(products: list[Product]) -> None:
    ensure_product_file()
    with PRODUCTS_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["product_id", "name", "price", "class_name"],
        )
        writer.writeheader()
        for product in products:
            writer.writerow(
                {
                    "product_id": product.product_id,
                    "name": product.name,
                    "price": str(product.price),
                    "class_name": product.class_name,
                }
            )


def _normalize_product_id(product_id: str) -> str:
    return product_id.strip().upper()


def next_product_id() -> str:
    highest_number = 0
    for product in read_products():
        product_id = _normalize_product_id(product.product_id)
        if not product_id.startswith("P"):
            continue
        try:
            highest_number = max(highest_number, int(product_id[1:]))
        except ValueError:
            continue
    return f"P{highest_number + 1:03d}"


def upsert_product(product: Product) -> None:
    products = read_products()
    normalized_id = _normalize_product_id(product.product_id)
    normalized_class_name = product.class_name.strip()
    updated_products: list[Product] = []
    replaced = False
    for existing in products:
        if (
            _normalize_product_id(existing.product_id) == normalized_id
            or existing.class_name == normalized_class_name
        ):
            if not replaced:
                updated_products.append(product)
                replaced = True
            continue
        updated_products.append(existing)

    if not replaced:
        updated_products.append(product)
    _write_products(updated_products)


def delete_product_by_id(product_id: str) -> bool:
    target = _normalize_product_id(product_id)
    products = read_products()
    filtered = [product for product in products if _normalize_product_id(product.product_id) != target]
    if len(filtered) == len(products):
        return False
    _write_products(filtered)
    return True


def delete_product_by_class(class_name: str) -> bool:
    target = class_name.strip()
    products = read_products()
    filtered = [product for product in products if product.class_name != target]
    if len(filtered) == len(products):
        return False
    _write_products(filtered)
    return True


class ProductManagerWindow(tk.Toplevel):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.title("商品管理")
        self.geometry("620x430")
        self.minsize(620, 430)

        self.product_id_var = tk.StringVar()
        self.name_var = tk.StringVar()
        self.price_var = tk.StringVar()
        self.class_name_var = tk.StringVar()

        container = ttk.Frame(self, padding=12)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        form_box = ttk.LabelFrame(container, text="新增 / 更新商品", padding=10)
        form_box.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        form_box.columnconfigure(1, weight=1)

        fields = [
            ("商品編號", self.product_id_var),
            ("商品名稱", self.name_var),
            ("價格", self.price_var),
            ("class_name", self.class_name_var),
        ]
        for row, (label, variable) in enumerate(fields):
            ttk.Label(form_box, text=label).grid(row=row, column=0, sticky="w", pady=6)
            ttk.Entry(form_box, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=6)

        hint = ttk.Label(
            form_box,
            text="商品編號可留空自動產生；class_name 需與 src/item 資料夾一致。",
            wraplength=250,
            foreground="#555",
        )
        hint.grid(row=4, column=0, columnspan=2, sticky="w", pady=(6, 10))

        button_row = ttk.Frame(form_box)
        button_row.grid(row=5, column=0, columnspan=2, sticky="ew")
        for index in range(2):
            button_row.columnconfigure(index, weight=1)
        ttk.Button(button_row, text="加入 / 更新", command=self.save_product).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(button_row, text="清空", command=self.clear_form).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )
        ttk.Button(button_row, text="關閉", command=self.destroy).grid(
            row=1, column=0, columnspan=2, sticky="ew", pady=(8, 0)
        )

        list_box = ttk.LabelFrame(container, text="商品清單", padding=10)
        list_box.grid(row=0, column=1, sticky="nsew")
        list_box.rowconfigure(0, weight=1)
        list_box.columnconfigure(0, weight=1)

        self.product_list = tk.Listbox(list_box, height=14, font=("Microsoft JhengHei UI", 10))
        self.product_list.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(list_box, orient="vertical", command=self.product_list.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.product_list.configure(yscrollcommand=scrollbar.set)
        self.product_list.bind("<<ListboxSelect>>", self.on_select)

        list_buttons = ttk.Frame(list_box)
        list_buttons.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        for index in range(2):
            list_buttons.columnconfigure(index, weight=1)
        ttk.Button(list_buttons, text="刪除所選", command=self.delete_selected).grid(
            row=0, column=0, sticky="ew", padx=(0, 4)
        )
        ttk.Button(list_buttons, text="重新整理", command=self.refresh_list).grid(
            row=0, column=1, sticky="ew", padx=(4, 0)
        )

        self.refresh_list()

    def clear_form(self) -> None:
        self.product_id_var.set("")
        self.name_var.set("")
        self.price_var.set("")
        self.class_name_var.set("")

    def refresh_list(self) -> None:
        self.product_list.delete(0, tk.END)
        for product in read_products():
            self.product_list.insert(
                tk.END,
                f"{product.product_id} | {product.name} | ${product.price} | {product.class_name}",
            )

    def on_select(self, _event: tk.Event) -> None:
        selection = self.product_list.curselection()
        if not selection:
            return
        entry = self.product_list.get(selection[0])
        parts = [part.strip() for part in entry.split("|")]
        if len(parts) != 4:
            return
        self.product_id_var.set(parts[0])
        self.name_var.set(parts[1])
        self.price_var.set(parts[2].replace("$", "").strip())
        self.class_name_var.set(parts[3])

    def save_product(self) -> None:
        product_id = self.product_id_var.get().strip() or next_product_id()
        name = self.name_var.get().strip()
        price_text = self.price_var.get().strip()
        class_name = self.class_name_var.get().strip()

        if not name or not price_text or not class_name:
            messagebox.showwarning("資料不足", "請填入商品名稱、價格與 class_name。", parent=self)
            return

        try:
            price = int(float(price_text))
        except ValueError:
            messagebox.showwarning("價格格式錯誤", "價格請輸入數字。", parent=self)
            return

        product = Product(product_id=product_id, name=name, price=price, class_name=class_name)
        upsert_product(product)
        self.product_id_var.set(product.product_id)
        self.refresh_list()
        messagebox.showinfo("商品已儲存", f"{product.name} 已更新。", parent=self)

    def delete_selected(self) -> None:
        selection = self.product_list.curselection()
        if not selection:
            messagebox.showwarning("尚未選取", "請先選擇要刪除的商品。", parent=self)
            return

        entry = self.product_list.get(selection[0])
        product_id = entry.split("|")[0].strip()
        if not messagebox.askyesno("確認刪除", f"確定要刪除 {entry} 嗎？", parent=self):
            return

        if delete_product_by_id(product_id):
            self.clear_form()
            self.refresh_list()
            messagebox.showinfo("刪除完成", "商品已刪除。", parent=self)
        else:
            messagebox.showwarning("刪除失敗", "找不到對應商品。", parent=self)
