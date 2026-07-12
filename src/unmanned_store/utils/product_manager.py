from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


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
