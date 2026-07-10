from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PRODUCTS_CSV = PROJECT_ROOT / "src" / "unmanned_store" / "data" / "products.csv"

DEFAULT_PRODUCTS = [
    {"product_id": "P001", "name": "可樂", "price": "30", "class_name": "coke"},
    {"product_id": "P002", "name": "餅乾", "price": "25", "class_name": "cookie"},
    {"product_id": "P003", "name": "泡麵", "price": "45", "class_name": "noodle"},
    {"product_id": "P004", "name": "水", "price": "20", "class_name": "water"},
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


def get_product_by_class(class_name: str) -> Product | None:
    target = class_name.strip()
    for product in read_products():
        if product.class_name == target:
            return product
    return None


def list_class_names() -> list[str]:
    return [product.class_name for product in read_products()]
