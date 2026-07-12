from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP

from .member_manager import Member, discount_label
from .product_manager import Product


def round_money(amount: float | Decimal) -> int:
    return int(Decimal(str(amount)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass
class Cart:
    items: list[Product] = field(default_factory=list)

    def add_item(self, product: Product) -> None:
        self.items.append(product)

    def remove_item(self, index: int) -> bool:
        if index < 0 or index >= len(self.items):
            return False
        del self.items[index]
        return True

    def clear(self) -> None:
        self.items.clear()

    def original_total(self) -> int:
        return sum(product.price for product in self.items)

    def discounted_total(self, discount: float) -> int:
        return round_money(Decimal(self.original_total()) * Decimal(str(discount)))

    def detail_lines(self) -> list[str]:
        if not self.items:
            return ["購物車目前沒有商品"]
        return [f"{product.name} ${product.price}" for product in self.items]

    def checkout_message(self, member: Member) -> str:
        return "\n".join(
            [
                "商品明細:",
                *self.detail_lines(),
                "",
                f"會員: {member.name}",
                f"會員等級: {member.level}",
                f"折扣: {discount_label(member.discount)}",
                "",
                f"原價總金額: {self.original_total()}",
                f"折扣後金額: {self.discounted_total(member.discount)}",
                "",
                "結帳完成！",
            ]
        )
