from __future__ import annotations

import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent.parent))

from unmanned_store.utils.gift_detector import required_gift_for_total, scan_gift_camera


def main() -> None:
    raw_total = input("Enter checkout total: ").strip()
    try:
        total = int(float(raw_total))
    except ValueError:
        print("Please enter a valid number.")
        return

    rule = required_gift_for_total(total)
    if rule is None:
        print("No gift required. Total is below 100.")
        return

    print(f"Total {total}: required gift is {rule.display_name}.")
    print("Show the gift to the camera. Press ESC to cancel.")
    result = scan_gift_camera(total)
    print(result.message)


if __name__ == "__main__":
    main()
