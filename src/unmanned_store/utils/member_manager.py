from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MEMBERS_CSV = PROJECT_ROOT / "src" / "unmanned_store" / "data" / "members.csv"
FACE_ROOT = PROJECT_ROOT / "src" / "face"

DEFAULT_MEMBER = {
    "member_id": "M001",
    "name": "學生會員示範者",
    "level": "學生會員",
    "discount": "0.85",
    "face_folder": "src/face/student_member",
}

LEVEL_DISCOUNTS = {
    "一般會員": 0.9,
    "學生會員": 0.85,
    "VIP會員": 0.8,
}


@dataclass
class Member:
    member_id: str
    name: str
    level: str
    discount: float
    face_folder: str

    @classmethod
    def non_member(cls) -> "Member":
        return cls("NON_MEMBER", "非會員", "非會員", 1.0, "")

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Member":
        return cls(
            member_id=row.get("member_id", "").strip(),
            name=row.get("name", "").strip(),
            level=row.get("level", "").strip(),
            discount=float(row.get("discount", "1.0") or 1.0),
            face_folder=row.get("face_folder", "").strip(),
        )

    def to_row(self) -> dict[str, str]:
        return {
            "member_id": self.member_id,
            "name": self.name,
            "level": self.level,
            "discount": str(self.discount),
            "face_folder": self.face_folder,
        }


def ensure_member_files() -> None:
    MEMBERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    FACE_ROOT.mkdir(parents=True, exist_ok=True)
    if not MEMBERS_CSV.exists():
        write_members([Member.from_row(DEFAULT_MEMBER)])


def read_members() -> list[Member]:
    ensure_member_files()
    members: list[Member] = []
    with MEMBERS_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row.get("member_id"):
                members.append(Member.from_row(row))
    if not members:
        members.append(Member.from_row(DEFAULT_MEMBER))
        write_members(members)
    return members


def write_members(members: Iterable[Member]) -> None:
    MEMBERS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with MEMBERS_CSV.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["member_id", "name", "level", "discount", "face_folder"],
        )
        writer.writeheader()
        for member in members:
            writer.writerow(member.to_row())


def normalize_face_folder(face_folder: str | Path) -> str:
    path = Path(face_folder)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    try:
        return str(path.resolve())
    except OSError:
        return str(path.absolute())


def get_member_by_face_folder(face_folder: str | Path) -> Member | None:
    target = normalize_face_folder(face_folder)
    for member in read_members():
        if normalize_face_folder(member.face_folder) == target:
            return member
    return None


def get_student_member() -> Member | None:
    for member in read_members():
        if member.member_id == "M001" or member.level == "學生會員":
            return member
    return None


def next_member_id() -> str:
    max_number = 1
    for member in read_members():
        if member.member_id.startswith("M") and member.member_id[1:].isdigit():
            max_number = max(max_number, int(member.member_id[1:]))
    return f"M{max_number + 1:03d}"


def next_general_face_folder() -> Path:
    FACE_ROOT.mkdir(parents=True, exist_ok=True)
    index = 1
    while True:
        folder = FACE_ROOT / f"general_member_{index:03d}"
        if not folder.exists():
            return folder
        index += 1


def add_member(name: str, level: str, face_folder: Path) -> Member:
    discount = LEVEL_DISCOUNTS.get(level, 0.9)
    member = Member(
        member_id=next_member_id(),
        name=name.strip(),
        level=level,
        discount=discount,
        face_folder=face_folder.relative_to(PROJECT_ROOT).as_posix(),
    )
    members = read_members()
    members.append(member)
    write_members(members)
    return member


def reset_demo_members() -> tuple[int, int]:
    members = read_members()
    kept = [member for member in members if member.member_id == "M001"]
    removed_members = len(members) - len(kept)
    write_members(kept or [Member.from_row(DEFAULT_MEMBER)])

    removed_folders = 0
    FACE_ROOT.mkdir(parents=True, exist_ok=True)
    for folder in FACE_ROOT.glob("general_member_*"):
        if folder.is_dir():
            shutil.rmtree(folder, ignore_errors=True)
            removed_folders += 1
    return removed_members, removed_folders


def discount_label(discount: float) -> str:
    if discount == 1.0:
        return "原價"
    if discount == 0.9:
        return "9 折"
    if discount == 0.85:
        return "85 折"
    if discount == 0.8:
        return "8 折"
    return f"{discount:g} 折"
