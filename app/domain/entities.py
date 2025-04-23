from typing import Optional, Any
from dataclasses import dataclass


@dataclass
class CoordinatesEntity:
    x1: list[int, int]
    x2: list[int, int]
    x3: list[int, int]
    x4: list[int, int]


@dataclass
class ItemEntity:
    coordinates: CoordinatesEntity
    text: str
    score: float


@dataclass
class PageEntity:
    items: list[ItemEntity]
    text: str
    width: int
    height: int


@dataclass
class ReferenceEntity:
    id: str
    target: str
    text: str
    pages: list[PageEntity]
    width: int
    height: int
    template: dict[str, Any]
    example: dict[str, Any]
    qr: bool
    signature: bool
    render: bool


@dataclass
class FileEntity:
    file: bytes
    references_info: list[ReferenceEntity]


@dataclass
class ClassifyEntity:
    reference_result: str
    qr: list[str]
    signature: int
    render: str
