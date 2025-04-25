from dataclasses import dataclass
from typing import Any, List, Tuple, Dict


@dataclass
class CoordinatesEntity:
    x1: Tuple[int, int]
    x2: Tuple[int, int]
    x3: Tuple[int, int]
    x4: Tuple[int, int]


@dataclass
class ItemEntity:
    coordinates: CoordinatesEntity
    text: str
    score: float


@dataclass
class PageEntity:
    items: List[ItemEntity]
    text: str
    width: int
    height: int


@dataclass
class ReferenceEntity:
    id: str
    target: str
    text: str
    pages: List[PageEntity]
    width: int
    height: int
    template: Dict[str, Any]
    example: Dict[str, Any]
    qr: bool
    signature: bool
    render: bool


@dataclass
class FileEntity:
    file: bytes
    references_info: List[ReferenceEntity]


@dataclass
class ClassifyEntity:
    reference_result: str
    qr: List[str]
    signature: int
    render: str
