from typing import Any, List, Tuple, Dict, Union

from pydantic import BaseModel, Field


class CoordinatesSchema(BaseModel):
    x1: Tuple[int, int]
    x2: Tuple[int, int]
    x3: Tuple[int, int]
    x4: Tuple[int, int]


class ItemSchema(BaseModel):
    coordinates: CoordinatesSchema
    text: str
    score: float


class PageSchema(BaseModel):
    items: List[ItemSchema]


class ReferenceSchema(BaseModel):
    id: str
    target: str
    text: str
    pages: List[PageSchema]
    width: int
    height: int
    template: Dict[str, Any]
    example: Dict[str, Any]
    qr: bool
    signature: bool


class ClassifySchema(BaseModel):
    reference_result: str
    qr: List[str]
    signature: int
    render: str


class OCRWord(BaseModel):
    value: str = Field(..., examples=["example"])
    geometry: List[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    confidence: float = Field(..., examples=[0.99])
    crop_orientation: Dict[str, Any] = Field(
        ..., examples=[{"value": 0, "confidence": None}]
    )


class OCRLine(BaseModel):
    geometry: List[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    words: List[OCRWord] = Field(
        ...,
        examples=[
            {
                "value": "example",
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "confidence": 0.99,
                "crop_orientation": {"value": 0, "confidence": None},
            }
        ],
    )


class OCRBlock(BaseModel):
    geometry: List[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    lines: List[OCRLine] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "words": [
                    {
                        "value": "example",
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "confidence": 0.99,
                        "crop_orientation": {"value": 0, "confidence": None},
                    }
                ],
            }
        ],
    )


class OCRPage(BaseModel):
    blocks: List[OCRBlock] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "lines": [
                    {
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "objectness_score": 0.99,
                        "words": [
                            {
                                "value": "example",
                                "geometry": [0.0, 0.0, 0.0, 0.0],
                                "objectness_score": 0.99,
                                "confidence": 0.99,
                                "crop_orientation": {
                                    "value": 0,
                                    "confidence": None,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    )


class OCROut(BaseModel):
    name: str = Field(..., examples=["example.jpg"])
    orientation: Dict[str, Union[float, None]] = Field(
        ..., examples=[{"value": 0.0, "confidence": 0.99}]
    )
    language: Dict[str, Union[str, float, None]] = Field(
        ..., examples=[{"value": "en", "confidence": 0.99}]
    )
    dimensions: Tuple[int, int] = Field(..., examples=[(100, 100)])
    items: List[OCRPage] = Field(
        ...,
        examples=[
            {
                "geometry": [0.0, 0.0, 0.0, 0.0],
                "objectness_score": 0.99,
                "lines": [
                    {
                        "geometry": [0.0, 0.0, 0.0, 0.0],
                        "objectness_score": 0.99,
                        "words": [
                            {
                                "value": "example",
                                "geometry": [0.0, 0.0, 0.0, 0.0],
                                "objectness_score": 0.99,
                                "confidence": 0.99,
                                "crop_orientation": {
                                    "value": 0,
                                    "confidence": None,
                                },
                            }
                        ],
                    }
                ],
            }
        ],
    )
