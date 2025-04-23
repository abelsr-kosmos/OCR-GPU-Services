from typing import Any

from pydantic import BaseModel, Field


class CoordinatesSchema(BaseModel):
    x1: list[int, int]
    x2: list[int, int]
    x3: list[int, int]
    x4: list[int, int]


class ItemSchema(BaseModel):
    coordinates: CoordinatesSchema
    text: str
    score: float


class PageSchema(BaseModel):
    items: list[ItemSchema]


class ReferenceSchema(BaseModel):
    id: str
    target: str
    text: str
    pages: list[PageSchema]
    width: int
    height: int
    template: dict[str, Any]
    example: dict[str, Any]
    qr: bool
    signature: bool


class ClassifySchema(BaseModel):
    reference_result: str
    qr: list[str]
    signature: int
    render: str


class OCRWord(BaseModel):
    value: str = Field(..., examples=["example"])
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    confidence: float = Field(..., examples=[0.99])
    crop_orientation: dict[str, Any] = Field(
        ..., examples=[{"value": 0, "confidence": None}]
    )


class OCRLine(BaseModel):
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    words: list[OCRWord] = Field(
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
    geometry: list[float] = Field(..., examples=[[0.0, 0.0, 0.0, 0.0]])
    objectness_score: float = Field(..., examples=[0.99])
    lines: list[OCRLine] = Field(
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
    blocks: list[OCRBlock] = Field(
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
    orientation: dict[str, float | None] = Field(
        ..., examples=[{"value": 0.0, "confidence": 0.99}]
    )
    language: dict[str, str | float | None] = Field(
        ..., examples=[{"value": "en", "confidence": 0.99}]
    )
    dimensions: tuple[int, int] = Field(..., examples=[(100, 100)])
    items: list[OCRPage] = Field(
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
