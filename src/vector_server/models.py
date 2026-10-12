"""Public conversion options and computational limits."""

import re
from dataclasses import dataclass, field
from typing import Literal

Scalar = int | float | str | bool


class VectorizationError(ValueError):
    """The image or requested conversion cannot be processed."""


@dataclass(frozen=True)
class VectorOptions:
    """User-facing controls independent of engine implementation."""

    preset: Literal["artwork", "photo"] = "artwork"
    colors: int = 0
    detail: float = 0.7
    background: str = "transparent"

    def __post_init__(self) -> None:
        if self.preset not in {"artwork", "photo"}:
            raise VectorizationError("preset must be artwork or photo")
        if self.colors != 0 and not 2 <= self.colors <= 64:
            raise VectorizationError("colors must be 0 (auto) or between 2 and 64")
        if not 0 <= self.detail <= 1:
            raise VectorizationError("detail must be between 0 and 1")
        if self.background not in {"transparent", "white"} and not re.fullmatch(
            r"#[0-9a-fA-F]{6}", self.background
        ):
            raise VectorizationError("background must be transparent, white or #RRGGBB")


@dataclass(frozen=True)
class EngineLimits:
    """Hard bounds on decoded images and optimization complexity."""

    max_pixels: int = 24_000_000
    max_dimension: int = 512
    max_regions: int = 2000
    max_iterations: int = 30

    def __post_init__(self) -> None:
        if self.max_pixels < 1 or not 16 <= self.max_dimension <= 2048:
            raise ValueError("invalid image limits")
        if not 1 <= self.max_regions <= 4096 or not 0 <= self.max_iterations <= 200:
            raise ValueError("invalid engine limits")


@dataclass(frozen=True)
class VectorResult:
    """SVG text and JSON-compatible conversion measurements."""

    svg: str
    metadata: dict[str, Scalar] = field(default_factory=dict)
