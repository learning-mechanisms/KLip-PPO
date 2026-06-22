"""Shared Pydantic base for all configs."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Immutable, validated configuration model."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
        str_strip_whitespace=True,
        use_attribute_docstrings=True,
    )
