"""Shared base-model config helpers."""

from __future__ import annotations

import pydantic as pd

from .base_model import Flow360BaseModel, snake_to_camel

base_model_config = pd.ConfigDict(**Flow360BaseModel.model_config)

__all__ = ["base_model_config", "snake_to_camel"]
