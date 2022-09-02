"""
Flow360 base Model
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Extra, Field


class Flow360BaseModel(BaseModel, extra=Extra.allow):
    """
    Flow360 base Model
    """

    name: str
    user_id: Optional[str] = Field(alias="userId")
    solver_version: Optional[str] = Field(alias="solverVersion")
    tags: Optional[List[str]]
    created_at: Optional[str] = Field(alias="createdAt")
    updated_at: Optional[datetime] = Field(alias="updatedAt")
    updated_by: Optional[str] = Field(alias="updatedBy")
    user_upload_file_name: Optional[str]
