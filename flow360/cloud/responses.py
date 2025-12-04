"""Response models for Flow360 API responses"""

from typing import List, Optional

import pydantic as pd_v2


class CopyExampleResponse(pd_v2.BaseModel):
    """Response model for copying an example"""

    id: str = pd_v2.Field(description="Project ID created from the example")


class ExampleItem(pd_v2.BaseModel):
    """Response model for a single example item"""

    tags: List[str] = pd_v2.Field(description="Tags associated with the example")
    id: str = pd_v2.Field(description="Example ID")
    type: str = pd_v2.Field(description="Type of the example resource")
    resource_id: str = pd_v2.Field(alias="resourceId", description="Project ID of the example")
    s3path: str = pd_v2.Field(description="S3 path to the example image")
    title: str = pd_v2.Field(description="Title of the example")
    created_at: str = pd_v2.Field(alias="createdAt", description="Creation timestamp")
    order: int = pd_v2.Field(description="Display order of the example")


class ExamplesListResponse(pd_v2.BaseModel):
    """Response model for the examples list API"""

    data: List[ExampleItem] = pd_v2.Field(description="List of available examples")
    warning: Optional[str] = pd_v2.Field(default=None, description="Warning message if any")

