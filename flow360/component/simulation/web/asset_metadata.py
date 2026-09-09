"""Base model for asset metadata"""

import pydantic.v1 as pd


class MetadataBaseModel(pd.BaseModel):
    """Base model for asset metadata"""

    # pylint: disable=too-few-public-methods
    class Config:
        """Pydantic model configuration"""

        extra = pd.Extra.ignore
        allow_mutation = True
        allow_population_by_field_name = True
        validate_assignment = True
        validate_all = True
