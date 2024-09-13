import contextvars
from pydantic import Field
from typing import Optional

SURFACE_MESH = "surface_mesh"
VOLUME_MESH = "volume_mesh"
CASE = "case"
# when running validation with ALL, it will report errors happing in all scenarios in one validation pass
ALL = "all"

_validation_level_ctx = contextvars.ContextVar("validation_level", default=None)

class ValidationLevelContext:
    def __init__(self, level: str):
        if level not in {None, SURFACE_MESH, VOLUME_MESH, CASE, ALL}:
            raise ValueError(f"Invalid validation level: {level}")
        self.level = level
        self.token = None

    def __enter__(self):
        self.token = _validation_level_ctx.set(self.level)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _validation_level_ctx.reset(self.token)

def get_validation_level():
    return _validation_level_ctx.get()


def ConditionallyRequiredField(default=None, *, required_for: Optional[str] = None, **kwargs):
    # use this field for required fields but for only certain scenario, for example volume meshing only
    return Field(default, json_schema_extra=dict(required_for=required_for), **kwargs)


def ConditionalField(default=None, *, relevant_for: Optional[str] = None, **kwargs):
    # use this field for fields that are not required but make sense in certain scenario, for example UDD for case
    return Field(default, json_schema_extra=dict(relevant_for=relevant_for), **kwargs)


def CaseField(default=None, **kwargs):
    # use this field for fields that are not required but make sense only for Case
    return ConditionalField(default, relevant_for=CASE, **kwargs)