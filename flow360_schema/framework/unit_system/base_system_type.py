"""Base unit system type enumeration for Flow360 schemas."""

from enum import Enum


class BaseSystemType(Enum):
    """Type of the base unit system to use for unit inference."""

    SI = "SI"
    CGS = "CGS"
    IMPERIAL = "Imperial"
    FLOW360 = "Flow360"
