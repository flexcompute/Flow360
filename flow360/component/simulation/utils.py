"""Relay schema-owned utility functions for the simulation component."""

# pylint: disable=unused-import
from flow360_schema.framework.bounding_box import BoundingBox, BoundingBoxType
from flow360_schema.models.simulation.utils import (
    get_combined_subclasses,
    is_exact_instance,
    is_instance_of_type_in_union,
    sanitize_params_dict,
)
