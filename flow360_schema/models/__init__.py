"""Flow360 schema model definitions.

Business models (solver params, operating conditions, etc.) live here.
Base infrastructure lives in ``framework/``.

Note: entity_info and entities are NOT re-exported here. Import directly
from concrete submodules:
  from flow360_schema.models.entity_info import ...
  from flow360_schema.models.entities.surface_entities import ...

ReferenceGeometry is also imported directly from
``flow360_schema.models.reference_geometry`` to keep package initialization
independent from the expression framework.
"""

from ..geometry.design_intent import CADComponent, CADDesignIntent, CADEntityRef, CADTreeNode

__all__ = [
    "CADComponent",
    "CADDesignIntent",
    "CADTreeNode",
    "CADEntityRef",
]
