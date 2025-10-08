"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel

# These corresponds to the private_attribute_entity_type_name of supported entity types.
TargetClass = Literal["Surface", "Edge", "GenericVolume", "GeometryBodyGroup"]


class Predicate(Flow360BaseModel):
    """
    Single predicate in a selector.
    """

    # For now only name matching is supported
    attribute: Literal["name"] = pd.Field("name", description="The attribute to match/filter on.")
    operator: Literal[
        "equals",
        "notEquals",
        "in",
        "notIn",
        "matches",
        "notMatches",
    ] = pd.Field()
    value: Union[str, List[str]] = pd.Field()
    # Applies only to matches/notMatches; default to glob if not specified explicitly.
    non_glob_syntax: Optional[Literal["regex"]] = pd.Field(
        None,
        description="If specified, the pattern (`value`) will be treated "
        "as a non-glob pattern with the specified syntax.",
    )


class EntitySelector(Flow360BaseModel):
    """Entity selector for an EntityList.

    - target_class chooses the entity pool
    - logic combines child predicates (AND = intersection, OR = union)
    """

    target_class: TargetClass = pd.Field()
    logic: Literal["AND", "OR"] = pd.Field("AND")
    children: List[Predicate] = pd.Field()


@dataclass
class EntityDictDatabase:
    """
    [Internal Use Only]
    Entity database for entity selectors.
    This is intended to strip off differences between root resources and
    ensure the expansion has a uniform data interface.

    Each data member maps between attribute used for matching and the entity raw JSON dictionary.
    """

    surfaces: list[dict] = field(default_factory=list)
    edges: list[dict] = field(default_factory=list)
    generic_volumes: list[dict] = field(default_factory=list)
    geometry_body_groups: list[dict] = field(default_factory=list)
