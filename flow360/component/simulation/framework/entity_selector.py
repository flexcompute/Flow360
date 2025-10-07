"""
Entity selector models

Defines a minimal, stable schema for selecting entities by rules.
"""

from typing import List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel

TargetClass = Literal["Surface", "GhostSurface", "Edge", "Volume"]


class Predicate(Flow360BaseModel):
    """
    Single predicate in a selector.
    """

    # For now only name matching is supported
    attribute: Literal["name"] = pd.Field("name")
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
    pattern_syntax: Optional[Literal["glob", "regex"]] = pd.Field("glob")


class EntitySelector(Flow360BaseModel):
    """Entity selector for an EntityList.

    - target_class chooses the entity pool
    - logic combines child predicates (AND = intersection, OR = union)
    """

    target_class: TargetClass = pd.Field()
    logic: Literal["AND", "OR"] = pd.Field("AND")
    children: List[Predicate] = pd.Field()
