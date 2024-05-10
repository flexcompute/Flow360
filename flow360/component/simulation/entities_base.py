from abc import ABCMeta
from typing import Any, List

import pydantic as pd

from flow360.component.simulation.base_model import Flow360BaseModel


class EntitiesBase(Flow360BaseModel, metaclass=ABCMeta):
    """Abstraction of `entities` implementation.
    Goal:
    - Allow user to apply same global sets of Fields to all given entities.
    - Allow user to select multiple managed entites with matching name pattern and apply changes.

    Depending on subclass, the `entities` attribute will manage different types of entities.
    But most likely the managed types are zones, surfaces or edges.

    One difficulty is how we share the fields that belong to two different classes.
    E.g.
    ```python
        my_rot_zone = CylindricalZone(axis = (1,0,0))
        Rotation(entities=[my_rot_zone], angular_velocity = 0.2)
        SomeOtherModel(entities=[my_rot_zone])
    ```
    How do I access the `angular_velocity` inside `SomeOtherModel`? Is it possible at all?
    """

    entities: List[Any] = pd.Field()

    def by_name(pattern):
        """Returns a list of managed entities whose name matches the given pattern."""
        pass
