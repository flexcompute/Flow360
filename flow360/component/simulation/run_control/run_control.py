"""Relay run-control model definitions from schema."""

# pylint: disable=unused-import

<<<<<<< HEAD
from flow360_schema.models.simulation.run_control.run_control import RunControl
=======
import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.run_control.stopping_criterion import (
    StoppingCriterion,
)


class RunControl(Flow360BaseModel):
    """
    :class:`RunControl` class for run control settings.
    For general overview see :ref:`Run Control <runControl>`.

    Example
    -------

    >>> criterion = fl.StoppingCriterion(...)
    >>> fl.RunControl(
    ...     stopping_criteria = [criterion],
    ... )

    ====
    """

    stopping_criteria: Optional[List[StoppingCriterion]] = pd.Field(
        None,
        description="A list of :class:`StoppingCriterion` for the solver. "
        "All criteria must be met at the same time to stop the solver.",
    )
    type_name: Literal["RunControl"] = pd.Field("RunControl", frozen=True)
>>>>>>> 39901da4 ([Hotfix 25.9]: [FXC-7659] Added user guide ref to run control docstrings (#2044))
