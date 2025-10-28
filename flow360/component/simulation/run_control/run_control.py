"""Module for the run control settings of simulation."""

from typing import List, Literal, Optional

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.run_control.stopping_criterion import (
    StoppingCriterion,
)


class RunControl(Flow360BaseModel):
    """
    :class:`RunControl` class for run control settings.

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
