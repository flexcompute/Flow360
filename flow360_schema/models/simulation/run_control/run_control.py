"""Module for the run control settings of simulation."""

from typing import Literal

import pydantic as pd

from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.models.simulation.run_control.stopping_criterion import StoppingCriterion


class RunControl(Flow360BaseModel):
    """
    :class:`RunControl` class for run control settings.
    For general overview of run control see :ref:`Run Control <runControl>`.

    Example
    -------

    >>> criterion = fl.StoppingCriterion(...)
    >>> fl.RunControl(
    ...     stopping_criteria = [criterion],
    ... )

    ====
    """

    stopping_criteria: list[StoppingCriterion] | None = pd.Field(
        None,
        description="A list of :class:`StoppingCriterion` for the solver. "
        "All criteria must be met at the same time to stop the solver.",
    )
    type_name: Literal["RunControl"] = pd.Field("RunControl", frozen=True)
