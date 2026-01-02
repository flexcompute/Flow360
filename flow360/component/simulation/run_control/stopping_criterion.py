"""Module for setting up the stopping criterion of simulation."""

from typing import List, Literal, Optional, Union

import pydantic as pd

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.param_utils import serialize_model_obj_to_id
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.output_fields import _FIELD_IS_SCALAR_MAPPING
from flow360.component.simulation.outputs.outputs import (
    MonitorOutputType,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceProbeOutput,
)
from flow360.component.simulation.unit_system import u
from flow360.component.simulation.user_code.core.types import (
    SolverVariable,
    UnytQuantity,
    UserVariable,
    ValueOrExpression,
    get_input_value_dimensions,
    get_input_value_length,
    infer_units_by_unit_system,
    is_variable_with_unit_system_as_units,
    solver_variable_to_user_variable,
)
from flow360.component.simulation.validation.validation_context import (
    ParamsValidationInfo,
    contextual_field_validator,
)


class StoppingCriterion(Flow360BaseModel):
    """

    :class:`StoppingCriterion` class for :py:attr:`RunControl.stopping_criteria` settings.

    Example
    -------

    Define a stopping criterion on a :class:`ProbeOutput` with a tolerance of 0.01.
    The ProbeOutput monitors the moving range of Helicity in a moving window of 10 steps,
    at the location of (0, 0, 0,005) * fl.u.m.

    >>> monitored_variable = fl.UserVariable(
    ...     name="Helicity_user",
    ...     value=fl.math.dot(fl.solution.velocity, fl.solution.vorticity),
    ... )
    >>> criterion = fl.StoppingCriterion(
    ...     name="Criterion_1",
    ...     monitor_output=fl.ProbeOutput(
    ...         name="Helicity_probe",
    ...         output_fields=[
    ...             monitored_variable,
    ...         ],
    ...         probe_points=fl.Point(name="Point1", location=(0, 0, 0.005) * fl.u.m),
    ...         moving_statistic = fl.MovingStatistic(method = "range", moving_window_size = 10)
    ...     ),
    ...     monitor_field=monitored_variable,
    ...     tolerance=0.01,
    ... )

    ====
    """

    name: Optional[str] = pd.Field("StoppingCriterion", description="Name of this criterion.")
    monitor_field: Union[UserVariable, str] = pd.Field(
        description="The field to be monitored. This field must be "
        "present in the `output_fields` of `monitor_output`."
    )
    monitor_output: Union[MonitorOutputType, str] = pd.Field(
        description="The monitored output or its id."
    )
    tolerance: ValueOrExpression[Union[UnytQuantity, float]] = pd.Field(
        description="The tolerance threshold of this criterion."
    )
    tolerance_window_size: Optional[int] = pd.Field(
        None,
        description="The number of data points from the monitor_output to be used to check whether "
        "the :math:`|max-min|/2` of the monitored field within this window is below tolerance or not. "
        "If not set, the criterion will directly compare the latest value with tolerance.",
        ge=2,
    )
    type_name: Literal["StoppingCriterion"] = pd.Field("StoppingCriterion", frozen=True)

    def preprocess(
        self,
        *,
        params=None,
        exclude: List[str] = None,
        required_by: List[str] = None,
        flow360_unit_system=None,
    ) -> Flow360BaseModel:
        exclude_criterion = exclude + ["tolerance"]
        return super().preprocess(
            params=params,
            exclude=exclude_criterion,
            required_by=required_by,
            flow360_unit_system=flow360_unit_system,
        )

    @pd.field_validator("monitor_field", mode="before")
    @classmethod
    def _convert_solver_variable_as_user_variable(cls, value):
        if isinstance(value, SolverVariable):
            return solver_variable_to_user_variable(value)
        return value

    @pd.field_validator("monitor_field", mode="after")
    @classmethod
    def _check_monitor_field_is_scalar(cls, v):
        if (isinstance(v, UserVariable) and get_input_value_length(v.value) != 0) or (
            isinstance(v, str) and v in _FIELD_IS_SCALAR_MAPPING and not _FIELD_IS_SCALAR_MAPPING[v]
        ):
            raise ValueError("The stopping criterion can only be defined on a scalar field.")
        return v

    @pd.field_validator("monitor_output", mode="after")
    @classmethod
    def _convert_monitor_output_obj_to_id(cls, value):
        """Convert monitor_output object to id"""
        if isinstance(value, str):
            return value
        return serialize_model_obj_to_id(model_obj=value)

    @contextual_field_validator("monitor_output", mode="after", required_context=["output_dict"])
    @classmethod
    def _check_monitor_exists_in_output_list(cls, v, param_info: ParamsValidationInfo):
        """Ensure the monitor output exist in the outputs list of SimulationParams."""
        # output_dict is None if outputs field had validation errors
        if param_info.output_dict.get(v) is None:
            raise ValueError("The monitor output does not exist in the outputs list.")
        return v

    @contextual_field_validator("monitor_output", mode="after", required_context=["output_dict"])
    @classmethod
    def _check_single_point_in_probe_output(cls, v, param_info: ParamsValidationInfo):
        monitor_output = param_info.output_dict.get(v)
        if not isinstance(monitor_output, (ProbeOutput, SurfaceProbeOutput)):
            return v
        if len(monitor_output.entities.stored_entities) == 1 and isinstance(
            monitor_output.entities.stored_entities[0], Point
        ):
            return v
        raise ValueError(
            "For stopping criterion setup, only one single `Point` entity is allowed "
            "in `ProbeOutput`/`SurfaceProbeOutput`."
        )

    @contextual_field_validator("monitor_output", mode="after", required_context=["output_dict"])
    @classmethod
    def _check_field_exists_in_monitor_output(
        cls, v, info: pd.ValidationInfo, param_info: ParamsValidationInfo
    ):
        """Ensure the monitor field exist in the monitor output."""
        monitor_output = param_info.output_dict.get(v)
        monitor_field = info.data.get("monitor_field", None)
        if monitor_field not in monitor_output.output_fields.items:
            raise ValueError("The monitor field does not exist in the monitor output.")
        return v

    @contextual_field_validator("tolerance", mode="before", required_context=["output_dict"])
    @classmethod
    def _preprocess_field_with_unit_system(
        cls, value, info: pd.ValidationInfo, param_info: ParamsValidationInfo
    ):
        if is_variable_with_unit_system_as_units(value):
            return value
        if info.data.get("monitor_field") is None:
            # `field` validation failed.
            raise ValueError(
                "The monitor field is invalid and therefore unit inference is not possible."
            )
        if info.data.get("monitor_output") is None:
            raise ValueError(
                "The monitor output is invalid and therefore unit inference is not possible."
            )
        units = value["units"]
        monitor_field = info.data["monitor_field"]
        monitor_output = param_info.output_dict.get(info.data.get("monitor_output"))
        field_dimensions = get_input_value_dimensions(value=monitor_field.value)
        if isinstance(monitor_output, SurfaceIntegralOutput):
            field_dimensions = field_dimensions * u.dimensions.length**2
        value = infer_units_by_unit_system(
            value=value, value_dimensions=field_dimensions, unit_system=units
        )
        return value

    @pd.field_validator("tolerance", mode="after")
    @classmethod
    def check_tolerance_value_for_string_monitor_field(cls, v, info: pd.ValidationInfo):
        """Ensure the tolerance is float when string field is used."""

        monitor_field = info.data.get("monitor_field", None)
        if isinstance(monitor_field, str) and not isinstance(v, float):
            raise ValueError(
                f"The monitor field ({monitor_field}) specified by string "
                "can only be used with a nondimensional tolerance."
            )
        return v

    @contextual_field_validator("tolerance", mode="after", required_context=["output_dict"])
    @classmethod
    def _check_tolerance_and_monitor_field_match_dimensions(
        cls, v, info: pd.ValidationInfo, param_info: ParamsValidationInfo
    ):
        """Ensure the tolerance has the same dimensions as the monitor field."""
        monitor_field = info.data.get("monitor_field", None)
        if not isinstance(monitor_field, UserVariable):
            return v
        field_dimensions = get_input_value_dimensions(value=monitor_field.value)
        monitor_output = param_info.output_dict.get(info.data.get("monitor_output", None), None)
        if isinstance(monitor_output, SurfaceIntegralOutput):
            field_dimensions = field_dimensions * u.dimensions.length**2
        tolerance_dimensions = get_input_value_dimensions(value=v)
        if tolerance_dimensions != field_dimensions:
            raise ValueError("The dimensions of monitor field and tolerance do not match.")
        return v
