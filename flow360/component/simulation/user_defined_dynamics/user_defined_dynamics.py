"""User defined dynamic model for SimulationParams"""

from typing import Dict, List, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.framework.expressions import StringExpression
from flow360.component.simulation.primitives import Cylinder, GenericVolume, Surface
from flow360.component.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360.component.simulation.validation.validation_utils import (
    check_deleted_surface_in_entity_list,
)


class UserDefinedDynamic(Flow360BaseModel):
    """
    :class:`UserDefinedDynamic` class for defining the user defined dynamics inputs.

    Example
    -------
    The following example comes from the :ref:`User Defined Dynamics Tutorial Case <UDDGridRotation>`.
    Please refer to :ref:`this tutorial<userDefinedDynamics>` for more details about the User Defined Dynamics.

    >>> fl.UserDefinedDynamic(
    ...    name="dynamicTheta",
    ...    input_vars=["momentY"],
    ...    constants={
    ...        "I": 0.443768309310345,
    ...        "zeta": zeta,
    ...        "K": K,
    ...        "omegaN": omegaN,
    ...        "theta0": theta0,
    ...    },
    ...    output_vars={
    ...        "omegaDot": "state[0];",
    ...        "omega": "state[1];",
    ...        "theta": "state[2];",
    ...    },
    ...    state_vars_initial_value=[str(initOmegaDot), "0.0", str(initTheta)],
    ...    update_law=[
    ...        "if (pseudoStep == 0) (momentY - K * ( state[2] - theta0 ) "
    ...         + "- 2 * zeta * omegaN * I *state[1] ) / I; else state[0];",
    ...        "if (pseudoStep == 0) state[1] + state[0] * timeStepSize; else state[1];",
    ...        "if (pseudoStep == 0) state[2] + state[1] * timeStepSize; else state[2];",
    ...    ],
    ...    input_boundary_patches=volume_mesh["plateBlock/noSlipWall"],
    ...    output_target=volume_mesh["plateBlock"],
    ... )

    ====

    """

    name: str = pd.Field(
        "User defined dynamics", description="Name of the dynamics defined by the user."
    )
    input_vars: List[str] = pd.Field(
        description="List of the inputs to define the user defined dynamics. For example :code:`CL`, :code:`CD`, "
        + ":code:`bet_NUM_torque`,  :code:`bet_NUM_thrust`, (NUM is the index of the BET disk starting from 0), "
        + ":code:`momentX`, :code:`momentY`, :code:`momentZ` (X/Y/Z moments with respect to "
        + ":py:attr:`~ReferenceGeometry.moment_center`), :code:`forceX`, :code:`forceY`, :code:`forceZ`. "
        + "For a full list of supported variable, see :ref:`here <SupportedVariablesInUserExpression_>`."
    )
    constants: Optional[Dict[str, float]] = pd.Field(
        None, description="A list of constants that can be used in the expressions."
    )
    output_vars: Optional[Dict[str, StringExpression]] = pd.Field(
        None,
        description="Name of the output variables and the expression for the output variables using state "
        + "variables. For example :code:`alphaAngle`, :code:`betaAngle`, :code:`bet_NUM_omega` (NUM is the index "
        + "of the BET disk starting from 0), :code:`theta`, :code:`omega` and  :code:`omegaDot` (rotation angle/"
        + "velocity/acceleration in radians for sliding interfaces). For a full list of supported variable, see "
        + ":ref:`here <SupportedVariablesInUserExpression_>`. Please exercise caution when choosing output "
        + "variables, as any modifications to their values will be directly mirrored in the solver. Expressions "
        + "follows similar guidelines as :ref:`User Defined Expressions<userDefinedExpressionsKnowledgeBase>`.",
    )
    state_vars_initial_value: List[StringExpression] = pd.Field(
        description="The initial value of state variables are specified here. The entries could be either values "
        + "(in the form of strings, e.g., :code:`0.0`) or expression with constants defined earlier or any input "
        + "and output variable. (e.g., :code:`2.0 * alphaAngle + someConstant`). The list entries correspond to "
        + "the initial values for :code:`state[0]`, :code:`state[1]`, ..., respectively."
    )
    update_law: List[StringExpression] = pd.Field(
        "List of expressions for updating state variables. The list entries correspond to the update laws for "
        + ":code:`state[0]`, :code:`state[1]`, ..., respectively. These expressions follows similar guidelines as "
        + ":ref:`user Defined Expressions<userDefinedExpressionsKnowledgeBase>`."
    )
    input_boundary_patches: Optional[EntityList[Surface]] = pd.Field(
        None,
        description="The list of :class:`~flow360.Surface` entities to which the input variables belongs. "
        + "If multiple boundaries are specified then the summation over the boundaries are used as the input. "
        + "For input variables that already specified the source in the name (like bet_NUM_torque) "
        + "this entry does not have any effect.",
    )
    output_target: Optional[Union[Cylinder, GenericVolume, Surface]] = pd.Field(
        None,
        description="The target to which the output variables belong to. For example this can be the rotating "
        + "volume zone name. Only one output target is supported per user defined dynamics instance. Only "
        + ":class:`~flow360.Cylinder` entity is supported as target for now.",
    )  # Limited to `Cylinder` for now as we have only tested using UDD to control rotation.

    @pd.field_validator("input_boundary_patches", mode="after")
    @classmethod
    def ensure_surface_existence(cls, value):
        """Ensure all boundaries will be present after mesher"""
        if value is None:
            return value
        return check_deleted_surface_in_entity_list(value)

    @pd.field_validator("output_target", mode="after")
    @classmethod
    def ensure_output_surface_existence(cls, value):
        """Ensure that the output target surface is not a deleted surface"""
        validation_info = get_validation_info()
        if validation_info is None or validation_info.auto_farfield_method is None:
            # validation not necessary now.
            return value

        # - Check if the surfaces are deleted.
        # pylint: disable=protected-access
        if isinstance(value, Surface) and value._will_be_deleted_by_mesher(
            validation_info.auto_farfield_method
        ):
            raise ValueError(
                f"Boundary `{value.name}` will likely be deleted after mesh generation. Therefore it cannot be used."
            )
        return value
