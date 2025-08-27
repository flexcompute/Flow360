"""
Contains basic components(solvers) that composes the `volume` type models.
Each volume model represents a physical phenomena that require a combination of solver features to model.

E.g.
NavierStokes, turbulence and transition composes FluidDynamics `volume` type

"""

from __future__ import annotations

from abc import ABCMeta
from typing import Annotated, Dict, List, Literal, Optional, Union

import numpy as np
import pydantic as pd
from pydantic import NonNegativeFloat, NonNegativeInt, PositiveFloat, PositiveInt
from typing_extensions import Self

from flow360.component.simulation.framework.base_model import (
    Conflicts,
    Flow360BaseModel,
)
from flow360.component.simulation.framework.entity_base import EntityList
from flow360.component.simulation.primitives import Box, GenericVolume

# from .time_stepping import UnsteadyTimeStepping

HEAT_EQUATION_EVAL_MAX_PER_PSEUDOSTEP_UNSTEADY = 40
HEAT_EQUATION_EVALUATION_FREQUENCY_STEADY = 10


class LinearSolver(Flow360BaseModel):
    """:class:`LinearSolver` class for setting up the linear solver.

    Example
    -------
    >>> fl.LinearSolver(
    ...     max_iterations=50,
    ...     absoluteTolerance=1e-10
    ... )
    """

    max_iterations: PositiveInt = pd.Field(
        30, description="Maximum number of linear solver iterations."
    )
    absolute_tolerance: Optional[PositiveFloat] = pd.Field(
        None,
        description="The linear solver converges when the final residual of the pseudo steps below this value."
        + "Either absolute tolerance or relative tolerance can be used to determine convergence.",
    )
    relative_tolerance: Optional[PositiveFloat] = pd.Field(
        None,
        description="The linear solver converges when the ratio of the final residual and the initial "
        + "residual of the pseudo step is below this value.",
    )

    model_config = pd.ConfigDict(
        conflicting_fields=[Conflicts(field1="absolute_tolerance", field2="relative_tolerance")]
    )


class GenericSolverSettings(Flow360BaseModel, metaclass=ABCMeta):
    """:class:`GenericSolverSettings` class"""

    absolute_tolerance: PositiveFloat = pd.Field(1.0e-10)
    relative_tolerance: NonNegativeFloat = pd.Field(
        0,
        description="Tolerance to the relative residual, below which the solver goes to the next physical step. "
        + "Relative residual is defined as the ratio of the current pseudoStep's residual to the maximum "
        + "residual present in the first 10 pseudoSteps within the current physicalStep. "
        + "NOTE: relativeTolerance is ignored in steady simulations and only absoluteTolerance is "
        + "used as the convergence criterion.",
    )
    order_of_accuracy: Literal[1, 2] = pd.Field(2, description="Order of accuracy in space.")
    equation_evaluation_frequency: PositiveInt = pd.Field(
        1, description="Frequency at which to solve the equation."
    )
    linear_solver: LinearSolver = pd.Field(LinearSolver())
    private_attribute_dict: Optional[Dict] = pd.Field(None)


class NavierStokesSolver(GenericSolverSettings):
    """:class:`NavierStokesSolver` class for setting up the compressible Navier-Stokes solver.
    For more information on setting up the numerical parameters for the Navier-Stokes solver,
    refer to :ref:`Navier-Stokes solver knowledge base <knowledge_base_navierStokesSolver>`.

    Example
    -------
    >>> fl.NavierStokesSolver(
    ...     absolute_tolerance=1e-10,
    ...     numerical_dissipation_factor=0.01,
    ...     linear_solver=LinearSolver(max_iterations=50),
    ...     low_mach_preconditioner=True,
    ... )
    """

    absolute_tolerance: PositiveFloat = pd.Field(
        1.0e-10,
        description="Tolerance for the NS residual, below which the solver goes to the next physical step.",
    )

    CFL_multiplier: PositiveFloat = pd.Field(
        1.0,
        description="Factor to the CFL definitions defined in the "
        + ":ref:`Time Stepping <timeStepping>` section.",
    )
    kappa_MUSCL: pd.confloat(ge=-1, le=1) = pd.Field(
        -1,
        description="Kappa for the MUSCL scheme, range from [-1, 1], with 1 being unstable. "
        + "The default value of -1 leads to a 2nd order upwind scheme and is the most stable. "
        + "A value of 0.33 leads to a blended upwind/central scheme and is recommended for low "
        + "subsonic flows leading to reduced dissipation.",
    )

    numerical_dissipation_factor: pd.confloat(ge=0.01, le=1) = pd.Field(
        1,
        description="A factor in the range [0.01, 1.0] which exponentially reduces the "
        + "dissipation of the numerical flux. The recommended starting value for most "
        + "low-dissipation runs is 0.2.",
    )
    limit_velocity: bool = pd.Field(False, description="Limiter for velocity")
    limit_pressure_density: bool = pd.Field(False, description="Limiter for pressure and density.")

    type_name: Literal["Compressible"] = pd.Field("Compressible", frozen=True)

    low_mach_preconditioner: bool = pd.Field(
        False, description="Use preconditioning for accelerating low Mach number flows."
    )
    low_mach_preconditioner_threshold: Optional[NonNegativeFloat] = pd.Field(
        None,
        description="For flow regions with Mach numbers smaller than threshold, the input "
        + "Mach number to the preconditioner is assumed to be the threshold value if it is "
        + "smaller than the threshold. The default value for the threshold is the freestream "
        + "Mach number.",
    )

    update_jacobian_frequency: PositiveInt = pd.Field(
        4, description="Frequency at which the jacobian is updated."
    )
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(
        0,
        description="When physical step is less than this value, the jacobian matrix is "
        + "updated every pseudo step.",
    )


class SpalartAllmarasModelConstants(Flow360BaseModel):
    """
    :class:`SpalartAllmarasModelConstants` class specifies the constants of the Spalart-Allmaras model.

    Example
    -------
    >>> fl.SpalartAllmaras(
    ...     modeling_constants = SpalartAllmarasModelConstants(C_w2=2.718)
    ... )
    """

    type_name: Literal["SpalartAllmarasConsts"] = pd.Field("SpalartAllmarasConsts", frozen=True)
    C_DES: NonNegativeFloat = pd.Field(0.72)
    C_d: NonNegativeFloat = pd.Field(8.0)
    C_cb1: NonNegativeFloat = pd.Field(0.1355)
    C_cb2: NonNegativeFloat = pd.Field(0.622)
    C_sigma: NonNegativeFloat = pd.Field(2.0 / 3.0)
    C_v1: NonNegativeFloat = pd.Field(7.1)
    C_vonKarman: NonNegativeFloat = pd.Field(0.41)
    C_w2: float = pd.Field(0.3)
    C_t3: NonNegativeFloat = pd.Field(1.2)
    C_t4: NonNegativeFloat = pd.Field(0.5)
    C_min_rd: NonNegativeFloat = pd.Field(10.0)


class KOmegaSSTModelConstants(Flow360BaseModel):
    """
    :class:`KOmegaSSTModelConstants` class specifies the constants of the SST k-omega model.

    Example
    -------
    >>> fl.KOmegaSST(
    ...     modeling_constants = KOmegaSSTModelConstants(C_sigma_omega1=2.718)
    ... )
    """

    type_name: Literal["kOmegaSSTConsts"] = pd.Field("kOmegaSSTConsts", frozen=True)
    C_DES1: NonNegativeFloat = pd.Field(0.78)
    C_DES2: NonNegativeFloat = pd.Field(0.61)
    C_d1: NonNegativeFloat = pd.Field(20.0)
    C_d2: NonNegativeFloat = pd.Field(3.0)
    C_sigma_k1: NonNegativeFloat = pd.Field(0.85)
    C_sigma_k2: NonNegativeFloat = pd.Field(1.0)
    C_sigma_omega1: NonNegativeFloat = pd.Field(0.5)
    C_sigma_omega2: NonNegativeFloat = pd.Field(0.856)
    C_alpha1: NonNegativeFloat = pd.Field(0.31)
    C_beta1: NonNegativeFloat = pd.Field(0.075)
    C_beta2: NonNegativeFloat = pd.Field(0.0828)
    C_beta_star: NonNegativeFloat = pd.Field(0.09)


TurbulenceModelConstants = Annotated[
    Union[SpalartAllmarasModelConstants, KOmegaSSTModelConstants],
    pd.Field(discriminator="type_name"),
]


class TurbulenceModelControls(Flow360BaseModel):
    """
    :class:`TurbulenceModelControls` class specifies modeling constants and enforces turbulence model
    behavior on a zonal basis, as defined by mesh entities or boxes in space. These controls
    supersede the global turbulence model solver settings.

    Example
    _______
    >>> fl.TurbulenceModelControls(
    ...     modeling_constants=fl.SpalartAllmarasConstants(C_w2=2.718),
    ...     enforcement="RANS",
    ...     entities=[
    ...         volume_mesh["block-1"],
    ...         fl.Box.from_principal_axes(
    ...             name="box",
    ...             axes=[(0, 1, 0), (0, 0, 1)],
    ...             center=(0, 0, 0) * fl.u.m,
    ...             size=(0.2, 0.3, 2) * fl.u.m,
    ...         ),
    ...     ],
    ... )
    """

    modeling_constants: Optional[TurbulenceModelConstants] = pd.Field(
        None,
        description="A class of `SpalartAllmarasModelConstants` or `KOmegaSSTModelConstants`  used to "
        + "specify constants in specific regions of the domain.",
    )

    enforcement: Optional[Literal["RANS", "LES"]] = pd.Field(
        None, description="Force RANS or LES mode in a specific control region."
    )

    entities: EntityList[GenericVolume, Box] = pd.Field(
        alias="volumes",
        description="The entity in which to apply the `TurbulenceMOdelControls``. "
        + "The entity should be defined by :class:`Box` or zones from the geometry/volume mesh."
        + "The axes of entity must be specified to serve as the the principle axes of the "
        + "`TurbulenceModelControls` region.",
    )


class DetachedEddySimulation(Flow360BaseModel):
    """
    :class:`DetachedEddySimulation` class is used for running hybrid RANS-LES simulations
    "It is supported for both SpalartAllmaras and kOmegaSST turbulence models, with and"
    "without AmplificationFactorTransport transition model enabled."

    Example
    -------
    >>> fl.SpalartAllmaras(
    ...     hybrid_model = DetachedEddySimulation(shielding_function = 'ZDES', grid_size_for_LES = 'maxEdgeLength')
    ... )
    """

    shielding_function: Literal["DDES", "ZDES"] = pd.Field(
        "DDES",
        description="Specifies the type of shielding used for the detached eddy simulation. The allowed inputs are"
        ":code:`DDES` (Delayed Detached Eddy Simulation proposed by Spalart 2006) and :code:`ZDES`"
        "(proposed by Deck and Renard 2020).",
    )
    grid_size_for_LES: Literal["maxEdgeLength", "meanEdgeLength"] = pd.Field(
        "maxEdgeLength",
        description="Specifies the length used for the computation of LES length scale. "
        + "The allowed inputs are :code:`maxEdgeLength` and :code:`meanEdgeLength`.",
    )


class TurbulenceModelSolver(GenericSolverSettings, metaclass=ABCMeta):
    """:class:`TurbulenceModelSolver` class for setting up turbulence model solver.
    For more information on setting up the numerical parameters for the turbulence model solver,
    refer to :ref:`the turbulence model solver knowledge base <knowledge_base_turbulenceModelSolver>`.

    Example
    -------
    >>> fl.TurbulenceModelSolver(absolute_tolerance=1e-10)
    """

    CFL_multiplier: PositiveFloat = pd.Field(
        2.0,
        description="Factor to the CFL definitions defined in the "
        + ":ref:`Time Stepping <timeStepping>` section.",
    )
    type_name: str = pd.Field(
        description=":code:`SpalartAllmaras`, :code:`kOmegaSST`, or :code:`None`."
    )
    absolute_tolerance: PositiveFloat = pd.Field(
        1e-8,
        description="Tolerance for the turbulence model residual, below which the solver progresses to the "
        + "next physical step (unsteady) or completes the simulation (steady).",
    )
    equation_evaluation_frequency: PositiveInt = pd.Field(
        4, description="Frequency at which to update the turbulence equation."
    )
    reconstruction_gradient_limiter: pd.confloat(ge=0, le=2) = pd.Field(
        1.0,
        description="The strength of gradient limiter used in reconstruction of solution "
        + "variables at the faces (specified in the range [0.0, 2.0]). 0.0 corresponds to "
        + "setting the gradient equal to zero, and 2.0 means no limiting.",
    )
    quadratic_constitutive_relation: bool = pd.Field(
        False,
        description="Use quadratic constitutive relation for turbulence shear stress tensor "
        + "instead of Boussinesq Approximation.",
    )
    modeling_constants: Optional[TurbulenceModelConstants] = pd.Field(
        discriminator="type_name",
        description=" A :class:`TurbulenceModelConstants` object containing the DDES coefficients "
        + "in the solver: **SpalartAllmaras**: :code:`C_DES` (= 0.72), :code:`C_d` (= 8.0),"
        + '**kOmegaSST**: :code:`"C_DES1"` (= 0.78), '
        + ":code:`C_DES2` (= 0.61), :code:`C_d1` (= 20.0), :code:`C_d2` (= 3.0), "
        + "*(values shown in the parentheses are the default values used in Flow360).*",
    )
    update_jacobian_frequency: PositiveInt = pd.Field(
        4, description="Frequency at which the jacobian is updated."
    )
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(
        0,
        description="For physical steps less than the input value, the jacobian matrix is "
        + "updated every pseudo-step overriding the :py:attr:`update_jacobian_frequency` value.",
    )

    linear_solver: LinearSolver = pd.Field(
        LinearSolver(max_iterations=20),
        description="Linear solver settings, see :class:`LinearSolver` documentation.",
    )

    hybrid_model: Optional[DetachedEddySimulation] = pd.Field(
        None, description="Model used for running hybrid RANS-LES simulations"
    )

    rotation_correction: bool = pd.Field(
        False, description="Rotation correction for the turbulence model."
    )

    controls: Optional[List[TurbulenceModelControls]] = pd.Field(
        None,
        strict=True,  # Note: To ensure propoer err msg when none-list is fed.
        description="List of control zones to enforce specific turbulence model constants "
        + "and behavior.",
    )

    @pd.model_validator(mode="after")
    def _check_zonal_modeling_constants_consistency(self) -> Self:
        if self.controls is None:
            return self

        for index, control in enumerate(self.controls):
            if control.modeling_constants is None:
                continue
            if not isinstance(
                control.modeling_constants, SpalartAllmarasModelConstants
            ) and isinstance(self, SpalartAllmaras):
                raise ValueError(
                    "Turbulence model is SpalartAllmaras, but controls.modeling"
                    "_constants is of a conflicting class "
                    f"in control region {index}."
                )
            if not isinstance(control.modeling_constants, KOmegaSSTModelConstants) and isinstance(
                self, KOmegaSST
            ):
                raise ValueError(
                    "Turbulence model is KOmegaSST, but controls.modeling_constants"
                    f" is of a conflicting class in control region {index}."
                )
        return self


class KOmegaSST(TurbulenceModelSolver):
    """
    :class:`KOmegaSST` class for setting up the turbulence solver based on the SST k-omega model.

    Example
    -------
    >>> fl.KOmegaSST(
    ...     absolute_tolerance=1e-10,
    ...     linear_solver=LinearSolver(max_iterations=25),
    ...     update_jacobian_frequency=2,
    ...     equation_evaluation_frequency=1,
    ... )
    """

    type_name: Literal["kOmegaSST"] = pd.Field("kOmegaSST", frozen=True)
    modeling_constants: KOmegaSSTModelConstants = pd.Field(
        KOmegaSSTModelConstants(),
        description="A :class:`KOmegaSSTModelConstants` object containing the coefficients "
        + "used in the SST k-omega model. For the default values used in Flow360, "
        + "please refer to :class:`KOmegaSSTModelConstants`.",
    )


class SpalartAllmaras(TurbulenceModelSolver):
    """
    :class:`SpalartAllmaras` class for setting up the turbulence solver based on the Spalart-Allmaras model.

    Example
    -------
    >>> fl.SpalartAllmaras(
    ...     absolute_tolerance=1e-10,
    ...     linear_solver=LinearSolver(max_iterations=25),
    ...     update_jacobian_frequency=2,
    ...     equation_evaluation_frequency=1,
    ... )
    """

    type_name: Literal["SpalartAllmaras"] = pd.Field("SpalartAllmaras", frozen=True)

    modeling_constants: Optional[SpalartAllmarasModelConstants] = pd.Field(
        SpalartAllmarasModelConstants(),
        description="A :class:`SpalartAllmarasModelConstants` object containing the coefficients "
        + "used in the Spalart-Allmaras model. For the default values used in Flow360, "
        + "please refer to :class:`SpalartAllmarasModelConstants`.",
    )
    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0, le=2)] = pd.Field(
        0.5,
        description="The strength of gradient limiter used in reconstruction of solution "
        + "variables at the faces (specified in the range [0.0, 2.0]). 0.0 corresponds to "
        + "setting the gradient equal to zero, and 2.0 means no limiting.",
    )


class NoneSolver(Flow360BaseModel):
    """:class:`NoneSolver` class for disabling the turbulence solver."""

    type_name: Literal["None"] = pd.Field("None", frozen=True)


TurbulenceModelSolverType = Annotated[
    Union[NoneSolver, SpalartAllmaras, KOmegaSST], pd.Field(discriminator="type_name")
]


class HeatEquationSolver(GenericSolverSettings):
    """:class:`HeatEquationSolver` class for setting up heat equation solver.

    Example
    -------
    >>> fl.HeatEquationSolver(
    ...     equation_evaluation_frequency=10,
    ...     linear_solver_config=LinearSolver(
    ...         max_iterations=50,
    ...         absoluteTolerance=1e-10
    ...     )
    ... )
    """

    type_name: Literal["HeatEquation"] = pd.Field("HeatEquation", frozen=True)
    absolute_tolerance: PositiveFloat = pd.Field(
        1e-9,
        description="Absolute residual tolerance that determines the convergence of the heat equation in "
        + "conjugate heat transfer. This value should be the same or higher than the absolute tolerance "
        + "for the linear solver by a small margin.",
    )
    equation_evaluation_frequency: PositiveInt = pd.Field(
        10,
        description="Frequency at which to solve the heat equation in conjugate heat transfer simulations.",
    )
    order_of_accuracy: Literal[2] = pd.Field(2, description="Order of accuracy in space.")

    linear_solver: LinearSolver = pd.Field(
        LinearSolver(max_iterations=50, absolute_tolerance=1e-10),
        description="Linear solver settings, see :class:`LinearSolver` documentation.",
    )


class TransitionModelSolver(GenericSolverSettings):
    """:class:`TransitionModelSolver` class for setting up transition model solver.
    For more information on setting up the numerical parameters for the transition model solver,
    refer to :ref:`the transition model solver knowledge base <knowledge_base_transitionModelSolver>`.

    Warning
    -------
    :py:attr:`N_crit` and :py:attr:`turbulence_intensity_percent` cannot be specified at the same time.

    Example
    -------
    >>> fl.TransitionModelSolver(
    ...     linear_solver=fl.LinearSolver(max_iterations=50),
    ...     absolute_tolerance=1e-8,
    ...     update_jacobian_frequency=1,
    ...     equation_evaluation_frequency=1,
    ...     turbulence_intensity_percent=0.04,
    ... )
    """

    model_config = pd.ConfigDict(
        conflicting_fields=[Conflicts(field1="N_crit", field2="turbulence_intensity_percent")]
    )

    type_name: Literal["AmplificationFactorTransport"] = pd.Field(
        "AmplificationFactorTransport", frozen=True
    )
    CFL_multiplier: PositiveFloat = pd.Field(
        2.0,
        description="Factor to the CFL definitions defined in the "
        + ":ref:`Time Stepping <timeStepping>` section.",
    )
    absolute_tolerance: PositiveFloat = pd.Field(
        1e-7,
        description="Tolerance for the transition model residual, below which the solver progresses to "
        + "the next physical step (unsteady) or completes the simulation (steady).",
    )
    equation_evaluation_frequency: PositiveInt = pd.Field(
        4, description="Frequency at which to update the transition equation."
    )
    turbulence_intensity_percent: Optional[pd.confloat(ge=0.03, le=2.5)] = pd.Field(
        None,
        description=":ref:`Turbulence Intensity <TurbI>`, Range from [0.03-2.5]. "
        + "Only valid when :py:attr:`N_crit` is not specified.",
    )
    # pylint: disable=invalid-name
    N_crit: Optional[pd.confloat(ge=1.0, le=11.0)] = pd.Field(
        None,
        description=":ref:`Critical Amplification Factor <NCrit>`, Range from [1-11]. "
        + "Only valid when :py:attr:`turbulence_intensity_percent` is not specified.",
    )
    update_jacobian_frequency: PositiveInt = pd.Field(
        4, description="Frequency at which the jacobian is updated."
    )
    max_force_jac_update_physical_steps: NonNegativeInt = pd.Field(
        0,
        description="For physical steps less than the input value, the jacobian matrix "
        + "is updated every pseudo-step overriding the :py:attr:`update_jacobian_frequency` value.",
    )

    reconstruction_gradient_limiter: Optional[pd.confloat(ge=0.0, le=2.0)] = pd.Field(
        1.0,
        description="The strength of gradient limiter used in reconstruction of solution "
        + "variables at the faces (specified in the range [0.0, 2.0]). 0.0 corresponds to "
        + "setting the gradient equal to zero, and 2.0 means no limiting.",
    )

    trip_regions: Optional[EntityList[Box]] = pd.Field(
        None, description="A list of :class:`~flow360.Box` entities defining the trip zones."
    )

    linear_solver: LinearSolver = pd.Field(
        LinearSolver(max_iterations=20),
        description="Linear solver settings, see :class:`LinearSolver` documentation.",
    )

    @pd.model_validator(mode="after")
    def _set_aft_ncrit(self) -> Self:
        """
        Compute the critical amplification factor for AFT transition solver based on
        input turbulence intensity and input Ncrit. Computing Ncrit from turbulence
        intensity takes priority if both are specified.
        """

        if self.turbulence_intensity_percent is not None:
            ncrit_converted = -8.43 - 2.4 * np.log(
                0.025 * np.tanh(self.turbulence_intensity_percent / 2.5)
            )
            self.turbulence_intensity_percent = None
            self.N_crit = ncrit_converted
        elif self.N_crit is None:
            self.N_crit = 8.15

        return self


TransitionModelSolverType = Annotated[
    Union[NoneSolver, TransitionModelSolver], pd.Field(discriminator="type_name")
]
