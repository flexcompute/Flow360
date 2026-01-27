"""
Turbulence quantities parameters
"""

# pylint: disable=unused-import
from abc import ABCMeta
from functools import wraps
from typing import Annotated, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import (
    FrequencyType,
    KinematicViscosityType,
    LengthType,
    SpecificEnergyType,
)


class TurbulentKineticEnergy(Flow360BaseModel):
    """
    turbulentKineticEnergy : SpecificEnergyType [energy / mass]
        Turbulent kinetic energy. Applicable only when using SST model.
    """

    type_name: Literal["TurbulentKineticEnergy"] = pd.Field("TurbulentKineticEnergy", frozen=True)
    # pylint: disable=no-member
    turbulent_kinetic_energy: SpecificEnergyType.NonNegative = pd.Field()


class TurbulentIntensity(Flow360BaseModel):
    """
    turbulentIntensity : non-dimensional [`-`]
        Turbulent intensity. Applicable only when using SST model.
        This is related to turbulent kinetic energy as:
        `turbulentKineticEnergy = 1.5*pow(U_ref * turbulentIntensity, 2)`.
        Note the use of the freestream velocity U_ref instead of C_inf.
    """

    type_name: Literal["TurbulentIntensity"] = pd.Field("TurbulentIntensity", frozen=True)
    turbulent_intensity: pd.NonNegativeFloat = pd.Field()


class _SpecificDissipationRate(Flow360BaseModel, metaclass=ABCMeta):
    """
    specificDissipationRate : FrequencyType  [1 / time]
        Turbulent specific dissipation rate. Applicable only when using SST model.
    """

    type_name: Literal["SpecificDissipationRate"] = pd.Field("SpecificDissipationRate", frozen=True)
    # pylint: disable=no-member
    specific_dissipation_rate: FrequencyType.NonNegative = pd.Field()


class TurbulentViscosityRatio(Flow360BaseModel):
    """
    turbulentViscosityRatio : non-dimensional [`-`]
        The ratio of turbulent eddy viscosity over the freestream viscosity. Applicable for both SA and SST model.
    """

    type_name: Literal["TurbulentViscosityRatio"] = pd.Field("TurbulentViscosityRatio", frozen=True)
    turbulent_viscosity_ratio: pd.PositiveFloat = pd.Field()


class TurbulentLengthScale(Flow360BaseModel, metaclass=ABCMeta):
    """
    turbulentLengthScale : LengthType [length]
        The turbulent length scale is an estimation of the size of the eddies that are modeled/not resolved.
        Applicable only when using SST model. This is related to the turbulent kinetic energy and turbulent
        specific dissipation rate as: `L_T = sqrt(k)/(pow(beta_0^*, 0.25)*w)` where `L_T` is turbulent length scale,
        `k` is turbulent kinetic energy, `beta_0^*` is 0.09 and `w` is turbulent specific dissipation rate.
        Applicable only when using SST model.
    """

    type_name: Literal["TurbulentLengthScale"] = pd.Field("TurbulentLengthScale", frozen=True)
    # pylint: disable=no-member
    turbulent_length_scale: LengthType.Positive = pd.Field()


class ModifiedTurbulentViscosityRatio(Flow360BaseModel):
    """
    modifiedTurbulentViscosityRatio : non-dimensional [`-`]
        The ratio of modified turbulent eddy viscosity (SA) over the freestream viscosity.
        Applicable only when using SA model.
    """

    type_name: Literal["ModifiedTurbulentViscosityRatio"] = pd.Field(
        "ModifiedTurbulentViscosityRatio", frozen=True
    )
    modified_turbulent_viscosity_ratio: pd.PositiveFloat = pd.Field()


class ModifiedTurbulentViscosity(Flow360BaseModel):
    """
    modifiedTurbulentViscosity : KinematicViscosityType [length**2 / time]
        The modified turbulent eddy viscosity (SA). Applicable only when using SA model.
    """

    type_name: Literal["ModifiedTurbulentViscosity"] = pd.Field(
        "ModifiedTurbulentViscosity", frozen=True
    )
    # pylint: disable=no-member
    modified_turbulent_viscosity: Optional[KinematicViscosityType.Positive] = pd.Field()


# pylint: disable=missing-class-docstring
class SpecificDissipationRateAndTurbulentKineticEnergy(
    _SpecificDissipationRate, TurbulentKineticEnergy
):
    type_name: Literal["SpecificDissipationRateAndTurbulentKineticEnergy"] = pd.Field(
        "SpecificDissipationRateAndTurbulentKineticEnergy", frozen=True
    )


class TurbulentViscosityRatioAndTurbulentKineticEnergy(
    TurbulentViscosityRatio, TurbulentKineticEnergy
):
    type_name: Literal["TurbulentViscosityRatioAndTurbulentKineticEnergy"] = pd.Field(
        "TurbulentViscosityRatioAndTurbulentKineticEnergy", frozen=True
    )


class TurbulentLengthScaleAndTurbulentKineticEnergy(TurbulentLengthScale, TurbulentKineticEnergy):
    type_name: Literal["TurbulentLengthScaleAndTurbulentKineticEnergy"] = pd.Field(
        "TurbulentLengthScaleAndTurbulentKineticEnergy", frozen=True
    )


class TurbulentIntensityAndSpecificDissipationRate(TurbulentIntensity, _SpecificDissipationRate):
    type_name: Literal["TurbulentIntensityAndSpecificDissipationRate"] = pd.Field(
        "TurbulentIntensityAndSpecificDissipationRate", frozen=True
    )


class TurbulentIntensityAndTurbulentViscosityRatio(TurbulentIntensity, TurbulentViscosityRatio):
    type_name: Literal["TurbulentIntensityAndTurbulentViscosityRatio"] = pd.Field(
        "TurbulentIntensityAndTurbulentViscosityRatio", frozen=True
    )


class TurbulentIntensityAndTurbulentLengthScale(TurbulentIntensity, TurbulentLengthScale):
    type_name: Literal["TurbulentIntensityAndTurbulentLengthScale"] = pd.Field(
        "TurbulentIntensityAndTurbulentLengthScale", frozen=True
    )


class SpecificDissipationRateAndTurbulentViscosityRatio(
    _SpecificDissipationRate, TurbulentViscosityRatio
):
    type_name: Literal["SpecificDissipationRateAndTurbulentViscosityRatio"] = pd.Field(
        "SpecificDissipationRateAndTurbulentViscosityRatio", frozen=True
    )


class SpecificDissipationRateAndTurbulentLengthScale(
    _SpecificDissipationRate, TurbulentLengthScale
):
    type_name: Literal["SpecificDissipationRateAndTurbulentLengthScale"] = pd.Field(
        "SpecificDissipationRateAndTurbulentLengthScale", frozen=True
    )


class TurbulentViscosityRatioAndTurbulentLengthScale(TurbulentViscosityRatio, TurbulentLengthScale):
    type_name: Literal["TurbulentViscosityRatioAndTurbulentLengthScale"] = pd.Field(
        "TurbulentViscosityRatioAndTurbulentLengthScale", frozen=True
    )


# pylint: enable=missing-class-docstring
# pylint: disable=duplicate-code

TurbulenceQuantitiesType = Annotated[
    Union[
        TurbulentViscosityRatio,
        TurbulentKineticEnergy,
        TurbulentIntensity,
        TurbulentLengthScale,
        ModifiedTurbulentViscosityRatio,
        ModifiedTurbulentViscosity,
        SpecificDissipationRateAndTurbulentKineticEnergy,
        TurbulentViscosityRatioAndTurbulentKineticEnergy,
        TurbulentLengthScaleAndTurbulentKineticEnergy,
        TurbulentIntensityAndSpecificDissipationRate,
        TurbulentIntensityAndTurbulentViscosityRatio,
        TurbulentIntensityAndTurbulentLengthScale,
        SpecificDissipationRateAndTurbulentViscosityRatio,
        SpecificDissipationRateAndTurbulentLengthScale,
        TurbulentViscosityRatioAndTurbulentLengthScale,
    ],
    pd.Field(discriminator="type_name"),
]


# pylint: disable=too-many-arguments, too-many-return-statements, too-many-branches, invalid-name
# using class naming convetion here
def TurbulenceQuantities(
    viscosity_ratio=None,
    modified_viscosity_ratio=None,
    modified_viscosity=None,
    specific_dissipation_rate=None,
    turbulent_kinetic_energy=None,
    turbulent_length_scale=None,
    turbulent_intensity=None,
) -> TurbulenceQuantitiesType:
    r"""

    :func:`TurbulenceQuantities` function specifies turbulence conditions
    for the :class:`~flow360.Inflow` or :class:`~flow360.Freestream`
    at boundaries. The turbulence properties that can be
    specified are listed below. All values are dimensional.
    For valid specifications as well as the default values,
    please see the `Notes` section below.

    Parameters
    ----------
    viscosity_ratio : >= 0
        The ratio between the turbulent viscosity and freestream laminar
        viscosity. Applicable to both :class:`~flow360.KOmegaSST` and
        :class:`~flow360.SpalartAllmaras`. Its value will be converted to
        :py:attr:`modifiedTurbulentViscosityRatio` when using
        SpalartAllmaras model.
    modified_viscosity_ratio : >= 0
        The ratio between the modified turbulent viscosity (in SA model) and
        freestream laminar viscosity.
        Applicable to :class:`~flow360.SpalartAllmaras`.
    modified_viscosity : >=0
        The modified turbulent viscosity, aka nuHat.
        Applicable to :class:`~flow360.SpalartAllmaras`.
    specific_dissipation_rate : >= 0
        The turbulent specific dissipation rate. Applicable to :class:`~flow360.KOmegaSST`.
    turbulent_kinetic_energy : >=0
        The turbulent kinetic energy. Applicable to :class:`~flow360.KOmegaSST`.
    turbulent_length_scale : > 0
        The turbulent length scale is an estimation of the size of
        the eddies that are modeled/not resolved.
        Applicable to :class:`~flow360.KOmegaSST`.
    turbulent_intensity : >= 0
        The turbulent intensity is related to the turbulent kinetic energy by
        :math:`k = 1.5(U_{ref} * I)^2` where :math:`k` is the dimensional
        turbulent kinetic energy, :math:`U_{ref}` is the reference velocity
        and :math:`I` is the turbulent intensity. The value represents the
        actual magnitude of intensity instead of percentage. Applicable to
        :class:`~flow360.KOmegaSST`.

    Returns
    -------
        A matching tubulence specification object.

    Raises
    -------
    ValueError
        If the TurbulenceQuantities inputs do not represent a valid specification.

    Notes
    -----

    The valid combinations of multiple turbulence quantities is summarized as follows,

    default
        The default turbulence depends on the turbulence model.
        For SA model *without transition model* this is equivalent to set
        :code:`modified_viscosity_ratio = 3.0` (or effectively :code:`viscosity_ratio = 0.210438`).
        For SA model *with transition model*, :code:`modified_viscosity_ratio = 0.1`
        (or effectively :code:`viscosity_ratio = 2.794e-7`). For SST model the default turbulence is
        :code:`viscosity_ratio = 0.01` with default :code:`specific_dissipation_rate` = :math:`MachRef/L_{box}`
        where :math:`L_{box} \triangleq exp\left(\displaystyle\sum_{i=1}^{3}log(x_{i,max}-x_{i,min}\right)`.
        :math:`x_{i,max},x_{i,min}` is the bounding box dimension for wall boundaries.
    :code:`viscosity_ratio` alone
        This applies to both SST and SA model. For SST model this is effectively
        an override of the above default :code:`viscosity_ratio` value while keeping
        the default specificDissipationRate. For SA model the :code:`viscosity_ratio`
        will be converted to the :code:`modified_viscosity_ratio`.
    :code:`turbulent_kinetic_energy` or :code:`turbulent_intensity` alone
        For SST model only. :code:`specific_dissipation_rate` will be set to the default value.
    :code:`turbulent_length_scale` alone
        For SST model only. :code:`specific_dissipation_rate` will be set to the default value.
    :code:`modified_viscosity`
        For SA model only.
    :code:`modified_viscosity_ratio`
        For SA model only.
    :code:`turbulent_kinetic_energy` or :code:`turbulent_intensity` with :code:`specific_dissipation_rate`
        For SST model only.
    :code:`turbulent_kinetic_energy` or :code:`turbulent_intensity` with :code:`viscosity_ratio`
        For SST model only.
    :code:`turbulent_kinetic_energy` or :code:`turbulent_intensity` with :code:`turbulent_length_scale`
        For SST model only.
    :code:`specific_dissipation_rate` with :code:`viscosity_ratio`
        For SST model only.
    :code:`specific_dissipation_rate` with :code:`turbulent_length_scale`
        For SST model only.
    :code:`viscosity_ratio` with :code:`turbulent_length_scale`
        For SST model only.

    Example
    -------
    Apply modified turbulent viscosity ratio for SA model.

    >>> fl.TurbulenceQuantities(modified_viscosity_ratio=10)

    Apply turbulent kinetic energy and specific dissipation rate for SST model.

    >>> fl.TurbulenceQuantities(
        turbulent_kinetic_energy=0.2 * fl.u.m**2 / fl.u.s**2,
        specific_dissipation_rate=100 / fl.u.s)

    Apply specific dissipation rate and turbulent viscosity ratio for SST model.

    >>> fl.TurbulenceQuantities(specific_dissipation_rate=150 / fl.u.s, viscosity_ratio=1000)

    """
    non_none_arg_count = sum(arg is not None for arg in locals().values())
    if non_none_arg_count == 0:
        return None

    if non_none_arg_count > 2:
        raise ValueError(
            "Provided number of inputs exceeds the limit for any of the listed specifications. "
            + "Please recheck TurbulenceQuantities inputs and make sure they represent a valid specification."
        )

    if viscosity_ratio is not None:
        if non_none_arg_count == 1:
            return TurbulentViscosityRatio(turbulent_viscosity_ratio=viscosity_ratio)
        if turbulent_kinetic_energy is not None:
            return TurbulentViscosityRatioAndTurbulentKineticEnergy(
                turbulent_viscosity_ratio=viscosity_ratio,
                turbulent_kinetic_energy=turbulent_kinetic_energy,
            )
        if turbulent_intensity is not None:
            return TurbulentIntensityAndTurbulentViscosityRatio(
                turbulent_viscosity_ratio=viscosity_ratio,
                turbulent_intensity=turbulent_intensity,
            )
        if specific_dissipation_rate is not None:
            return SpecificDissipationRateAndTurbulentViscosityRatio(
                turbulent_viscosity_ratio=viscosity_ratio,
                specific_dissipation_rate=specific_dissipation_rate,
            )
        if turbulent_length_scale is not None:
            return TurbulentViscosityRatioAndTurbulentLengthScale(
                turbulent_viscosity_ratio=viscosity_ratio,
                turbulent_length_scale=turbulent_length_scale,
            )

    if modified_viscosity_ratio is not None and non_none_arg_count == 1:
        return ModifiedTurbulentViscosityRatio(
            modified_turbulent_viscosity_ratio=modified_viscosity_ratio
        )

    if modified_viscosity is not None and non_none_arg_count == 1:
        return ModifiedTurbulentViscosity(modified_turbulent_viscosity=modified_viscosity)

    if turbulent_intensity is not None:
        if non_none_arg_count == 1:
            return TurbulentIntensity(turbulent_intensity=turbulent_intensity)
        if specific_dissipation_rate is not None:
            return TurbulentIntensityAndSpecificDissipationRate(
                turbulent_intensity=turbulent_intensity,
                specific_dissipation_rate=specific_dissipation_rate,
            )
        if turbulent_length_scale is not None:
            return TurbulentIntensityAndTurbulentLengthScale(
                turbulent_intensity=turbulent_intensity,
                turbulent_length_scale=turbulent_length_scale,
            )

    if turbulent_kinetic_energy is not None:
        if non_none_arg_count == 1:
            return TurbulentKineticEnergy(turbulent_kinetic_energy=turbulent_kinetic_energy)
        if specific_dissipation_rate is not None:
            return SpecificDissipationRateAndTurbulentKineticEnergy(
                turbulent_kinetic_energy=turbulent_kinetic_energy,
                specific_dissipation_rate=specific_dissipation_rate,
            )
        if turbulent_length_scale is not None:
            return TurbulentLengthScaleAndTurbulentKineticEnergy(
                turbulent_kinetic_energy=turbulent_kinetic_energy,
                turbulent_length_scale=turbulent_length_scale,
            )

    if turbulent_length_scale is not None and non_none_arg_count == 1:
        return TurbulentLengthScale(turbulent_length_scale=turbulent_length_scale)

    if specific_dissipation_rate is not None:
        if turbulent_length_scale is not None:
            return SpecificDissipationRateAndTurbulentLengthScale(
                specific_dissipation_rate=specific_dissipation_rate,
                turbulent_length_scale=turbulent_length_scale,
            )

    raise ValueError(
        "Provided inputs do not create a valid specification. "
        + "Please recheck TurbulenceQuantities inputs and make sure they represent a valid specification."
    )
