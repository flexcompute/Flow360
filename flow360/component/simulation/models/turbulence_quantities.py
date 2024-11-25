"""
Turbulence quantities parameters
"""

# pylint: disable=unused-import
from abc import ABCMeta
from typing import Annotated, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.unit_system import (
    FrequencyType,
    LengthType,
    SpecificEnergyType,
    ViscosityType,
)


class TurbulentKineticEnergy(Flow360BaseModel):
    """
    turbulentKineticEnergy : non-dimensional [`C_inf^2`]
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
    specificDissipationRate : non-dimensional [`C_inf/L_gridUnit`]
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
    turbulentLengthScale : non-dimensional [`L_gridUnit`]
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
    modifiedTurbulentViscosity : non-dimensional [`C_inf*L_gridUnit`]
        The modified turbulent eddy viscosity (SA). Applicable only when using SA model.
    """

    type_name: Literal["ModifiedTurbulentViscosity"] = pd.Field(
        "ModifiedTurbulentViscosity", frozen=True
    )
    # pylint: disable=no-member
    modified_turbulent_viscosity: Optional[ViscosityType.Positive] = pd.Field()


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
    """

    :func:`TurbulenceQuantities` function specifies turbulence conditions
    for the :class:`~flow360.Inflow` or :class:`~flow360.Freestream`
    at boundaries. The turbulence properties that can be
    specified are listed below. All values are dimensional.
    For valid specifications as well as the default values,
    please refer to :ref:`knowledge base<knowledgeBaseTurbulenceQuantities>`.

    Parameters
    ----------
    viscosity_ratio : >= 0
        The ratio between the turbulent viscosity and freestream laminar
        viscosity. Applicable to both :class:`~flow360.KOmegaSST` and
        `~flow360.SpalartAllmaras`. Its value will be converted to
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

    Example
    -------

    >>> fl.TurbulenceQuantities(modified_viscosity_ratio=10)

    """
    non_none_arg_count = sum(arg is not None for arg in locals().values())
    if non_none_arg_count == 0:
        return None

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
        "Please recheck TurbulenceQuantities inputs and make sure they represents a valid specification."
    )
