"""
Turbulence quantities parameters
"""

# pylint: disable=unused-import
from abc import ABCMeta
from typing import Optional, Union

import pydantic as pd

from ..types import NonNegativeFloat, PositiveFloat
from .params_base import Flow360BaseModel


class TurbulentKineticEnergy(Flow360BaseModel):
    """
    turbulentKineticEnergy : non-dimensional [`C_inf^2`]
        Turbulent kinetic energy. Applicable only when using SST model.
    """

    turbulent_kinetic_energy: Optional[NonNegativeFloat] = pd.Field(alias="turbulentKineticEnergy")


class TurbulentIntensity(Flow360BaseModel):
    """
    turbulentIntensity : non-dimensional [`-`]
        Turbulent intensity. Applicable only when using SST model.
        This is related to turbulent kinetic energy as:
        `turbulentKineticEnergy = 1.5*pow(U_ref * turbulentIntensity, 2)`.
        Note the use of the freestream velocity U_ref instead of C_inf.
    """

    turbulent_intensity: Optional[NonNegativeFloat] = pd.Field(alias="turbulentIntensity")


class _SpecificDissipationRate(Flow360BaseModel, metaclass=ABCMeta):
    """
    specificDissipationRate : non-dimensional [`C_inf/L_gridUnit`]
        Turbulent specific dissipation rate. Applicable only when using SST model.
    """

    specific_dissipation_rate: Optional[NonNegativeFloat] = pd.Field(
        alias="specificDissipationRate"
    )


class TurbulentViscosityRatio(Flow360BaseModel):
    """
    turbulentViscosityRatio : non-dimensional [`-`]
        The ratio of turbulent eddy viscosity over the freestream viscosity. Applicable for both SA and SST model.
    """

    turbulent_viscosity_ratio: Optional[NonNegativeFloat] = pd.Field(
        alias="turbulentViscosityRatio"
    )


class TurbulentLengthScale(Flow360BaseModel, metaclass=ABCMeta):
    """
    turbulentLengthScale : non-dimensional [`L_gridUnit`]
        The turbulent length scale is an estimation of the size of the eddies that are modeled/not resolved.
        Applicable only when using SST model. This is related to the turbulent kinetic energy and turbulent
        specific dissipation rate as: `L_T = sqrt(k)/(pow(beta_0^*, 0.25)*w)` where `L_T` is turbulent length scale,
        `k` is turbulent kinetic energy, `beta_0^*` is 0.09 and `w` is turbulent specific dissipation rate.
        Applicable only when using SST model.
    """

    turbulent_length_scale: Optional[PositiveFloat] = pd.Field(alias="turbulentLengthScale")


class ModifiedTurbulentViscosityRatio(Flow360BaseModel):
    """
    modifiedTurbulentViscosityRatio : non-dimensional [`-`]
        The ratio of modified turbulent eddy viscosity (SA) over the freestream viscosity.
        Applicable only when using SA model.
    """

    modified_turbulent_viscosity_ratio: Optional[PositiveFloat] = pd.Field(
        alias="modifiedTurbulentViscosityRatio"
    )


class ModifiedTurbulentViscosity(Flow360BaseModel):
    """
    modifiedTurbulentViscosity : non-dimensional [`C_inf*L_gridUnit`]
        The modified turbulent eddy viscosity (SA). Applicable only when using SA model.
    """

    modified_turbulent_viscosity: Optional[PositiveFloat] = pd.Field(
        alias="modifiedTurbulentViscosity"
    )


# pylint: disable=missing-class-docstring
class SpecificDissipationRateAndTurbulentKineticEnergy(
    _SpecificDissipationRate, TurbulentKineticEnergy
):
    pass


class TurbulentViscosityRatioAndTurbulentKineticEnergy(
    TurbulentViscosityRatio, TurbulentKineticEnergy
):
    pass


class TurbulentLengthScaleAndTurbulentKineticEnergy(TurbulentLengthScale, TurbulentKineticEnergy):
    pass


class TurbulentIntensityAndSpecificDissipationRate(TurbulentIntensity, _SpecificDissipationRate):
    pass


class TurbulentIntensityAndTurbulentViscosityRatio(TurbulentIntensity, TurbulentViscosityRatio):
    pass


class TurbulentIntensityAndTurbulentLengthScale(TurbulentIntensity, TurbulentLengthScale):
    pass


class SpecificDissipationRateAndTurbulentViscosityRatio(
    _SpecificDissipationRate, TurbulentViscosityRatio
):
    pass


class SpecificDissipationRateAndTurbulentLengthScale(
    _SpecificDissipationRate, TurbulentLengthScale
):
    pass


class TurbulentViscosityRatioAndTurbulentLengthScale(TurbulentViscosityRatio, TurbulentLengthScale):
    pass


# pylint: enable=missing-class-docstring

TurbulenceQuantitiesType = Union[
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
    """Return a matching tubulence specification object"""
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
