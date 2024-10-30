"""
Module Summary
---------------
This module provides gas models, specifically the abstract base class `_GasModel` and its concrete subclass `_AirModel`.

"""

from abc import ABC, abstractmethod

import numpy as np

from flow360.component.v1.unit_system import (
    DensityType,
    PressureType,
    TemperatureType,
    VelocityType,
    ViscosityType,
    u,
)


class _GasModel(ABC):
    """
    Class representing air properties based on the gas model.

    Parameters
    ----------
    R : Quantity
        Specific gas constant for air.

    """

    R = None

    @classmethod
    def pressure_from_density_temperature(
        cls, density: DensityType, temperature: TemperatureType
    ) -> PressureType:
        """
        Calculate pressure from density and temperature.

        Parameters:
        -----------
        density : DensityType
            Density of the gas.
        temperature : TemperatureType
            Temperature of the gas.

        Returns:
        --------
        PressureType
            Pressure calculated from the given density and temperature.
        """

        pressure = cls.R * density * temperature
        return pressure

    @classmethod
    def density_from_pressure_temperature(
        cls, pressure: PressureType, temperature: TemperatureType
    ) -> DensityType:
        """
        Calculate density from pressure and temperature.

        Parameters:
        -----------
        pressure : PressureType
            Pressure of the gas.
        temperature : TemperatureType
            Temperature of the gas.

        Returns:
        --------
        DensityType
            Density calculated from the given pressure and temperature.
        """

        density = pressure / cls.R / temperature
        return density

    @classmethod
    @abstractmethod
    def viscosity_from_temperature(cls, temperature):
        """
        Abstract method for calculating viscosity from temperature.
        """


# pylint: disable=no-member
class _AirModel(_GasModel):
    """
    Class representing air properties based on the gas model.

    Parameters
    ----------
    R : Quantity
        Specific gas constant for air.

    """

    R = 287.0529 * u.J / u.kg / u.K

    @classmethod
    def viscosity_from_temperature(cls, temperature: TemperatureType) -> ViscosityType:
        """viscosity using Sutherland’s Law


        Parameters
        ----------
        temperature : TemperatureType
            temperature in with units (e.g., K, C)

        Returns
        -------
        ViscosityType
            returns viscosity with units
        """
        viscosity = (
            1.458e-6
            * pow(temperature.to("K").v.item(), 1.5)
            / (temperature.to("K").v.item() + 110.4)
        )
        return viscosity * u.Pa * u.s

    @classmethod
    def speed_of_sound(cls, temperature: TemperatureType) -> VelocityType:
        """Calculates the speed of sound in the air based on the temperature. Returns dimensioned value"""
        return np.sqrt(1.4 * cls.R * temperature.to("K")).to("m/s")
