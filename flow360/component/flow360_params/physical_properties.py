import math
from abc import ABC, abstractclassmethod

import numpy as np

from .unit_system import DensityType, PressureType, TemperatureType, ViscosityType, u


class _GasModel(ABC):
    # specific gas constant
    R = None

    @classmethod
    def pressure_from_density_temperature(
        cls, density: DensityType, temperature: TemperatureType
    ) -> PressureType:
        pressure = cls.R * density * temperature
        return pressure

    @classmethod
    def density_from_pressure_temperature(
        cls, pressure: PressureType, temperature: TemperatureType
    ) -> DensityType:
        density = pressure / cls.R / temperature
        return density

    @abstractclassmethod
    def viscosity_from_temperature(self, temperature):
        pass


class _AirModel(_GasModel):
    R = 287.0529 * u.J / u.kg / u.K

    @classmethod
    def viscosity_from_temperature(self, temperature: TemperatureType) -> ViscosityType:
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
    def speed_of_sound(cls, temperature: TemperatureType) -> ViscosityType:
        return np.sqrt(1.4 * cls.R * temperature.to("K")).to("m/s")
