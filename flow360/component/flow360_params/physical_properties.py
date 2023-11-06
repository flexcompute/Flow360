from .unit_system import ViscosityType
from abc import ABC, abstractclassmethod
import math

class _GasModel(ABC):
    R = None

    @classmethod
    def pressure_from_density_temperature(cls, density, temperature):
        pressure = cls.R * density * temperature
        return pressure

    @classmethod
    def density_from_pressure_temperature(cls, pressure, temperature):
        density = pressure / cls.R / temperature
        return density
    
    @abstractclassmethod
    def viscosity_from_temperature(self, temperature):
        pass




class _AirModel(_GasModel):
    R = 287.0529

    @classmethod
    def viscosity_from_temperature(self, temperature) -> ViscosityType:
        """viscosity using Sutherland’s Law


        Parameters
        ----------
        temperature : float
            temperature in Kelvins

        Returns
        -------
        _type_
            _description_
        """
        viscosity = 1.458E-6 * pow(temperature, 1.5) / (temperature + 110.4)
        return viscosity
    
    @classmethod
    def speed_of_sound(cls, temperature) -> float:
        return math.sqrt(1.4 * cls.R * temperature)
