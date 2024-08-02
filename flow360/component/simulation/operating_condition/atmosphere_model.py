""" U.S. STANDARD ATMOSPHERE 1976
# https://www.ngdc.noaa.gov/stp/space-weather/online-publications/miscellaneous/us-standard-atmosphere-1976/us-standard-atmosphere_st76-1562_noaa.pdf
 Source: design 360
 """

from math import exp


class StandardAtmosphereModel:
    """Standard atmosphere model for the Earth."""

    g0_prim = 9.80665
    r0 = 6356766  # earth radius in m
    R_star = 8.31432e3  # universal gas constant N*m/kmol/K
    M0 = 28.9644  # kg/kmol
    R = 287.0529

    H = [0, 11, 20, 32, 47, 51, 71, 84.8520]
    L_M = [-6.5, 0, 1, 2.8, 0, -2.8, -2.0]
    T_M = [288.15, 216.65, 216.65, 228.65, 270.65, 270.65, 214.65, 186.946]
    P_b = [
        101325,
        22632.06397346292,
        5474.888669677776,
        868.0186847552283,
        110.90630555496591,
        66.93887311868727,
        3.956420428040724,
        0.3733835899762152,
    ]

    def __init__(self, altitude_in_meters, temperature_offset_in_kelvin=0):
        if altitude_in_meters > 86000 or altitude_in_meters < -5000:
            raise ValueError(
                "The altitude should be between -5000 and 86000 meters. The input value is "
                + str(altitude_in_meters)
                + " meters."
            )
        self.altitude_in_meters = altitude_in_meters
        self.temperature_offset_in_kelvin = temperature_offset_in_kelvin

        self._temperature = self._calculate_temperature(
            self.altitude_in_meters, self.temperature_offset_in_kelvin
        )
        self._pressure = self._calculate_pressure(self.altitude_in_meters)
        self._density = self._calculate_density(self.temperature, self.pressure)

    @classmethod
    def b_index(cls, h):
        """Return the index of the layer in which the altitude is located."""
        for i in range(len(cls.H) - 2):
            if h < cls.H[i + 1]:
                return i
        return len(cls.H) - 2

    @classmethod
    def _calculate_geopotential_altitude(cls, altitude_in_meters):
        geopotential_altitude = cls.r0 * altitude_in_meters / (cls.r0 + altitude_in_meters)
        return geopotential_altitude

    @classmethod
    def _calculate_temperature(cls, altitude_in_meters, temperature_offset_in_kelvin):
        h = cls._calculate_geopotential_altitude(altitude_in_meters) / 1000
        b = cls.b_index(h)
        temperature = cls.T_M[b] + cls.L_M[b] * (h - cls.H[b])
        return temperature + temperature_offset_in_kelvin

    @classmethod
    def _calculate_pressure(cls, altitude_in_meters):
        h = cls._calculate_geopotential_altitude(altitude_in_meters) / 1000
        b = cls.b_index(h)
        if cls.L_M[b] == 0:
            factor = exp(-cls.g0_prim * cls.M0 / cls.R_star * 1000 * (h - cls.H[b]) / cls.T_M[b])
        else:
            factor = pow(
                (cls.T_M[b] / (cls.T_M[b] + cls.L_M[b] * (h - cls.H[b]))),
                (cls.g0_prim * cls.M0 / cls.R_star / cls.L_M[b] * 1000),
            )
        return cls.P_b[b] * factor

    @classmethod
    def _calculate_density(cls, temperature, pressure):
        return pressure / temperature * cls.M0 / cls.R_star

    @property
    def pressure(self) -> float:
        """Return pressure in Pa."""
        return self._pressure

    @property
    def density(self) -> float:
        """Return density in kg/m^3."""
        return self._density

    @property
    def temperature(self) -> float:
        """Return temperature in K."""
        return self._temperature
