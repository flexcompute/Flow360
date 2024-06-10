from typing import Optional

import pydantic as pd
import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.cached_model_base import CachedModelBase
from flow360.component.simulation.unit_system import (
    DensityType,
    LengthType,
    TemperatureType,
)
from tests.utils import to_file_from_file_test


class TempThermalStateCache(Flow360BaseModel):
    altitude: Optional[LengthType.Positive] = None
    temperature_offset: Optional[TemperatureType] = None


class TempThermalState(CachedModelBase):
    temperature: TemperatureType.Positive = pd.Field(288.15 * u.K, frozen=True)
    density: DensityType.Positive = pd.Field(1.225 * u.kg / u.m**3, frozen=True)
    _cached: TempThermalStateCache = TempThermalStateCache()

    @classmethod
    def from_standard_atmosphere(
        cls, altitude: LengthType.Positive = 0 * u.m, temperature_offset: TemperatureType = 0 * u.K
    ):
        density = 1.225 * u.kg / u.m**3
        temperature = 288.15 * u.K

        state = cls(
            density=density,
            temperature=temperature,
        )
        state._cached = TempThermalStateCache(
            altitude=altitude, temperature_offset=temperature_offset
        )
        return state

    @property
    def altitude(self) -> Optional[LengthType.Positive]:
        return self._cached.altitude


class TempOperatingCondition(Flow360BaseModel):
    thermal_state: TempThermalState
    some_value: float


@pytest.mark.usefixtures("array_equality_override")
def test_cache_model():
    operating_condition = TempOperatingCondition(
        some_value=1230,
        thermal_state=TempThermalState.from_standard_atmosphere(altitude=100 * u.m),
    )
    to_file_from_file_test(operating_condition)
