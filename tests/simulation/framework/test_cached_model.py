import json
import os
import tempfile
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
    constructor: Optional[str] = None
    altitude: Optional[LengthType.Positive] = None
    temperature_offset: Optional[TemperatureType] = None
    some_value: Optional[float] = None
    temperature: Optional[TemperatureType.Positive] = None
    density: Optional[DensityType.Positive] = None


class TempThermalState(CachedModelBase):
    temperature: TemperatureType.Positive = pd.Field(288.15 * u.K, frozen=True)
    density: DensityType.Positive = pd.Field(1.225 * u.kg / u.m**3, frozen=True)
    some_value: float = 0.1
    _cached: TempThermalStateCache = TempThermalStateCache()

    @CachedModelBase.model_constructor
    def from_standard_atmosphere(
        cls, altitude: LengthType.Positive = 0 * u.m, temperature_offset: TemperatureType = 0 * u.K
    ):
        density = 1.225 * u.kg / u.m**3
        temperature = 288.15 * u.K

        state = cls(
            density=density,
            temperature=temperature,
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

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp_file:
        temp_file_name = temp_file.name

    try:
        operating_condition.to_file(temp_file_name)
        with open(temp_file_name) as fp:
            model_dict = json.load(fp)
            assert model_dict["thermal_state"]["_cached"]["some_value"] == 0.1
            assert (
                model_dict["thermal_state"]["_cached"]["constructor"] == "from_standard_atmosphere"
            )
            loaded_model = TempOperatingCondition(**model_dict)
            assert loaded_model == operating_condition
            assert loaded_model.thermal_state._cached.altitude == 100 * u.m
            assert loaded_model.thermal_state._cached.temperature_offset == 0 * u.K
    finally:
        os.remove(temp_file_name)
