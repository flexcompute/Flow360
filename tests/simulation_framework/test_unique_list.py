import re

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_fields import CommonFields
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.unique_list import (
    UniqueAliasedItemList,
    UniqueItemList,
)


class _OutputItemBase(Flow360BaseModel):
    name: str = pd.Field()

    def __hash__(self):
        return hash(self.name + "-" + self.__class__.__name__)

    def __eq__(self, other):
        if isinstance(other, _OutputItemBase):
            return (self.name + "-" + self.__class__.__name__) == (
                other.name + "-" + other.__class__.__name__
            )
        return False

    def __str__(self):
        return f"{self.__class__.__name__} {self.name}"


class TempIsosurface(_OutputItemBase):
    field_magnitude: float = pd.Field()


class TempSlice(_OutputItemBase):
    field_magnitude: float = pd.Field()


class TempIsosurfaceOutput(Flow360BaseModel):
    isosurfaces: UniqueItemList[TempIsosurface] = pd.Field()
    output_fields: UniqueAliasedItemList[CommonFields] = pd.Field()


def test_unique_list():
    my_iso_1 = TempIsosurface(name="iso_1", field_magnitude=1.01)
    my_iso_1_dup = TempIsosurface(name="iso_1", field_magnitude=1.02)
    my_fake_slice = TempSlice(name="iso_1", field_magnitude=1.02)
    # 1: Test duplicate isosurfaces
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input item to this list must be unique but ['TempIsosurface iso_1'] appears multiple times."
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1, my_iso_1_dup], output_fields=["wallDistance"])
    # 2: Test duplicate output_fields
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input item to this list must be unique but ['wallDistance'] appears multiple times."
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1], output_fields=["wallDistance", "wallDistance"])
    # 3: Test duplicate output_fields by aliased name
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input item to this list must be unique but wallDistance and Wall distance are both present."
        ),
    ):
        TempIsosurfaceOutput(
            isosurfaces=[my_iso_1], output_fields=["wallDistance", "Wall distance"]
        )
    # 4: Test unvalid types:
    with pytest.raises(
        ValueError,
        match=re.escape("Input should be a valid dictionary or instance of TempIsosurface"),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1, my_fake_slice], output_fields=["wallDistance"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input should be 'Cp', 'gradW', 'kOmega', 'Mach', 'mut', 'mutRatio', 'nuHat', 'primitiveVars', 'qcriterion', 'residualNavierStokes', 'residualTransition', 'residualTurbulence', 's', 'solutionNavierStokes', 'solutionTransition', 'solutionTurbulence', 'T', 'vorticity', 'wallDistance', 'numericalDissipationFactor', 'residualHeatSolver', 'VelocityRelative', 'lowMachPreconditionerSensor', 'Coefficient of pressure', 'Gradient of primitive solution', 'k and omega', 'Mach number', 'Turbulent viscosity', 'Turbulent viscosity and freestream dynamic viscosity ratio', 'Spalart-Almaras variable', 'rho, u, v, w, p (density, 3 velocities and pressure)', 'Q criterion', 'N-S residual', 'Transition residual', 'Turbulence residual', 'Entropy', 'N-S solution', 'Transition solution', 'Turbulence solution', 'Temperature', 'Vorticity', 'Wall distance', 'NumericalDissipationFactor sensor', 'Heat equation residual', 'Velocity with respect to non-inertial frame' or 'Low-Mach preconditioner factor'"
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1], output_fields=["wallDistance", 1234])
