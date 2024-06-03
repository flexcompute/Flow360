import re
from typing import Literal

import pydantic as pd
import pytest

from flow360.component.flow360_params.flow360_fields import (
    CommonFieldNames,
    CommonFieldNamesFull,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.unique_list import (
    UniqueAliasedStringList,
    UniqueItemList,
)
from flow360.component.simulation.primitives import Surface, SurfacePair


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
        return f"{self.__class__.__name__} with name: {self.name}"


class TempIsosurface(_OutputItemBase):
    field_magnitude: float = pd.Field()


class TempSlice(_OutputItemBase):
    pass


class TempIsosurfaceOutput(Flow360BaseModel):
    isosurfaces: UniqueItemList[TempIsosurface] = pd.Field()
    output_fields: UniqueAliasedStringList[Literal[CommonFieldNames, CommonFieldNamesFull]] = (
        pd.Field()
    )


class TempPeriodic(Flow360BaseModel):
    surface_pairs: UniqueItemList[SurfacePair]


def test_unique_list():
    my_iso_1 = TempIsosurface(name="iso_1", field_magnitude=1.01)
    my_iso_1_dup = TempIsosurface(name="iso_1", field_magnitude=1.02)
    my_slice = TempSlice(name="slice_1")
    # 1: Test duplicate isosurfaces
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input item to this list must be unique but ['TempIsosurface with name: iso_1'] appears multiple times."
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1, my_iso_1_dup], output_fields=["wallDistance"])

    # 2: Test duplicate output_fields
    output = TempIsosurfaceOutput(
        isosurfaces=[my_iso_1], output_fields=["wallDistance", "wallDistance"]
    )
    assert output.output_fields.items == ["wallDistance"]

    # 3: Test duplicate output_fields by aliased name
    output = TempIsosurfaceOutput(
        isosurfaces=[my_iso_1],
        output_fields=[
            "Wall distance",
            "wallDistance",
            "Wall distance",
            "Cp",
            "wallDistance",
            "Q criterion",
        ],
    )

    assert output.output_fields.items == ["Wall distance", "Cp", "Q criterion"]

    # 4: Test unvalid types:
    with pytest.raises(
        ValueError,
        match=re.escape("Input should be a valid dictionary or instance of TempIsosurface"),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1, my_slice], output_fields=["wallDistance"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input should be 'Cp', 'gradW', 'kOmega', 'Mach', 'mut', 'mutRatio', 'nuHat', 'primitiveVars', 'qcriterion', 'residualNavierStokes', 'residualTransition', 'residualTurbulence', 's', 'solutionNavierStokes', 'solutionTransition', 'solutionTurbulence', 'T', 'vorticity', 'wallDistance', 'numericalDissipationFactor', 'residualHeatSolver', 'VelocityRelative', 'lowMachPreconditionerSensor', 'Coefficient of pressure', 'Gradient of primitive solution', 'k and omega', 'Mach number', 'Turbulent viscosity', 'Turbulent viscosity and freestream dynamic viscosity ratio', 'Spalart-Almaras variable', 'rho, u, v, w, p (density, 3 velocities and pressure)', 'Q criterion', 'N-S residual', 'Transition residual', 'Turbulence residual', 'Entropy', 'N-S solution', 'Transition solution', 'Turbulence solution', 'Temperature', 'Vorticity', 'Wall distance', 'NumericalDissipationFactor sensor', 'Heat equation residual', 'Velocity with respect to non-inertial frame' or 'Low-Mach preconditioner factor'"
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1], output_fields=["wallDistance", 1234])


def test_unique_list_with_surface_pair():
    surface1 = Surface(name="MySurface1")
    surface2 = Surface(name="MySurface2")
    periodic = TempPeriodic(
        surface_pairs=[
            [surface1, surface2],
        ]
    )
    assert periodic

    with pytest.raises(
        ValueError,
        match=re.escape("A surface cannot be paired with itself."),
    ):
        SurfacePair(pair=[surface1, surface1])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input item to this list must be unique but ['MySurface1,MySurface2'] appears multiple times."
        ),
    ):
        TempPeriodic(
            surface_pairs=[
                [surface1, surface2],
                [surface2, surface1],
            ]
        )
