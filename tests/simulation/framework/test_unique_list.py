import re
from typing import Literal

import pydantic as pd
import pytest

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.unique_list import (
    UniqueItemList,
    UniqueStringList,
)
from flow360.component.simulation.outputs.output_fields import CommonFieldNames
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
    output_fields: UniqueItemList[CommonFieldNames] = pd.Field()


class TempPeriodic(Flow360BaseModel):
    surface_pairs: UniqueItemList[SurfacePair]


def test_unique_list():
    my_iso_1 = TempIsosurface(name="iso_1", field_magnitude=1.01)
    my_iso_1_dup = TempIsosurface(name="iso_1", field_magnitude=1.02)
    my_slice = TempSlice(name="slice_1")
    # 1: Test duplicate isosurfaces
    output = TempIsosurfaceOutput(
        isosurfaces=[my_iso_1, my_iso_1_dup], output_fields=["wallDistance"]
    )

    assert len(output.isosurfaces.items) == 1

    # 2: Test duplicate output_fields
    output = TempIsosurfaceOutput(
        isosurfaces=[my_iso_1], output_fields=["wallDistance", "wallDistance"]
    )
    assert output.output_fields.items == ["wallDistance"]

    # 3: Test duplicate output_fields by aliased name
    output = TempIsosurfaceOutput(
        isosurfaces=[my_iso_1],
        output_fields=[
            "wallDistance",
            "Cp",
            "wallDistance",
            "qcriterion",
        ],
    )

    assert output.output_fields.items == ["wallDistance", "Cp", "qcriterion"]

    # 4: Test unvalid types:
    with pytest.raises(
        ValueError,
        match=re.escape("Input should be a valid dictionary or instance of TempIsosurface"),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1, my_slice], output_fields=["wallDistance"])

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Input should be "
            "'Cp', "
            "'Cpt', "
            "'gradW', "
            "'kOmega', "
            "'Mach', "
            "'mut', "
            "'mutRatio', "
            "'nuHat', "
            "'primitiveVars', "
            "'qcriterion', "
            "'residualNavierStokes', "
            "'residualTransition', "
            "'residualTurbulence', "
            "'s', "
            "'solutionNavierStokes', "
            "'solutionTransition', "
            "'solutionTurbulence', "
            "'T', "
            "'velocity', "
            "'velocity_x', "
            "'velocity_y', "
            "'velocity_z', "
            "'velocity_magnitude', "
            "'pressure', "
            "'vorticity', "
            "'vorticityMagnitude', "
            "'vorticity_x', "
            "'vorticity_y', "
            "'vorticity_z', "
            "'wallDistance', "
            "'numericalDissipationFactor', "
            "'residualHeatSolver', "
            "'VelocityRelative', "
            "'lowMachPreconditionerSensor', "
            "'velocity_m_per_s', "
            "'velocity_x_m_per_s', "
            "'velocity_y_m_per_s', "
            "'velocity_z_m_per_s', "
            "'velocity_magnitude_m_per_s', "
            "'pressure_pa' "
            "or 'helicity'"
        ),
    ):
        TempIsosurfaceOutput(isosurfaces=[my_iso_1], output_fields=["wallDistance", 1234])

    # 5: Test append triggering validation
    temp_iso = TempIsosurfaceOutput(isosurfaces=[my_iso_1], output_fields=["Cp", "wallDistance"])

    assert len(temp_iso.isosurfaces.items) == 1

    temp_iso.isosurfaces.append(my_iso_1)
    assert len(temp_iso.isosurfaces.items) == 1


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

    surface = TempPeriodic(
        surface_pairs=[
            [surface1, surface2],
            [surface2, surface1],
        ]
    )

    assert len(surface.surface_pairs.items) == 1
