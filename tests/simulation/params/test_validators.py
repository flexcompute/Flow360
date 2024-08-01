import unittest

import pytest

import flow360.component.simulation.units as u
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import UniformRefinement
from flow360.component.simulation.models.material import SolidMaterial
from flow360.component.simulation.models.surface_models import (
    HeatFlux,
    Inflow,
    MassFlowRate,
    SlipWall,
    TotalPressure,
    Wall,
)
from flow360.component.simulation.models.turbulence_quantities import (
    TurbulenceQuantities,
)
from flow360.component.simulation.models.volume_models import (
    AngularVelocity,
    BETDisk,
    Fluid,
    PorousMedium,
    Rotation,
    Solid,
)
from flow360.component.simulation.operating_condition import (
    AerospaceCondition,
    ThermalState,
)
from flow360.component.simulation.primitives import (
    Box,
    Cylinder,
    GenericVolume,
    ReferenceGeometry,
    Surface,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Unsteady
from flow360.component.simulation.unit_system import CGS_unit_system
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from tests.simulation.translator.utils.xv15BETDisk_param_generator import (
    create_steady_hover_param,
)
from tests.utils import to_file_from_file_test

assertions = unittest.TestCase("__init__")


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.mark.usefixtures("array_equality_override")
def test_bet_disk_initial_blade_direction(create_steady_hover_param):
    sim_param = create_steady_hover_param
    SimulationParams.model_validate(sim_param)

    sim_param_2 = sim_param.model_copy(deep=True)
    for model in sim_param_2.models:
        if isinstance(model, BETDisk):
            model.blade_line_chord = 0.1 * u.inch
            with pytest.raises(
                ValueError,
                match="the initial_blade_direction is required to specify since its blade_line_chord is non-zero",
            ):
                SimulationParams.model_validate(sim_param_2)
