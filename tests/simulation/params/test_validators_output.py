import json
import re

import pytest

import flow360 as fl
import flow360.component.simulation.units as u
from flow360.component.simulation.entity_info import VolumeMeshEntityInfo
from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.solver_numerics import (
    KOmegaSST,
    NoneSolver,
    SpalartAllmaras,
)
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid, PorousMedium
from flow360.component.simulation.outputs.output_entities import Point
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    Isosurface,
    IsosurfaceOutput,
    MovingStatistic,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceOutput,
    SurfaceProbeOutput,
    TimeAverageSurfaceOutput,
    VolumeOutput,
)
from flow360.component.simulation.primitives import ImportedSurface, Surface
from flow360.component.simulation.services import (
    ValidationCalledBy,
    clear_context,
    validate_model,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import (
    SI_unit_system,
    imperial_unit_system,
)
from flow360.component.simulation.user_code.core.types import UserVariable
from flow360.component.simulation.user_code.functions import math
from flow360.component.simulation.user_code.variables import solution
from flow360.component.simulation.validation.validation_context import (
    CASE,
    ParamsValidationInfo,
    ValidationContext,
)


@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)


@pytest.fixture()
def reset_context():
    clear_context()


def test_aeroacoustic_observer_unit_validator():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "All observer locations should have the same unit. But now it has both `cm` and `mm`."
        ),
    ):
        AeroAcousticOutput(
            name="test",
            observers=[
                fl.Observer(position=[0.2, 0.02, 0.03] * u.cm, group_name="0"),
                fl.Observer(position=[0.0001, 0.02, 0.03] * u.mm, group_name="1"),
            ],
        )


def test_unsteadiness_to_use_aero_acoustics():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[1] AeroAcousticOutput:`AeroAcousticOutput` can only be activated with `Unsteady` simulation."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=NoneSolver())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="mut", iso_value=1)],
                        output_fields=["Cp"],
                    ),
                    AeroAcousticOutput(
                        name="test",
                        observers=[
                            fl.Observer(position=[0.2, 0.02, 0.03] * u.mm, group_name="0"),
                            fl.Observer(position=[0.0001, 0.02, 0.03] * u.mm, group_name="1"),
                        ],
                    ),
                ],
                time_stepping=fl.Steady(),
            )


def test_turbulence_enabled_output_fields():
    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput:, kOmega is not a valid output field when using turbulence model: None."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=NoneSolver())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="mut", iso_value=1)],
                        output_fields=["kOmega"],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] IsosurfaceOutput:, nuHat is not a valid iso field when using turbulence model: kOmegaSST."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=KOmegaSST())],
                outputs=[
                    IsosurfaceOutput(
                        name="iso",
                        entities=[Isosurface(name="tmp", field="nuHat", iso_value=1)],
                        output_fields=["Cp"],
                    )
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] VolumeOutput:, kOmega is not a valid output field when using turbulence model: SpalartAllmaras."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(turbulence_model_solver=SpalartAllmaras())],
                outputs=[VolumeOutput(output_fields=["kOmega"])],
            )


def test_surface_user_variables_in_output_fields():
    uv_surface1 = UserVariable(
        name="uv_surface1", value=math.dot(solution.velocity, solution.CfVec)
    )
    uv_surface2 = UserVariable(
        name="uv_surface2", value=solution.node_forces_per_unit_area[0] * solution.Cp * solution.Cf
    )

    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(
                    entities=Surface(name="fluid/body"), output_fields=[uv_surface1, uv_surface2]
                )
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface1` cannot be used in `VolumeOutput` "
            + "since it contains Surface solver variable(s): solution.CfVec."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[VolumeOutput(output_fields=[uv_surface1])],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Variable `uv_surface2` cannot be used in `ProbeOutput` "
            + "since it contains Surface solver variable(s): "
            + "solution.Cf, solution.node_forces_per_unit_area."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=[uv_surface2],
                    )
                ],
            )


def test_duplicate_surface_usage():
    my_var = UserVariable(name="my_var", value=solution.node_forces_per_unit_area[1])
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The same surface `fluid/body` is used in multiple `SurfaceOutput`s. "
            "Please specify all settings for the same surface in one output."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceOutput(entities=Surface(name="fluid/body"), output_fields=[my_var]),
                    SurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                    ),
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The same surface `fluid/body` is used in multiple `TimeAverageSurfaceOutput`s. "
            "Please specify all settings for the same surface in one output."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    TimeAverageSurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[my_var]
                    ),
                    TimeAverageSurfaceOutput(
                        entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                    ),
                ],
                time_stepping=Unsteady(steps=10, step_size=1e-3),
            )

    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceOutput(entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]),
                TimeAverageSurfaceOutput(
                    entities=Surface(name="fluid/body"), output_fields=[solution.CfVec]
                ),
            ],
            time_stepping=Unsteady(steps=10, step_size=1e-3),
        )


def test_moving_statitic_validator():
    wall_1 = Surface(name="wall_1", private_attribute_is_interface=False)
    asset_cache = AssetCache(
        project_length_unit="m",
        project_entity_info=VolumeMeshEntityInfo(boundaries=[wall_1]),
    )

    with SI_unit_system:
        monitored_variable = UserVariable(
            name="Helicity_MONITOR",
            value=math.dot(solution.velocity, solution.vorticity),
        )
        params = SimulationParams(
            time_stepping=Steady(max_steps=5000),
            models=[Fluid(), Wall(entities=wall_1)],
            outputs=[
                ProbeOutput(
                    name="point_legacy2",
                    output_fields=["Mach", monitored_variable],
                    probe_points=Point(name="Point1", location=(-0.026642, 0.56614, 0) * u.m),
                    moving_statistic=MovingStatistic(method="std", moving_window_size=15),
                )
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert len(errors) == 1
    assert (
        errors[0]["msg"] == "Value error, For steady simulation, "
        "the number of steps should be a multiple of 10."
    )

    with SI_unit_system:
        monitored_variable = UserVariable(
            name="Helicity_MONITOR",
            value=math.dot(solution.velocity, solution.vorticity),
        )
        params = SimulationParams(
            time_stepping=Steady(max_steps=5000),
            models=[Fluid(), Wall(entities=wall_1)],
            outputs=[
                ProbeOutput(
                    name="point_legacy2",
                    output_fields=["Mach", monitored_variable],
                    probe_points=Point(name="Point1", location=(-0.026642, 0.56614, 0) * u.m),
                    moving_statistic=MovingStatistic(method="std", moving_window_size=20),
                )
            ],
            private_attribute_asset_cache=asset_cache,
        )

    _, errors, _ = validate_model(
        validated_by=ValidationCalledBy.LOCAL,
        params_as_dict=params.model_dump(mode="json"),
        root_item_type="VolumeMesh",
        validation_level="Case",
    )
    assert errors is None


def test_duplicate_probe_names():

    # should have no error
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ProbeOutput(
                    name="probe_output_1",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["Cp"],
                ),
                ProbeOutput(
                    name="probe_output_2",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["velocity_x"],
                ),
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`outputs`[1] ProbeOutput: Output name probe_output has already been used for a "
            "`ProbeOutput` or `SurfaceProbeOutput`. Output names must be unique among all probe outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["Cp"],
                    ),
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_x"],
                    ),
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "`outputs`[1] SurfaceProbeOutput: Output name probe_output has already been used for a "
            "`ProbeOutput` or `SurfaceProbeOutput`. Output names must be unique among all probe outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["pressure"],
                    ),
                    SurfaceProbeOutput(
                        name="probe_output",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_y"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                ],
            )


def test_duplicate_probe_entity_names():

    # should have no error
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                ProbeOutput(
                    name="probe_output",
                    probe_points=[
                        Point(name="point_1", location=[1, 2, 3] * u.m),
                        Point(name="point_2", location=[1, 2, 3] * u.m),
                    ],
                    output_fields=["Cp"],
                ),
                ProbeOutput(
                    name="probe_output2",
                    probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                    output_fields=["velocity_x"],
                ),
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] ProbeOutput: Entity name point_1 has already been used in the "
            "same `ProbeOutput`. Entity names must be unique."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    ProbeOutput(
                        name="probe_output_1",
                        probe_points=[
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                        ],
                        output_fields=["Cp"],
                    ),
                    ProbeOutput(
                        name="probe_output_2",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_x"],
                    ),
                ],
            )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "In `outputs`[0] SurfaceProbeOutput: Entity name point_1 has already been used in the "
            "same `SurfaceProbeOutput`. Entity names must be unique."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceProbeOutput(
                        name="probe_output_1",
                        probe_points=[
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                            Point(name="point_1", location=[1, 2, 3] * u.m),
                        ],
                        output_fields=["pressure"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                    SurfaceProbeOutput(
                        name="probe_output_2",
                        probe_points=[Point(name="point_1", location=[1, 2, 3] * u.m)],
                        output_fields=["velocity_y"],
                        target_surfaces=[Surface(name="fluid/body")],
                    ),
                ],
            )


def test_surface_integral_entity_types():
    uv_surface1 = UserVariable(
        name="uv_surface1", value=math.dot(solution.velocity, solution.CfVec)
    )
    surface = Surface(name="fluid/body")
    imported_surface = ImportedSurface(name="imported", file_name="imported.stl")
    with imperial_unit_system:
        SimulationParams(
            outputs=[
                SurfaceIntegralOutput(entities=surface, output_fields=[uv_surface1]),
                SurfaceIntegralOutput(
                    entities=imported_surface,
                    output_fields=[uv_surface1],
                ),
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "Imported and simulation surfaces cannot be used together in the same SurfaceIntegralOutput."
            " Please assign them to separate outputs."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                outputs=[
                    SurfaceIntegralOutput(
                        entities=[surface, imported_surface], output_fields=[uv_surface1]
                    ),
                ],
            )


def test_force_output_with_wall_models():
    """Test ForceOutput with Wall models works correctly."""
    wall_1 = Wall(entities=Surface(name="fluid/wing1"))
    wall_2 = Wall(entities=Surface(name="fluid/wing2"))

    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1, wall_2],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1, wall_2],
                    output_fields=["CL", "CD", "CMx"],
                )
            ],
        )

    # Test with extended force coefficients (SkinFriction/Pressure)
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CLSkinFriction", "CLPressure", "CDSkinFriction"],
                )
            ],
        )


def test_force_output_with_surface_and_volume_models():
    """Test ForceOutput with volume models (BETDisk, ActuatorDisk, PorousMedium)."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))
    with imperial_unit_system:
        porous_zone = fl.Box.from_principal_axes(
            name="box",
            axes=[[0, 1, 0], [0, 0, 1]],
            center=[0, 0, 0] * fl.u.m,
            size=[0.2, 0.3, 2] * fl.u.m,
        )
        porous_medium = PorousMedium(
            entities=[porous_zone],
            darcy_coefficient=[1e6, 0, 0],
            forchheimer_coefficient=[1, 0, 0],
            volumetric_heat_source=0,
        )

    # Valid case: only basic force coefficients
    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1, porous_medium],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1, porous_medium],
                    output_fields=["CL", "CD", "CFx", "CFy", "CFz", "CMx", "CMy", "CMz"],
                )
            ],
        )

    with pytest.raises(
        ValueError,
        match=re.escape(
            "When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields."
        ),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1, porous_medium],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1, porous_medium],
                        output_fields=["CL", "CLSkinFriction"],
                    )
                ],
            )


def test_force_output_duplicate_models():
    """Test that ForceOutput rejects duplicate models."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    with pytest.raises(
        ValueError,
        match=re.escape("Duplicate models are not allowed in the same `ForceOutput`."),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_1, wall_1],
                        output_fields=["CL", "CD"],
                    )
                ],
            )


def test_force_output_nonexistent_model():
    """Test that ForceOutput rejects models not in SimulationParams' models list."""
    wall_1 = Wall(entities=Surface(name="fluid/wing1"))
    wall_2 = Wall(entities=Surface(name="fluid/wing2"))

    non_wall2_context = ParamsValidationInfo({}, [])
    non_wall2_context.physics_model_dict = {wall_1.private_attribute_id: wall_1.model_dump()}

    with ValidationContext(CASE, non_wall2_context), pytest.raises(
        ValueError,
        match=re.escape("The model does not exist in simulation params' models list."),
    ):
        with imperial_unit_system:
            SimulationParams(
                models=[Fluid(), wall_1],
                outputs=[
                    fl.ForceOutput(
                        name="force_output",
                        models=[wall_2.private_attribute_id],
                        output_fields=["CL", "CD"],
                    )
                ],
            )


def test_force_output_with_moving_statistic():
    """Test ForceOutput with moving statistics."""
    wall_1 = Wall(entities=Surface(name="fluid/wing"))

    with imperial_unit_system:
        SimulationParams(
            models=[Fluid(), wall_1],
            outputs=[
                fl.ForceOutput(
                    name="force_output",
                    models=[wall_1],
                    output_fields=["CL", "CD"],
                    moving_statistic=fl.MovingStatistic(
                        method="mean", moving_window_size=20, start_step=100
                    ),
                )
            ],
        )


def test_force_output_with_model_id():
    # [Frontend] Simulating loading a ForceOutput object with the id of models,
    # ensure the validation for models works
    with open("data/simulation_force_output_webui.json", "r") as fh:
        data = json.load(fh)

    _, errors, _ = validate_model(
        params_as_dict=data, validated_by=ValidationCalledBy.LOCAL, root_item_type="VolumeMesh"
    )
    print(errors)
    expected_errors = [
        {
            "type": "value_error",
            "loc": ("outputs", 6, "models"),
            "msg": "Value error, Duplicate models are not allowed in the same `ForceOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 7, "models"),
            "msg": "Value error, When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 8, "models"),
            "msg": "Value error, The model does not exist in simulation params' models list.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("run_control", "stopping_criteria", 2, "monitor_output", "models"),
            "msg": "Value error, Duplicate models are not allowed in the same `ForceOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(expected_errors)
    for err, exp_err in zip(errors, expected_errors):
        assert err["loc"] == exp_err["loc"]
        assert err["type"] == exp_err["type"]
        assert err["ctx"]["relevant_for"] == exp_err["ctx"]["relevant_for"]
        assert err["msg"] == exp_err["msg"]
