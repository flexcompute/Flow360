import json
import os

from flow360.component.simulation.framework.param_utils import AssetCache
from flow360.component.simulation.models.surface_models import Wall
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.outputs import SurfaceOutput, VolumeOutput
from flow360.component.simulation.services import ValidationCalledBy, validate_model
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.time_stepping.time_stepping import Steady
from flow360.component.simulation.unit_system import imperial_unit_system
from flow360.component.volume_mesh import VolumeMeshV2


def test_output_frequency_settings_in_steady_simulation():
    volume_mesh = VolumeMeshV2.from_local_storage(
        mesh_id=None,
        local_storage_path=os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "vm_entity_provider",
        ),
    )
    simulation_path = os.path.join(
        os.path.dirname(__file__),
        "..",
        "data",
        "vm_entity_provider",
        "simulation.json",
    )
    with open(simulation_path, "r") as file:
        asset_cache_data = json.load(file).pop("private_attribute_asset_cache")
    asset_cache = AssetCache.deserialize(asset_cache_data)
    with imperial_unit_system:
        params = SimulationParams(
            models=[Wall(name="wall", entities=volume_mesh["*"])],
            time_stepping=Steady(),
            outputs=[
                VolumeOutput(output_fields=["Mach", "Cp"], frequency=2),
                SurfaceOutput(
                    output_fields=["Cp"],
                    entities=volume_mesh["*"],
                    frequency_offset=10,
                ),
            ],
            private_attribute_asset_cache=asset_cache,
        )

    params_as_dict = params.model_dump(exclude_none=True, mode="json")
    _, errors, _ = validate_model(
        params_as_dict=params_as_dict,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
        validation_level="All",
    )

    expected_errors = [
        {
            "loc": ("outputs", 0, "frequency"),
            "type": "value_error",
            "msg": "Value error, Output frequency cannot be specified in a steady simulation.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "loc": ("outputs", 1, "frequency_offset"),
            "type": "value_error",
            "msg": "Value error, Output frequency_offset cannot be specified in a steady simulation.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]
    assert len(errors) == len(expected_errors)
    for error, expected in zip(errors, expected_errors):
        assert error["loc"] == expected["loc"]
        assert error["type"] == expected["type"]
        assert error["msg"] == expected["msg"]
        assert error["ctx"]["relevant_for"] == expected["ctx"]["relevant_for"]


def test_force_output_with_model_id():
    simulation_path = os.path.join(
        os.path.dirname(__file__),
        "data",
        "simulation_force_output_webui.json",
    )
    with open(simulation_path, "r") as file:
        data = json.load(file)

    _, errors, _ = validate_model(
        params_as_dict=data,
        validated_by=ValidationCalledBy.LOCAL,
        root_item_type="VolumeMesh",
    )
    expected_errors = [
        {
            "type": "value_error",
            "loc": ("outputs", 3, "models"),
            "msg": "Value error, Duplicate models are not allowed in the same `ForceOutput`.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 4, "models"),
            "msg": "Value error, When ActuatorDisk/BETDisk/PorousMedium is specified, "
            "only CL, CD, CFx, CFy, CFz, CMx, CMy, CMz can be set as output_fields.",
            "ctx": {"relevant_for": ["Case"]},
        },
        {
            "type": "value_error",
            "loc": ("outputs", 5, "models"),
            "msg": "Value error, The model does not exist in simulation params' models list.",
            "ctx": {"relevant_for": ["Case"]},
        },
    ]

    assert len(errors) == len(expected_errors)
    for error, expected in zip(errors, expected_errors):
        assert error["loc"] == expected["loc"]
        assert error["type"] == expected["type"]
        assert error["ctx"]["relevant_for"] == expected["ctx"]["relevant_for"]
        assert error["msg"] == expected["msg"]
