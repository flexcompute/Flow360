import re
from typing import Union

from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.models.volume_models import Fluid
from flow360.component.simulation.outputs.outputs import (
    SliceOutput,
    SurfaceOutput,
    TimeAverageVolumeOutput,
    VolumeOutput,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    get_attribute_from_first_instance,
    preprocess_input,
)
from flow360.component.simulation.unit_system import LengthType


def to_dict(input_params):
    return input_params.model_dump(by_alias=True, exclude_none=True)


def to_dict_without_unit(input_params):
    unit_keys = {"value", "units"}
    output_dict = to_dict(input_params)
    for key, value in output_dict.items():
        output_dict[key] = (
            value["value"] if isinstance(value, dict) and value.keys() == unit_keys else value
        )
    return output_dict


def remove_empty_keys(input_dict):
    # TODO: implement
    return input_dict


def init_output_attr_dict(obj_list, class_type):
    return {
        "animationFrequency": get_attribute_from_first_instance(obj_list, class_type, "frequency"),
        "animationFrequencyOffset": get_attribute_from_first_instance(
            obj_list, class_type, "frequency_offset"
        ),
        "outputFormat": get_attribute_from_first_instance(obj_list, class_type, "output_format"),
    }


def get_output_fields(obj_list, class_type):
    return get_attribute_from_first_instance(obj_list, class_type, "output_fields").model_dump()[
        "items"
    ]


def merge_unique_item_lists(list1: list, list2: list) -> list:
    # TODO: implement
    return list1 + list2


@preprocess_input
def get_case_json(
    input_params: Union[str, dict, SimulationParams],
    mesh_unit: LengthType.Positive,
):
    """
    Get the surface mesh json from the simulation parameters.
    """
    outputs = input_params.outputs
    translated = {
        "boundaries": {},
        "volumeOutput": init_output_attr_dict(outputs, VolumeOutput),
        "surfaceOutput": init_output_attr_dict(outputs, SurfaceOutput),
        "sliceOutput": init_output_attr_dict(outputs, SliceOutput),
    }
    translated["volumeOutput"].update(
        {"outputFields": get_output_fields(outputs, VolumeOutput), "computeTimeAverages": False}
    )
    translated["surfaceOutput"].update(
        {
            "writeSingleFile": get_attribute_from_first_instance(
                outputs, SurfaceOutput, "write_single_file"
            ),
            "surfaces": {},
            "computeTimeAverages": False,
        }
    )
    translated["sliceOutput"].update({"slices": {}})

    bd = translated["boundaries"]

    op = input_params.operating_condition
    translated["freestream"] = {
        "alpha": op.alpha.to("degree").v.item(),
        "beta": op.beta.to("degree").v.item(),
        "Mach": op.velocity_magnitude.v.item(),
        "Temperature": op.thermal_state.temperature.to("K").v.item(),
        "muRef": op.thermal_state.mu_ref(mesh_unit),
    }
    if "reference_velocity_magnitude" in op and op.reference_velocity_magnitude:
        translated["freestream"]["MachRef"] = op.reference_velocity_magnitude.v.item()

    translated["timeStepping"] = to_dict(input_params.time_stepping)

    geometry = to_dict_without_unit(input_params.reference_geometry)
    translated["geometry"] = {
        "momentCenter": geometry["momentCenter"],
        "momentLength": geometry["momentLength"],
        "refArea": geometry["area"],
    }

    for model in input_params.models:
        if isinstance(model, Fluid):
            translated["navierStokesSolver"] = to_dict(model.navier_stokes_solver)
            translated["turbulenceModelSolver"] = to_dict(model.turbulence_model_solver)
        elif isinstance(model, Union[Freestream, SlipWall, SymmetryPlane, Wall]):
            for surface in model.entities.stored_entities:
                spec = to_dict(model)
                spec.pop("surfaces")
            if isinstance(model, Wall):
                spec.pop("useWallFunction")
                spec["type"] = "WallFunction" if model.use_wall_function else "NoSlipWall"
                if model.heat_spec:
                    spec.pop("heat_spec")
                    # TODO: implement
            bd[surface.name] = spec

    for output in input_params.outputs:
        # validation: no more than one VolumeOutput, Slice and Surface cannot have difference format etc.
        if isinstance(output, TimeAverageVolumeOutput):
            # TODO: update time average entries
            translated["volumeOutput"]["computeTimeAverages"] = True

            # "outputFields": get_output_fields(output, VolumeOutput)
        elif isinstance(output, SurfaceOutput):
            surfaces = translated["surfaceOutput"]["surfaces"]
            for surface in output.entities.stored_entities:
                surfaces[surface.name] = {
                    "outputFields": merge_unique_item_lists(
                        surfaces.get(surface.name, {}).get("outputFields", []),
                        output.output_fields.model_dump()["items"],
                    )
                }
        elif isinstance(output, SliceOutput):
            slices = translated["sliceOutput"]["slices"]
            # slices = to_dict(output).pop("slices")
            for slice in output.entities.items:
                slices[slice.name] = {
                    "outputFields": merge_unique_item_lists(
                        slices.get(slice.name, {}).get("outputFields", []),
                        output.output_fields.model_dump()["items"],
                    ),
                    "sliceOrigin": to_dict(slice)["sliceOrigin"],
                    "sliceNormal": to_dict(slice)["sliceNormal"],
                }
    print(translated["geometry"])
    print(translated["boundaries"])
    # print(translated["volumeOutput"])
    # print(translated["surfaceOutput"])
    # print(translated["sliceOutput"])

    return translated
