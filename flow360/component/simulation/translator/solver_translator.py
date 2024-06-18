"""Flow360 solver setting parameter translator."""

from copy import deepcopy
from typing import Union

from flow360.component.simulation.models.surface_models import (
    Freestream,
    SlipWall,
    SymmetryPlane,
    Wall,
)
from flow360.component.simulation.models.volume_models import BETDisk, Fluid, Rotation
from flow360.component.simulation.outputs.outputs import (
    SliceOutput,
    SurfaceOutput,
    TimeAverageVolumeOutput,
    VolumeOutput,
)
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.component.simulation.translator.utils import (
    convert_tuples_to_lists,
    get_attribute_from_first_instance,
    has_instance_in_list,
    merge_unique_item_lists,
    preprocess_input,
    remove_units_in_dict,
    replace_dict_key,
    replace_dict_value,
)
from flow360.component.simulation.unit_system import LengthType


def dump_dict(input_params):
    """Dumping param/model to dictionary."""
    return input_params.model_dump(by_alias=True, exclude_none=True)


def remove_empty_keys(input_dict):
    """I do not know what this is for --- Ben"""
    # pylint: disable=fixme
    # TODO: implement
    return input_dict


def init_output_attr_dict(obj_list, class_type):
    """Initialize the common output attribute."""
    return {
        "animationFrequency": get_attribute_from_first_instance(obj_list, class_type, "frequency"),
        "animationFrequencyOffset": get_attribute_from_first_instance(
            obj_list, class_type, "frequency_offset"
        ),
        "outputFormat": get_attribute_from_first_instance(obj_list, class_type, "output_format"),
    }


# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=too-many-locals
@preprocess_input
def get_solver_json(
    input_params: Union[str, dict, SimulationParams],
    # pylint: disable=no-member
    mesh_unit: LengthType.Positive,
):
    """
    Get the solver json from the simulation parameters.
    """

    translated = {}
    ##:: Step 1: Get geometry:
    geometry = remove_units_in_dict(dump_dict(input_params.reference_geometry))
    ml = geometry["momentLength"]
    translated["geometry"] = {
        "momentCenter": list(geometry["momentCenter"]),
        "momentLength": list(ml) if isinstance(ml, tuple) else [ml, ml, ml],
        "refArea": geometry["area"],
    }
    ##:: Step 2: Get freestream
    op = input_params.operating_condition
    translated["freestream"] = {
        "alphaAngle": op.alpha.to("degree").v.item(),
        "betaAngle": op.beta.to("degree").v.item(),
        "Mach": op.velocity_magnitude.v.item(),
        "Temperature": op.thermal_state.temperature.to("K").v.item(),
        "muRef": op.thermal_state.mu_ref(mesh_unit),
    }
    if "reference_velocity_magnitude" in op.model_fields.keys() and op.reference_velocity_magnitude:
        translated["freestream"]["MachRef"] = op.reference_velocity_magnitude.v.item()

    ##:: Step 3: Get boundaries
    translated["boundaries"] = {}
    for model in input_params.models:
        if isinstance(model, (Freestream, SlipWall, SymmetryPlane, Wall)):
            spec = dump_dict(model)
            spec.pop("surfaces")
            if isinstance(model, Wall):
                spec.pop("useWallFunction")
                spec["type"] = "WallFunction" if model.use_wall_function else "NoSlipWall"
                if model.heat_spec:
                    spec.pop("heat_spec")
                    # pylint: disable=fixme
                    # TODO: implement
            for surface in model.entities.stored_entities:
                translated["boundaries"][surface.name] = spec

    ##:: Step 4: Get outputs
    outputs = input_params.outputs

    if has_instance_in_list(outputs, VolumeOutput):
        translated["volumeOutput"] = init_output_attr_dict(outputs, VolumeOutput)
        replace_dict_value(translated["volumeOutput"], "outputFormat", "both", "paraview,tecplot")
        translated["volumeOutput"].update(
            {
                "outputFields": get_attribute_from_first_instance(
                    outputs, VolumeOutput, "output_fields"
                ).model_dump()["items"],
                "computeTimeAverages": False,
                "animationFrequencyTimeAverageOffset": 0,
                "animationFrequencyTimeAverage": -1,
                "startAverageIntegrationStep": -1,
            }
        )

    if has_instance_in_list(outputs, SurfaceOutput):
        translated["surfaceOutput"] = init_output_attr_dict(outputs, SurfaceOutput)
        replace_dict_value(translated["surfaceOutput"], "outputFormat", "both", "paraview,tecplot")
        translated["surfaceOutput"].update(
            {
                "writeSingleFile": get_attribute_from_first_instance(
                    outputs, SurfaceOutput, "write_single_file"
                ),
                "surfaces": {},
                "computeTimeAverages": False,
                "animationFrequencyTimeAverageOffset": 0,
                "animationFrequencyTimeAverage": -1,
                "startAverageIntegrationStep": -1,
                "outputFields": [],
            }
        )

    if has_instance_in_list(outputs, SliceOutput):
        translated["sliceOutput"] = init_output_attr_dict(outputs, SliceOutput)
        replace_dict_value(translated["sliceOutput"], "outputFormat", "both", "paraview,tecplot")
        translated["sliceOutput"].update({"slices": {}, "outputFields": []})

    for output in input_params.outputs:
        # validation: no more than one VolumeOutput, Slice and Surface cannot have difference format etc.
        if isinstance(output, TimeAverageVolumeOutput):
            # pylint: disable=fixme
            # TODO: update time average entries
            translated["volumeOutput"]["computeTimeAverages"] = True

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
            for slice_item in output.entities.items:
                slices[slice_item.name] = {
                    "outputFields": merge_unique_item_lists(
                        slices.get(slice_item.name, {}).get("outputFields", []),
                        output.output_fields.model_dump()["items"],
                    ),
                    "sliceOrigin": list(remove_units_in_dict(dump_dict(slice_item))["sliceOrigin"]),
                    "sliceNormal": list(remove_units_in_dict(dump_dict(slice_item))["sliceNormal"]),
                }

    ##:: Step 5: Get timeStepping
    ts = input_params.time_stepping
    if ts.type_name == "Unsteady":
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": ts.steps,
            "orderOfAccuracy": ts.order_of_accuracy,
            "maxPseudoSteps": ts.max_pseudo_steps,
            "timeStepSize": ts.step_size,
        }
    else:
        translated["timeStepping"] = {
            "CFL": dump_dict(ts.CFL),
            "physicalSteps": 1,
            "orderOfAccuracy": ts.order_of_accuracy,
            "maxPseudoSteps": ts.max_steps,
            "timeStepSize": "inf",
        }
    dump_dict(input_params.time_stepping)

    ##:: Step 6: Get solver settings
    for model in input_params.models:
        if isinstance(model, Fluid):
            translated["navierStokesSolver"] = dump_dict(model.navier_stokes_solver)
            replace_dict_key(translated["navierStokesSolver"], "typeName", "modelType")
            translated["turbulenceModelSolver"] = dump_dict(model.turbulence_model_solver)
            replace_dict_key(translated["turbulenceModelSolver"], "typeName", "modelType")
            modeling_constants = translated["turbulenceModelSolver"].get("modelingConstants", None)
            if modeling_constants:
                modeling_constants["C_d"] = modeling_constants.pop("CD", None)
                modeling_constants["C_DES"] = modeling_constants.pop("CDES", None)
                modeling_constants.pop("typeName", None)

    ##:: Step 7: Get BET and AD lists
    for model in input_params.models:
        if isinstance(model, BETDisk):
            disk_param = convert_tuples_to_lists(remove_units_in_dict(dump_dict(model)))
            replace_dict_key(disk_param, "machNumbers", "MachNumbers")
            replace_dict_key(disk_param, "reynoldsNumbers", "ReynoldsNumbers")
            volumes = disk_param.pop("volumes")
            for extra_attr in ["name", "type"]:
                if extra_attr in disk_param:
                    disk_param.pop(extra_attr)
            for v in volumes["storedEntities"]:
                disk_i = deepcopy(disk_param)
                disk_i["axisOfRotation"] = v["axis"]
                disk_i["centerOfRotation"] = v["center"]
                disk_i["radius"] = v["outerRadius"]
                disk_i["thickness"] = v["height"]
                bet_disks = translated.get("BETDisks", [])
                bet_disks.append(disk_i)
                translated["BETDisks"] = bet_disks

    ##:: Step 8: Get rotation
    for model in input_params.models:
        if isinstance(model, Rotation):
            for volume in model.entities.stored_entities:
                volumeZone = {
                    "modelType": "FluidDynamics",
                    "referenceFrame": {
                        "axisOfRotation": list(volume.axis),
                        "centerOfRotation": list(volume.center),
                    },
                }
                if model.parent_volume:
                    volumeZone["referenceFrame"]["parentVolumeName"] = model.parent_volume.name
                spec = dump_dict(model)["spec"]
                if isinstance(spec, str):
                    volumeZone["referenceFrame"]["thetaRadians"] = spec
                elif spec.get("units", "") == "flow360_angular_velocity_unit":
                    volumeZone["referenceFrame"]["omegaRadians"] = spec["value"]
            volumeZones = translated.get("volumeZones", {})
            volumeZones.update({volume.name: volumeZone})
            translated["volumeZones"] = volumeZones

    ##:: Step 9: Get porous media

    ##:: Step 10: Get heat transfer zones

    ##:: Step 11: Get user defined dynamics
    if input_params.user_defined_dynamics is not None:
        translated["userDefinedDynamics"] = []
        for udd in input_params.user_defined_dynamics:
            udd_dict = dump_dict(udd)
            udd_dict["dynamicsName"] = udd_dict.pop("name")
            if udd.input_boundary_patches is not None:
                udd_dict["inputBoundaryPatches"] = []
                for surface in udd.input_boundary_patches.stored_entities:
                    udd_dict["inputBoundaryPatches"].append(surface.name)
            if udd.output_target is not None:
                udd_dict["outputTargetName"] = udd.output_target.name
            translated["userDefinedDynamics"].append(udd_dict)

    return translated
