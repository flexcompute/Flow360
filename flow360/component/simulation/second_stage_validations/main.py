from typing import Literal

from flow360.component.simulation.second_stage_validations.case_validation import (
    _to_case_validation,
)
from flow360.component.simulation.second_stage_validations.surface_mesh_validation import (
    _to_surface_mesh_validation,
)
from flow360.component.simulation.second_stage_validations.validation_result import (
    ValidationResult,
)
from flow360.component.simulation.second_stage_validations.volume_mesh_validation import (
    _to_volume_mesh_validation,
)
from flow360.component.simulation.simulation_params import SimulationParams


def destination_validation(
    params: SimulationParams,
    entity_data: dict,
    root_item_type: Literal["Geometry", "VolumeMesh"],
    destination_item_type: Literal["SurfaceMesh", "VolumeMesh", "Case"] = "Case",
    raise_on_error: bool = False,
):
    if root_item_type == "VolumeMesh" and destination_item_type in {"SurfaceMesh", "VolumeMesh"}:
        raise ValueError(
            f"Invalid destination item type ({destination_item_type}) for root item type ({root_item_type})"
        )
    case_validation_results = ValidationResult()
    surface_meshing_validation_results = ValidationResult()
    volume_meshing_validation_results = ValidationResult()
    # TODO: proper args for the validation functions
    case_validation_results = _to_case_validation()
    if root_item_type == "Geometry":
        volume_meshing_validation_results = _to_volume_mesh_validation()
        surface_meshing_validation_results = _to_surface_mesh_validation()
    if raise_on_error is False:
        return (
            case_validation_results,
            volume_meshing_validation_results,
            surface_meshing_validation_results,
        )
    else:
        raise ValueError("Proper formating of errors pending ...")
