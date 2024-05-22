from typing import List, Optional, Union

import pydantic as pd

## Warning: pydantic V1
from flow360.component.flow360_params.unit_system import (
    UnitSystemType,
    unit_system_manager,
)
from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.params import MeshingParameters
from flow360.component.simulation.operating_condition import OperatingConditionTypes
from flow360.component.simulation.outputs import OutputTypes
from flow360.component.simulation.references import ReferenceGeometry
from flow360.component.simulation.surfaces import SurfaceTypes
from flow360.component.simulation.time_stepping.time_stepping import (
    SteadyTimeStepping,
    UnsteadyTimeStepping,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamics,
)
from flow360.component.simulation.volumes import VolumeTypes
from flow360.exceptions import Flow360ConfigError
from flow360.log import log
from flow360.user_config import UserConfig


class SimulationParams(Flow360BaseModel):
    """
        meshing (Optional[MeshingParameters]): Contains all the user specified meshing parameters that either enrich or modify the existing surface/volume meshing parameters from starting points.

    -----
        - Global settings that gets applied by default to all volumes/surfaces. However per-volume/per-surface values will **always** overwrite global ones.

        reference_geometry (Optional[ReferenceGeometry]): Global geometric reference values.
        operating_condition (Optional[OperatingConditionTypes]): Global operating condition.
    -----
        - `volumes` and `surfaces` describes the physical problem **numerically**. Therefore `volumes` may/maynot necessarily have to map to grid volume zones (e.g. BETDisk). For now `surfaces` are used exclusivly for boundary conditions.

        volumes (Optional[List[VolumeTypes]]): Numerics/physics defined on a volume.
        surfaces (Optional[List[SurfaceTypes]]): Numerics/physics defined on a surface.
    -----
        - Other configurations that are orthogonal to all previous items.

        time_stepping (Optional[Union[SteadyTimeStepping, UnsteadyTimeStepping]]): Temporal aspects of simulation.
        user_defined_dynamics (Optional[UserDefinedDynamics]): Additional user-specified dynamics on top of the existing ones or how volumes/surfaces are intertwined.
        outputs (Optional[List[OutputTypes]]): Surface/Slice/Volume/Isosurface outputs."""

    meshing: Optional[MeshingParameters] = pd.Field(None)

    reference_geometry: Optional[ReferenceGeometry] = pd.Field(None)
    operating_condition: Optional[OperatingConditionTypes] = pd.Field(None)
    #
    """
    meshing->edge_refinement, face_refinement, zone_refinement, volumes and surfaces should be class which has the:
    1. __getitem__ to allow [] access
    2. __setitem__ to allow [] assignment
    3. by_name(pattern:str) to use regexpr/glob to select all zones/surfaces with matched name
    3. by_type(pattern:str) to use regexpr/glob to select all zones/surfaces with matched type
    """
    models: Optional[List[Union[VolumeTypes, SurfaceTypes]]] = pd.Field(None)
    """
    Below can be mostly reused with existing models 
    """
    time_stepping: Optional[Union[SteadyTimeStepping, UnsteadyTimeStepping]] = pd.Field(None)
    user_defined_dynamics: Optional[List[UserDefinedDynamics]] = pd.Field(None)
    """
    Support for user defined expression?
    If so:
        1. Move over the expression validation functions.
        2. Have camelCase to snake_case naming converter for consistent user experience.
    Limitations:
        1. No per volume zone output. (single volume output)
    """
    outputs: Optional[List[OutputTypes]] = pd.Field(None)


class UnvalidatedSimulationParams(Flow360BaseModel):
    """
    Unvalidated parameters
    """

    model_config = pd.ConfigDict(extra="allow")

    def __init__(self, filename: str = None, **kwargs):
        if UserConfig.do_validation:
            raise Flow360ConfigError(
                "This is DEV feature. To use it activate by: fl.UserConfig.disable_validation()."
            )
        log.warning("This is DEV feature, use it only when you know what you are doing.")
        super().__init__(filename, **kwargs)

    def flow360_json(self) -> str:
        """Generate a JSON representation of the model"""

        # return self.json(encoder=flow360_json_encoder)
        pass
