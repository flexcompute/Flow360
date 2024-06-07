"""
Flow360 simulation parameters
"""

from __future__ import annotations

from typing import List, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.meshing_param.params import MeshingParameters
from flow360.component.simulation.models.surface_models import SurfaceModelTypes
from flow360.component.simulation.models.volume_models import VolumeModelTypes
from flow360.component.simulation.operating_condition import OperatingConditionTypes
from flow360.component.simulation.outputs.outputs import OutputTypes
from flow360.component.simulation.primitives import ReferenceGeometry
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import (
    UnitSystem,
    UnitSystemType,
    unit_system_manager,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.error_messages import unit_system_inconsistent_msg, use_unit_system_msg
from flow360.exceptions import Flow360ConfigurationError, Flow360RuntimeError
from flow360.version import __version__


class SimulationParams(Flow360BaseModel):
    """
        meshing (Optional[MeshingParameters]): Contains all the user specified meshing parameters that either enrich or
        modify the existing surface/volume meshing parameters from starting points.

    -----
        - Global settings that gets applied by default to all volumes/surfaces. However per-volume/per-surface values
        will **always** overwrite global ones.

        reference_geometry (Optional[ReferenceGeometry]): Global geometric reference values.
        operating_condition (Optional[OperatingConditionTypes]): Global operating condition.
    -----
        - `volumes` and `surfaces` describes the physical problem **numerically**. Therefore `volumes` may/maynot
        necessarily have to map to grid volume zones (e.g. BETDisk). For now `surfaces` are used exclusivly for boundary
        conditions.

        volumes (Optional[List[VolumeTypes]]): Numerics/physics defined on a volume.
        surfaces (Optional[List[SurfaceTypes]]): Numerics/physics defined on a surface.
    -----
        - Other configurations that are orthogonal to all previous items.

        time_stepping (Optional[Union[SteadyTimeStepping, UnsteadyTimeStepping]]): Temporal aspects of simulation.
        user_defined_dynamics (Optional[UserDefinedDynamics]): Additional user-specified dynamics on top of the existing
        ones or how volumes/surfaces are intertwined.
        outputs (Optional[List[OutputTypes]]): Surface/Slice/Volume/Isosurface outputs."""

    unit_system: UnitSystemType = pd.Field(frozen=True, discriminator="name")
    version: str = pd.Field(__version__, frozen=True)

    meshing: Optional[MeshingParameters] = pd.Field(None)
    reference_geometry: Optional[ReferenceGeometry] = pd.Field(None)
    operating_condition: OperatingConditionTypes = pd.Field()
    #
    """
    meshing->edge_refinement, face_refinement, zone_refinement, volumes and surfaces should be class which has the:
    1. __getitem__ to allow [] access
    2. __setitem__ to allow [] assignment
    3. by_name(pattern:str) to use regexpr/glob to select all zones/surfaces with matched name
    3. by_type(pattern:str) to use regexpr/glob to select all zones/surfaces with matched type
    """
    models: Optional[List[Union[VolumeModelTypes, SurfaceModelTypes]]] = pd.Field(None)
    """
    Below can be mostly reused with existing models 
    """
    time_stepping: Optional[Union[Steady, Unsteady]] = pd.Field(None)
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = pd.Field(None)
    """
    Support for user defined expression?
    If so:
        1. Move over the expression validation functions.
        2. Have camelCase to snake_case naming converter for consistent user experience.
    Limitations:
        1. No per volume zone output. (single volume output)
    """
    outputs: Optional[List[OutputTypes]] = pd.Field(None)

    model_config = pd.ConfigDict(include_hash=True)

    def _init_check_unit_system(self, **kwargs):
        """
        Check existence of unit system and raise an error if it is not set or inconsistent.
        """
        if unit_system_manager.current is None:
            raise Flow360RuntimeError(use_unit_system_msg)
        # pylint: disable=duplicate-code
        kwarg_unit_system = kwargs.pop("unit_system", None)
        if kwarg_unit_system is not None:
            if not isinstance(kwarg_unit_system, UnitSystem):
                kwarg_unit_system = UnitSystem.from_dict(**kwarg_unit_system)
            if kwarg_unit_system != unit_system_manager.current:
                raise Flow360RuntimeError(
                    unit_system_inconsistent_msg(
                        kwarg_unit_system.system_repr(), unit_system_manager.current.system_repr()
                    )
                )

        return kwargs

    def _init_no_context(self, filename, **kwargs):
        """
        Initialize the simulation parameters without a unit context.
        """
        if unit_system_manager.current is not None:
            raise Flow360RuntimeError(
                "When loading params from file: SimulationParams(filename), "
                "unit context must not be used."
            )

        model_dict = self._handle_file(filename=filename, **kwargs)

        version = model_dict.pop("version", None)
        unit_system = model_dict.get("unit_system")
        if version is not None and unit_system is not None:
            if version != __version__:
                raise NotImplementedError("No legacy support at the time being.")
            # pylint: disable=not-context-manager
            with UnitSystem.from_dict(**unit_system):
                super().__init__(**model_dict)
        else:
            raise Flow360RuntimeError(
                "Missing version or unit system info in file content, please check the input file."
            )

    def _init_with_context(self, **kwargs):
        """
        Initializes the simulation parameters with the given unit context.
        """
        kwargs = self._init_check_unit_system(**kwargs)
        super().__init__(unit_system=unit_system_manager.copy_current(), **kwargs)

    # pylint: disable=super-init-not-called
    # pylint: disable=fixme
    # TODO: avoid overloading the __init__ so IDE can proper prompt root level keys
    def __init__(self, filename: str = None, **kwargs):
        if filename is not None:
            self._init_no_context(filename, **kwargs)
        else:
            self._init_with_context(**kwargs)

    def copy(self, update=None, **kwargs) -> SimulationParams:
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                return super().copy(update=update, **kwargs)

        return super().copy(update=update, **kwargs)

    # pylint: disable=arguments-differ
    def preprocess(self, mesh_unit) -> SimulationParams:
        """TBD"""
        if mesh_unit is None:
            raise Flow360ConfigurationError("Mesh unit has not been supplied.")
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                return super().preprocess(self, mesh_unit=mesh_unit)
        return super().preprocess(self, mesh_unit=mesh_unit)