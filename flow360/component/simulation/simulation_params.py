"""
Flow360 simulation parameters
"""

from __future__ import annotations

from typing import Annotated, List, Literal, Optional, Union

import pydantic as pd

from flow360.component.simulation.framework.base_model import Flow360BaseModel
from flow360.component.simulation.framework.entity_registry import EntityRegistry
from flow360.component.simulation.framework.param_utils import (
    AssetCache,
    _set_boundary_full_name_with_zone_name,
    _update_entity_full_name,
    _update_zone_boundaries_with_metadata,
    register_entity_list,
)
from flow360.component.simulation.framework.updater import updater
from flow360.component.simulation.meshing_param.params import MeshingParams
from flow360.component.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationCylinder,
)
from flow360.component.simulation.models.surface_models import SurfaceModelTypes
from flow360.component.simulation.models.volume_models import (
    ActuatorDisk,
    BETDisk,
    Fluid,
    VolumeModelTypes,
)
from flow360.component.simulation.operating_condition.operating_condition import (
    OperatingConditionTypes,
)
from flow360.component.simulation.outputs.outputs import (
    AeroAcousticOutput,
    IsosurfaceOutput,
    OutputTypes,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceProbeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360.component.simulation.primitives import (
    ReferenceGeometry,
    _SurfaceEntityBase,
    _VolumeEntityBase,
)
from flow360.component.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360.component.simulation.unit_system import (
    UnitSystem,
    UnitSystemType,
    unit_system_manager,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.validation.validation_output import (
    _check_output_fields,
)
from flow360.component.simulation.validation.validation_simulation_params import (
    _check_cht_solver_settings,
    _check_complete_boundary_condition_and_unknown_surface,
    _check_consistency_ddes_volume_output,
    _check_consistency_wall_function_and_surface_output,
    _check_duplicate_entities_in_models,
    _check_low_mach_preconditioner_output,
    _check_numerical_dissipation_factor_output,
    _check_parent_volume_is_rotating,
)
from flow360.error_messages import (
    unit_system_inconsistent_msg,
    use_unit_system_for_simulation_msg,
)
from flow360.exceptions import Flow360ConfigurationError, Flow360RuntimeError
from flow360.version import __version__

from .validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
    CaseField,
    ConditionalField,
    context_validator,
)

ModelTypes = Annotated[Union[VolumeModelTypes, SurfaceModelTypes], pd.Field(discriminator="type")]


class _ParamModelBase(Flow360BaseModel):
    """
    Base class that abstracts out all Param type classes in Flow360.
    """

    version: str = pd.Field(__version__, frozen=True)
    unit_system: UnitSystemType = pd.Field(frozen=True, discriminator="name")
    model_config = pd.ConfigDict(include_hash=True)

    def _init_check_unit_system(self, **kwargs):
        """
        Check existence of unit system and raise an error if it is not set or inconsistent.
        """
        if unit_system_manager.current is None:
            raise Flow360RuntimeError(use_unit_system_for_simulation_msg)
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
                f"When loading params from file: {self.__class__.__name__}(filename), "
                "unit context must not be used."
            )

        model_dict = self._handle_file(filename=filename, **kwargs)

        version = model_dict.pop("version", None)
        unit_system = model_dict.get("unit_system")
        if version is not None and unit_system is not None:
            if version != __version__:
                model_dict = updater(  # pylint: disable=R0801
                    version_from=version, version_to=__version__, params_as_dict=model_dict
                )
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
        super().__init__(unit_system=unit_system_manager.current, **kwargs)

    # pylint: disable=super-init-not-called
    # pylint: disable=fixme
    # TODO: avoid overloading the __init__ so IDE can proper prompt root level keys
    def __init__(self, filename: str = None, **kwargs):
        if filename is not None:
            self._init_no_context(filename, **kwargs)
        else:
            self._init_with_context(**kwargs)

    def copy(self, update=None, **kwargs) -> _ParamModelBase:
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                return super().copy(update=update, **kwargs)

        return super().copy(update=update, **kwargs)


# pylint: disable=too-many-public-methods
class SimulationParams(_ParamModelBase):
    """All-in-one class for surface meshing + volume meshing + case configurations"""

    meshing: Optional[MeshingParams] = ConditionalField(
        None,
        context=[SURFACE_MESH, VOLUME_MESH],
        description="Surface and volume meshing parameters. See :class:`MeshingParams` for more details.",
    )

    reference_geometry: Optional[ReferenceGeometry] = CaseField(
        None,
        description="Global geometric reference values. See :class:`ReferenceGeometry` for more details.",
    )
    operating_condition: Optional[OperatingConditionTypes] = CaseField(
        None,
        discriminator="type_name",
        description="Global operating condition."
        " See :ref:`Operating Condition <operating_condition>` for more details.",
    )
    #

    # meshing->edge_refinement, face_refinement, zone_refinement, volumes and surfaces should be class which has the:
    # 1. __getitem__ to allow [] access
    # 2. __setitem__ to allow [] assignment
    # 3. by_name(pattern:str) to use regexpr/glob to select all zones/surfaces with matched name
    # 4. by_type(pattern:str) to use regexpr/glob to select all zones/surfaces with matched type

    models: Optional[List[ModelTypes]] = CaseField(
        None,
        description="Solver settings and numerical models and boundary condition settings."
        " See :ref:`Volume Models <volume_models>` and :ref:`Surface Models <surface_models>` for more details.",
    )
    time_stepping: Union[Steady, Unsteady] = CaseField(
        Steady(),
        discriminator="type_name",
        description="Time stepping settings. See :ref:`Time Stepping <timeStepping>` for more details.",
    )
    user_defined_dynamics: Optional[List[UserDefinedDynamic]] = CaseField(
        None,
        description="User defined dynamics. See :ref:`User Defined Dynamics <user_defined_dynamics>` for more details.",
    )

    user_defined_fields: List[UserDefinedField] = CaseField(
        [], description="User defined fields that can be used in outputs."
    )

    # Support for user defined expression?
    # If so:
    #    1. Move over the expression validation functions.
    #    2. Have camelCase to snake_case naming converter for consistent user experience.
    # Limitations:
    #    1. No per volume zone output. (single volume output)
    outputs: Optional[List[OutputTypes]] = CaseField(
        None, description="Output settings. See :ref:`Outputs <outputs>` for more details."
    )

    ##:: [INTERNAL USE ONLY] Private attributes that should not be modified manually.
    private_attribute_asset_cache: AssetCache = pd.Field(AssetCache(), frozen=True)

    # pylint: disable=arguments-differ
    def preprocess(self, mesh_unit=None, exclude: list = None) -> SimulationParams:
        """Internal function for non-dimensionalizing the simulation parameters"""
        if exclude is None:
            exclude = []

        if mesh_unit is None:
            raise Flow360ConfigurationError("Mesh unit has not been supplied.")
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                return super().preprocess(params=self, mesh_unit=mesh_unit, exclude=exclude)
        return super().preprocess(params=self, mesh_unit=mesh_unit, exclude=exclude)

    def convert_to_unit_system(
        self,
        unit_system: Literal["SI", "Imperial", "CGS"],
        exclude: list = None,
    ) -> SimulationParams:
        """Internal function for non-dimensionalizing the simulation parameters"""
        if exclude is None:
            exclude = []

        if unit_system not in ["SI", "Imperial", "CGS"]:
            raise Flow360ConfigurationError(
                f"Invalid unit system: {unit_system}. Must be one of ['SI', 'Imperial', 'CGS']"
            )
        converted_param = None
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                converted_param = super().convert_to_unit_system(
                    to_unit_system=unit_system, exclude=exclude
                )
        else:
            converted_param = super().convert_to_unit_system(
                to_unit_system=unit_system, exclude=exclude
            )
        # Change the recorded unit system
        with model_attribute_unlock(converted_param, "unit_system"):
            converted_param.unit_system = UnitSystem.from_dict(**{"name": unit_system})
        return converted_param

    # pylint: disable=no-self-argument
    @pd.field_validator("models", mode="after")
    @classmethod
    def apply_defult_fluid_settings(cls, v):
        """apply default Fluid() settings if not found in models"""
        if v is None:
            v = []
        assert isinstance(v, list)
        if not any(isinstance(item, Fluid) for item in v):
            v.append(Fluid())
        return v

    @pd.field_validator("models", mode="after")
    @classmethod
    def check_parent_volume_is_rotating(cls, models):
        """Ensure that all the parent volumes listed in the `Rotation` model are not static"""
        return _check_parent_volume_is_rotating(models)

    @pd.field_validator("user_defined_fields", mode="after")
    @classmethod
    def check_duplicate_user_defined_fields(cls, v):
        """Check if we have duplicate user defined fields"""
        if v == []:
            return v

        known_user_defined_fields = set()
        for field in v:
            if field.name in known_user_defined_fields:
                raise ValueError(f"Duplicate user defined field name: {field.name}")
            known_user_defined_fields.add(field.name)

        return v

    @pd.model_validator(mode="after")
    def check_cht_solver_settings(self):
        """Check the Conjugate Heat Transfer settings, transferred from checkCHTSolverSettings"""
        return _check_cht_solver_settings(self)

    @pd.model_validator(mode="after")
    def check_consistency_wall_function_and_surface_output(self):
        """Only allow wallFunctionMetric output field when there is a Wall model with a wall function enabled"""
        return _check_consistency_wall_function_and_surface_output(self)

    @pd.model_validator(mode="after")
    def check_consistency_ddes_volume_output(self):
        """Only allow DDES output field when there is a corresponding solver with DDES enabled in models"""
        return _check_consistency_ddes_volume_output(self)

    @pd.model_validator(mode="after")
    def check_duplicate_entities_in_models(self):
        """Only allow each Surface/Volume entity to appear once in the Surface/Volume model"""
        return _check_duplicate_entities_in_models(self)

    @pd.model_validator(mode="after")
    def check_numerical_dissipation_factor_output(self):
        """Only allow numericalDissipationFactor output field when the NS solver has low numerical dissipation"""
        return _check_numerical_dissipation_factor_output(self)

    @pd.model_validator(mode="after")
    def check_low_mach_preconditioner_output(self):
        """Only allow lowMachPreconditioner output field when the lowMachPreconditioner is enabled in the NS solver"""
        return _check_low_mach_preconditioner_output(self)

    @pd.model_validator(mode="after")
    @context_validator(context=CASE)
    def check_complete_boundary_condition_and_unknown_surface(self):
        """Make sure that all boundaries have been assigned with a boundary condition"""
        return _check_complete_boundary_condition_and_unknown_surface(self)

    @pd.model_validator(mode="after")
    def check_output_fields(params):
        """Check output fields and iso fields are valid"""
        return _check_output_fields(params)

    def _move_registry_to_asset_cache(self, registry: EntityRegistry) -> EntityRegistry:
        """Recursively register all entities listed in EntityList to the asset cache."""
        # pylint: disable=no-member
        registry.clear()
        register_entity_list(self, registry)
        return registry

    def _update_entity_private_attrs(self, registry: EntityRegistry) -> EntityRegistry:
        """
        Once the SimulationParams is set, extract and upate information
        into all used entities by parsing the params.
        """

        ##::1. Update full names in the Surface entities with zone names
        # pylint: disable=no-member
        if self.meshing is not None and self.meshing.volume_zones is not None:
            for volume in self.meshing.volume_zones:
                if isinstance(volume, AutomatedFarfield):
                    _set_boundary_full_name_with_zone_name(
                        registry,
                        "farfield",
                        volume.private_attribute_entity.name,
                    )
                    _set_boundary_full_name_with_zone_name(
                        registry,
                        "symmetric*",
                        volume.private_attribute_entity.name,
                    )
                if isinstance(volume, RotationCylinder):
                    # pylint: disable=fixme
                    # TODO: Implement this
                    pass

        return registry

    @property
    def used_entity_registry(self) -> EntityRegistry:
        """
        Get a entity registry that collects all the entities used in the simulation.
        And also try to update the entities now that we have a global view of the simulation.
        """
        registry = EntityRegistry()
        registry = self._move_registry_to_asset_cache(registry)
        registry = self._update_entity_private_attrs(registry)
        return registry

    def _update_param_with_actual_volume_mesh_meta(self, volume_mesh_meta_data: dict):
        """
        Update the zone info from the actual volume mesh before solver execution.
        Will be executed in the casePipeline as part of preprocessing.
        Some thoughts:
        Do we also need to update the params when the **surface meshing** is done?
        """
        # pylint:disable=no-member
        used_entity_registry = self.used_entity_registry
        _update_entity_full_name(self, _SurfaceEntityBase, volume_mesh_meta_data)
        _update_entity_full_name(self, _VolumeEntityBase, volume_mesh_meta_data)
        _update_zone_boundaries_with_metadata(used_entity_registry, volume_mesh_meta_data)
        return self

    def is_steady(self):
        """
        returns True when SimulationParams is steady state
        """
        return isinstance(self.time_stepping, Steady)

    def has_actuator_disks(self):
        """
        returns True when SimulationParams has ActuatorDisk disk
        """
        if self.models is None:
            return False
        # pylint: disable=not-an-iterable
        return any(isinstance(item, ActuatorDisk) for item in self.models)

    def has_bet_disks(self):
        """
        returns True when SimulationParams has BET disk
        """
        if self.models is None:
            return False
        # pylint: disable=not-an-iterable
        return any(isinstance(item, BETDisk) for item in self.models)

    def has_isosurfaces(self):
        """
        returns True when SimulationParams has isosurfaces
        """
        if self.outputs is None:
            return False
        # pylint: disable=not-an-iterable
        return any(isinstance(item, IsosurfaceOutput) for item in self.outputs)

    def has_monitors(self):
        """
        returns True when SimulationParams has monitors
        """
        if self.outputs is None:
            return False
        # pylint: disable=not-an-iterable
        return any(
            isinstance(item, (ProbeOutput, SurfaceProbeOutput, SurfaceIntegralOutput))
            for item in self.outputs
        )

    def has_volume_output(self):
        """
        returns True when SimulationParams has volume output
        """
        if self.outputs is None:
            return False
        # pylint: disable=not-an-iterable
        return any(isinstance(item, VolumeOutput) for item in self.outputs)

    def has_aeroacoustics(self):
        """
        returns True when SimulationParams has aeroacoustics
        """
        if self.outputs is None:
            return False
        # pylint: disable=not-an-iterable
        return any(isinstance(item, (AeroAcousticOutput)) for item in self.outputs)

    def has_user_defined_dynamics(self):
        """
        returns True when SimulationParams has user defined dynamics
        """
        return self.user_defined_dynamics is not None and len(self.user_defined_dynamics) > 0
