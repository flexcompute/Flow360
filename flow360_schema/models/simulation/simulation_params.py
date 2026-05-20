"""
Flow360 simulation parameters
"""

from __future__ import annotations

import logging
from typing import Annotated, Literal

import pydantic as pd
import unyt as u

from flow360_schema.exceptions import Flow360ValueError
from flow360_schema.framework.base_model import Flow360BaseModel
from flow360_schema.framework.entity.entity_registry import EntityRegistry
from flow360_schema.framework.expression import (
    UserVariable,
    batch_get_user_variable_units,
    compute_surface_integral_unit,
)
from flow360_schema.framework.param_utils import (
    _set_boundary_full_name_with_zone_name,
    _update_entity_full_name,
    _update_zone_boundaries_with_metadata,
    register_entity_list,
)
from flow360_schema.framework.physical_dimensions import (
    AbsoluteTemperature,
    Density,
    Length,
    Mass,
    Time,
    Velocity,
)
from flow360_schema.framework.unit_system import UnitSystem, UnitSystemConfig
from flow360_schema.framework.validation.context import DeserializationContext, unit_system_manager
from flow360_schema.models.asset_cache import AssetCache
from flow360_schema.models.entities.base import _SurfaceEntityBase, _VolumeEntityBase
from flow360_schema.models.reference_geometry import ReferenceGeometry
from flow360_schema.models.simulation.conversion import (
    LIQUID_IMAGINARY_FREESTREAM_MACH,
    RestrictedUnitSystem,
)
from flow360_schema.models.simulation.framework.boundary_split import (
    BoundaryNameLookupTable,
    post_process_rotation_volume_entities,
    post_process_wall_models_for_rotating,
    update_entities_in_model,
)
from flow360_schema import __version__ as _SCHEMA_PACKAGE_VERSION
from flow360_schema.models.simulation.framework.updater import updater
from flow360_schema.models.simulation.framework.updater_utils import Flow360Version
from flow360_schema.models.simulation.meshing_param.params import (
    MeshingParams,
    ModularMeshingWorkflow,
)
from flow360_schema.models.simulation.meshing_param.volume_params import (
    AutomatedFarfield,
    RotationCylinder,
    RotationSphere,
    RotationVolume,
)
from flow360_schema.models.simulation.models.surface_models import SurfaceModelTypes
from flow360_schema.models.simulation.models.volume_models import (
    ActuatorDisk,
    BETDisk,
    Fluid,
    Solid,
    VolumeModelTypes,
)
from flow360_schema.models.simulation.operating_condition.operating_condition import (
    OperatingConditionTypes,
)
from flow360_schema.models.simulation.outputs.outputs import (
    AeroAcousticOutput,
    ForceDistributionOutput,
    ForceOutput,
    IsosurfaceOutput,
    OutputTypes,
    ProbeOutput,
    SurfaceIntegralOutput,
    SurfaceProbeOutput,
    UserDefinedField,
    VolumeOutput,
)
from flow360_schema.models.simulation.run_control.run_control import RunControl
from flow360_schema.models.simulation.time_stepping.time_stepping import Steady, Unsteady
from flow360_schema.models.simulation.units import validate_length
from flow360_schema.models.simulation.user_code.core.types import (
    get_post_processing_variables,
)
from flow360_schema.models.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360_schema.models.simulation.utils import sanitize_params_dict
from flow360_schema.models.simulation.validation.validation_output import (
    _check_aero_acoustics_observer_time_step_size,
    _check_local_cfl_output,
    _check_moving_statistic_applicability,
    _check_output_fields,
    _check_output_fields_valid_given_transition_model,
    _check_output_fields_valid_given_turbulence_model,
    _check_unique_force_distribution_output_names,
    _check_unique_surface_volume_probe_entity_names,
    _check_unique_surface_volume_probe_names,
    _check_unsteadiness_to_use_aero_acoustics,
)
from flow360_schema.models.simulation.validation.validation_simulation_params import (
    _check_and_add_noninertial_reference_frame_flag,
    _check_cht_solver_settings,
    _check_complete_boundary_condition_and_unknown_surface,
    _check_consistency_hybrid_model_volume_output,
    _check_consistency_wall_function_and_surface_output,
    _check_coordinate_system_constraints,
    _check_duplicate_actuator_disk_cylinder_names,
    _check_duplicate_entities_in_models,
    _check_duplicate_isosurface_names,
    _check_duplicate_surface_usage,
    _check_hybrid_model_to_use_zonal_enforcement,
    _check_krylov_solver_restrictions,
    _check_low_mach_preconditioner_output,
    _check_numerical_dissipation_factor_output,
    _check_parent_volume_is_rotating,
    _check_rotation_entities_have_volume_zone,
    _check_time_average_output,
    _check_tpg_not_with_isentropic_solver,
    _check_unique_selector_names,
    _check_unsteadiness_to_use_hybrid_model,
    _check_valid_models_for_liquid,
    _populate_validated_field_to_validation_context,
)
from flow360_schema.models.simulation.validation.validation_utils import has_mirroring_usage

from .validation.validation_context import (
    CASE,
    SURFACE_MESH,
    VOLUME_MESH,
    CaseField,
    ConditionalField,
    ParamsValidationInfo,
    context_validator,
    contextual_field_validator,
    contextual_model_validator,
)

logger = logging.getLogger(__name__)

__all__ = [
    "_ParamModelBase",
    "ModelTypes",
    "ReferenceGeometry",
    "SimulationParams",
]


def _unit_system_inconsistent_msg(kwarg_unit_system, context_unit_system):
    return f"""\
Tried to construct model with {kwarg_unit_system} inside {context_unit_system} context.
It can be caused by .copy() operation inside unit system context, \
or by providing unit system directly to the constructor.
"""


ModelTypes = Annotated[VolumeModelTypes | SurfaceModelTypes, pd.Field(discriminator="type")]


class _ParamModelBase(Flow360BaseModel):
    """
    Base class that abstracts out all Param type classes in Flow360.
    """

    version: str = pd.Field(_SCHEMA_PACKAGE_VERSION, frozen=True)
    unit_system: UnitSystemConfig = pd.Field(frozen=True)
    model_config = pd.ConfigDict(include_hash=True)

    @classmethod
    def _init_check_unit_system(cls, **kwargs):
        """
        Resolve the unit system from kwargs / active context / SI default.
        Raises if an explicit kwarg unit_system conflicts with the active context.
        Returns (resolved_unit_system, remaining_kwargs).
        """
        if unit_system_manager.current is None:
            raise Flow360ValueError(
                "Please use a unit system context (e.g. `with SI_unit_system:`) "
                "when constructing SimulationParams from Python."
            )

        kwarg_unit_system = kwargs.pop("unit_system", None)
        if kwarg_unit_system is not None:
            if isinstance(kwarg_unit_system, UnitSystemConfig):
                resolved = kwarg_unit_system.resolve()
            elif isinstance(kwarg_unit_system, dict):
                resolved = UnitSystemConfig.model_validate(kwarg_unit_system).resolve()
            elif isinstance(kwarg_unit_system, UnitSystem):
                resolved = kwarg_unit_system
            else:
                raise Flow360ValueError(f"Unexpected unit_system type: {type(kwarg_unit_system)}")
            if resolved != unit_system_manager.current:
                raise Flow360ValueError(
                    _unit_system_inconsistent_msg(
                        resolved.system_repr(),
                        unit_system_manager.current.system_repr(),
                    )
                )
        else:
            resolved = unit_system_manager.current

        return resolved, kwargs

    @classmethod
    def _get_version_from_dict(cls, model_dict: dict) -> str:
        version = model_dict.get("version")
        if version is None:
            raise Flow360ValueError("Failed to find SimulationParams version from the input.")
        return version

    @classmethod
    def _update_param_dict(cls, model_dict, version_to=_SCHEMA_PACKAGE_VERSION):
        """
        1. Find the version from the input dict.
        2. Update the input dict to `version_to` which by default is the current version.
        3. If the simulation.json has higher version, then return the dict as is without modification.

        Returns
        -------
        dict
            The updated parameters dictionary.
        bool
            Whether the `model_dict` has higher version than `version_to`.
        """
        input_version = cls._get_version_from_dict(model_dict=model_dict)
        forward_compatibility_mode = Flow360Version(input_version) > Flow360Version(version_to)
        if not forward_compatibility_mode:
            model_dict = updater(
                version_from=input_version,
                version_to=version_to,
                params_as_dict=model_dict,
            )
        return model_dict, forward_compatibility_mode

    @staticmethod
    def _sanitize_params_dict(model_dict):
        """
        !!!WARNING!!!: This function changes the input dict in place!!!

        Clean the redundant content in the params dict from WebUI
        """
        return sanitize_params_dict(model_dict)

    @classmethod
    def from_file(cls, filename: str):
        """Override to run sanitizer and version updater before validation."""
        model_dict = cls._handle_file(filename=filename)
        model_dict = cls._sanitize_params_dict(model_dict)
        model_dict, _ = cls._update_param_dict(model_dict)
        return cls.deserialize(model_dict)

    def _init_no_unit_context(self, filename, file_content, **kwargs):
        """
        Initialize the simulation parameters from file or dict content.
        """
        if filename is not None:
            model_dict = self._handle_file(filename=filename, **kwargs)
        else:
            model_dict = self._handle_dict(**file_content)

        model_dict = _ParamModelBase._sanitize_params_dict(model_dict)
        model_dict, _ = _ParamModelBase._update_param_dict(model_dict)

        with DeserializationContext():
            super().__init__(**model_dict)

    def _init_with_unit_context(self, **kwargs):
        """
        Initializes the simulation parameters with the given unit context.
        This is the entry when user construct Param with Python script.
        """
        _, kwargs = _ParamModelBase._init_check_unit_system(**kwargs)

        current = unit_system_manager.current
        super().__init__(unit_system=UnitSystemConfig(name=current.name), **kwargs)

    def __init__(self, filename: str = None, file_content: dict = None, **kwargs):
        if filename is not None or file_content is not None:
            self._init_no_unit_context(filename, file_content, **kwargs)
        elif unit_system_manager.current is not None:
            self._init_with_unit_context(**kwargs)
        elif "unit_system" in kwargs:
            with DeserializationContext():
                super().__init__(**kwargs)
        else:
            raise Flow360ValueError(
                "Please use a unit system context (e.g. `with SI_unit_system:`) "
                "when constructing SimulationParams from Python."
            )

    def copy(self, update=None, **kwargs) -> _ParamModelBase:
        if unit_system_manager.current is None:
            with self.unit_system.resolve():
                return super().copy(update=update, **kwargs)

        return super().copy(update=update, **kwargs)


class SimulationParams(_ParamModelBase):
    """All-in-one class for surface meshing + volume meshing + case configurations"""

    meshing: MeshingParams | ModularMeshingWorkflow | None = ConditionalField(
        None,
        context=[SURFACE_MESH, VOLUME_MESH],
        discriminator="type_name",
        description="Surface and volume meshing parameters. See :class:`MeshingParams` for more details.",
    )

    reference_geometry: ReferenceGeometry | None = CaseField(
        None,
        description="Global geometric reference values. See :class:`ReferenceGeometry` for more details.",
    )
    operating_condition: OperatingConditionTypes | None = CaseField(
        None,
        discriminator="type_name",
        description="Global operating condition."
        " See :ref:`Operating Condition <operating_condition>` for more details.",
    )

    models: list[ModelTypes] | None = CaseField(
        None,
        description="Solver settings and numerical models and boundary condition settings."
        " See :ref:`Volume Models <volume_models>` and :ref:`Surface Models <surface_models>` for more details.",
    )
    time_stepping: Steady | Unsteady = CaseField(
        Steady(),
        discriminator="type_name",
        description="Time stepping settings. See :ref:`Time Stepping <timeStepping>` for more details.",
    )
    user_defined_dynamics: list[UserDefinedDynamic] | None = CaseField(
        None,
        description="User defined dynamics. See :ref:`User Defined Dynamics <user_defined_dynamics>` for more details.",
    )

    user_defined_fields: list[UserDefinedField] = CaseField(
        [], description="User defined fields that can be used in outputs."
    )

    outputs: list[OutputTypes] | None = CaseField(
        None,
        description="Output settings. See :ref:`Outputs <outputs>` for more details.",
    )

    run_control: RunControl | None = CaseField(
        None,
        description="Run control settings of the simulation.",
    )

    private_attribute_asset_cache: AssetCache = pd.Field(AssetCache(), frozen=True)
    private_attribute_dict: dict | None = pd.Field(None)

    def _preprocess(self, mesh_unit=None, exclude: list = None) -> SimulationParams:
        """Internal function for non-dimensionalizing the simulation parameters"""
        if exclude is None:
            exclude = []

        if mesh_unit is None:
            raise Flow360ValueError("Mesh unit has not been supplied.")
        self._private_set_length_unit(validate_length(mesh_unit))
        if unit_system_manager.current is None:
            with self.unit_system.resolve():
                return super().preprocess(
                    params=self,
                    exclude=exclude,
                    flow360_unit_system=self.flow360_unit_system,
                )
        return super().preprocess(params=self, exclude=exclude, flow360_unit_system=self.flow360_unit_system)

    def _private_set_length_unit(self, validated_mesh_unit):
        self.private_attribute_asset_cache._force_set_attr("project_length_unit", validated_mesh_unit)

    @pd.validate_call
    def convert_unit(
        self,
        value,
        target_system: Literal["SI", "Imperial", "flow360"],
        length_unit: Length.Float64 | None = None,
    ):
        """
        Converts a given value to the specified unit system.

        This method takes a dimensioned quantity and converts it from its current unit system
        to the target unit system, optionally considering a specific length unit for the conversion.

        Parameters
        ----------
        value
            The dimensioned quantity to convert.
        target_system : str
            The target unit system for conversion.
        length_unit : Length.Float64, optional
            The length unit to use for conversion.

        Returns
        -------
        object
            The converted value in the specified target unit system.

        Raises
        ------
        Flow360ValueError
            If the input unit system is not compatible with the target system, or if the required
            length unit is missing.
        """

        if length_unit is not None:
            self._private_set_length_unit(validate_length(length_unit))

        if target_system in ("flow360", "flow360_v2"):
            return value.in_base(unit_system=self.flow360_unit_system)
        return value.in_base(unit_system=target_system)

    @pd.field_validator("models", mode="after")
    @classmethod
    def apply_default_fluid_settings(cls, value):
        """Apply default Fluid() settings if not found in models."""
        if value is None:
            value = []
        assert isinstance(value, list)
        if not any(isinstance(item, Fluid) for item in value):
            value.append(Fluid(private_attribute_id="__default_fluid"))
        return value

    @contextual_field_validator("models", mode="after")
    @classmethod
    def check_parent_volume_is_rotating(cls, models, param_info: ParamsValidationInfo):
        """Ensure that all the parent volumes listed in the `Rotation` model are not static"""
        return _check_parent_volume_is_rotating(models, param_info)

    @contextual_field_validator("models", mode="after")
    @classmethod
    def check_valid_models_for_liquid(cls, models, param_info: ParamsValidationInfo):
        """Ensure that all the boundary conditions used are valid."""
        return _check_valid_models_for_liquid(models, param_info)

    @contextual_field_validator("models", mode="after")
    @classmethod
    def check_duplicate_actuator_disk_cylinder_names(cls, models, param_info: ParamsValidationInfo):
        """Ensure that all the cylinder names used in ActuatorDisks are unique."""
        return _check_duplicate_actuator_disk_cylinder_names(models, param_info)

    @contextual_field_validator("models", mode="after")
    @classmethod
    def populate_validated_models_to_validation_context(cls, models, param_info: ParamsValidationInfo):
        """After models are validated, store {id: model_obj} in validation context."""
        return _populate_validated_field_to_validation_context(models, param_info, "physics_model_dict")

    @contextual_field_validator("user_defined_fields", mode="after")
    @classmethod
    def _disable_expression_for_liquid(
        cls,
        value,
        info: pd.ValidationInfo,
        param_info: ParamsValidationInfo,
    ):
        """Ensure that string expressions are disabled for liquid simulation."""
        if param_info.using_liquid_as_material is False:
            return value
        if value:
            raise ValueError(f"{info.field_name} cannot be used when using liquid as simulation material.")
        return value

    @pd.field_validator("outputs", mode="after")
    @classmethod
    def check_duplicate_isosurface_names(cls, outputs):
        """Check if we have isosurfaces with a duplicate name"""
        return _check_duplicate_isosurface_names(outputs)

    @contextual_field_validator("outputs", mode="after")
    @classmethod
    def check_duplicate_surface_usage(cls, outputs, param_info: ParamsValidationInfo):
        """Disallow the same boundary/surface being used in multiple outputs"""
        return _check_duplicate_surface_usage(outputs, param_info)

    @contextual_field_validator("outputs", mode="after")
    @classmethod
    def populate_validated_outputs_to_validation_context(cls, outputs, param_info: ParamsValidationInfo):
        """After outputs are validated, store {id: output_obj} in validation context."""
        return _populate_validated_field_to_validation_context(outputs, param_info, "output_dict")

    @pd.field_validator("user_defined_fields", mode="after")
    @classmethod
    def check_duplicate_user_defined_fields(cls, value):
        """Check if we have duplicate user defined fields"""
        if value == []:
            return value

        known_user_defined_fields = set()
        for field in value:
            if field.name in known_user_defined_fields:
                raise ValueError(f"Duplicate user defined field name: {field.name}")
            known_user_defined_fields.add(field.name)

        return value

    @pd.model_validator(mode="after")
    def check_cht_solver_settings(self):
        """Check the Conjugate Heat Transfer settings."""
        return _check_cht_solver_settings(self)

    @pd.model_validator(mode="after")
    def check_consistency_wall_function_and_surface_output(self):
        """Only allow wallFunctionMetric output field when there is a Wall model with a wall function enabled"""
        return _check_consistency_wall_function_and_surface_output(self)

    @pd.model_validator(mode="after")
    def check_consistency_hybrid_model_volume_output(self):
        """Only allow hybrid RANS-LES output field when there is a corresponding solver with
        hybrid RANS-LES enabled in models
        """
        return _check_consistency_hybrid_model_volume_output(self)

    @pd.model_validator(mode="after")
    def check_unsteadiness_to_use_hybrid_model(self):
        """Only allow hybrid RANS-LES output field for unsteady simulations"""
        return _check_unsteadiness_to_use_hybrid_model(self)

    @pd.model_validator(mode="after")
    def check_hybrid_model_to_use_zonal_enforcement(self):
        """Only allow LES/RANS zonal enforcement in hybrid RANS-LES mode"""
        return _check_hybrid_model_to_use_zonal_enforcement(self)

    @pd.model_validator(mode="after")
    def check_unsteadiness_to_use_aero_acoustics(self):
        """Only allow Aero acoustics when using unsteady simulation"""
        return _check_unsteadiness_to_use_aero_acoustics(self)

    @pd.model_validator(mode="after")
    def check_local_cfl_output(self):
        """Only allow localCFL output when using unsteady simulation"""
        return _check_local_cfl_output(self)

    @pd.model_validator(mode="after")
    def check_aero_acoustics_observer_time_step_size(self):
        """Validate that observer time step size is smaller than CFD time step size"""
        return _check_aero_acoustics_observer_time_step_size(self)

    @pd.model_validator(mode="after")
    def check_unique_surface_volume_probe_names(self):
        """Only allow unique probe names"""
        return _check_unique_surface_volume_probe_names(self)

    @contextual_model_validator(mode="after")
    def check_unique_surface_volume_probe_entity_names(self):
        """Only allow unique probe entity names"""
        return _check_unique_surface_volume_probe_entity_names(self)

    @pd.model_validator(mode="after")
    def check_unique_force_distribution_output_names(self):
        """Only allow unique force distribution names"""
        return _check_unique_force_distribution_output_names(self)

    @contextual_model_validator(mode="after")
    def check_duplicate_entities_in_models(self, param_info: ParamsValidationInfo):
        """Only allow each Surface/Volume entity to appear once in the Surface/Volume model"""
        return _check_duplicate_entities_in_models(self, param_info)

    @contextual_model_validator(mode="after")
    def check_unique_selector_names(self):
        """Ensure all EntitySelector names are unique"""
        return _check_unique_selector_names(self)

    @pd.model_validator(mode="after")
    def check_numerical_dissipation_factor_output(self):
        """Only allow numericalDissipationFactor output field when the NS solver has low numerical dissipation"""
        return _check_numerical_dissipation_factor_output(self)

    @pd.model_validator(mode="after")
    def check_low_mach_preconditioner_output(self):
        """Only allow lowMachPreconditioner output field when the lowMachPreconditioner is enabled in the NS solver"""
        return _check_low_mach_preconditioner_output(self)

    @pd.model_validator(mode="after")
    def check_tpg_not_with_isentropic_solver(self):
        """Temperature-dependent gas properties are not supported with CompressibleIsentropic (4x4) solver."""
        return _check_tpg_not_with_isentropic_solver(self)

    @pd.model_validator(mode="after")
    def check_krylov_solver_restrictions(self):
        """Krylov solver is not compatible with limiters or unsteady time stepping."""
        return _check_krylov_solver_restrictions(self)

    @contextual_model_validator(mode="after")
    @context_validator(context=CASE)
    def check_complete_boundary_condition_and_unknown_surface(self, param_info: ParamsValidationInfo):
        """Make sure that all boundaries have been assigned with a boundary condition"""
        return _check_complete_boundary_condition_and_unknown_surface(self, param_info)

    @pd.model_validator(mode="after")
    def check_output_fields(params):
        """Check output fields and iso fields are valid"""
        return _check_output_fields(params)

    @pd.model_validator(mode="after")
    def check_output_fields_valid_given_turbulence_model(params):
        """Check output fields are valid given the turbulence model"""
        return _check_output_fields_valid_given_turbulence_model(params)

    @pd.model_validator(mode="after")
    def check_output_fields_valid_given_transition_model(params):
        """Check output fields are valid given the transition model"""
        return _check_output_fields_valid_given_transition_model(params)

    @pd.model_validator(mode="after")
    def check_and_add_rotating_reference_frame_model_flag_in_volumezones(params):
        """Ensure that all volume zones have the rotating_reference_frame_model flag with correct values"""
        return _check_and_add_noninertial_reference_frame_flag(params)

    @contextual_model_validator(mode="after")
    @context_validator(context=CASE)
    def check_rotation_entities_have_volume_zone(self, param_info: ParamsValidationInfo):
        """For geometry/surface-mesh workflows, every Rotation entity must have a matching volume zone."""
        return _check_rotation_entities_have_volume_zone(self, param_info)

    @pd.model_validator(mode="after")
    def check_time_average_output(params):
        """Only allow TimeAverage output field in the unsteady simulations"""
        return _check_time_average_output(params)

    @pd.model_validator(mode="after")
    def check_moving_statistic_applicability(params):
        """Check moving statistic settings are applicable to the simulation time stepping set up."""
        return _check_moving_statistic_applicability(params)

    @contextual_model_validator(mode="after")
    def _validate_coordinate_system_constraints(self, param_info: ParamsValidationInfo):
        """Validate coordinate system usage constraints."""
        return _check_coordinate_system_constraints(self, param_info)

    @contextual_model_validator(mode="after")
    def _validate_mirroring_requires_geometry_ai(self, param_info: ParamsValidationInfo):
        """Ensure mirroring is only used when GeometryAI is enabled."""
        if has_mirroring_usage(self.private_attribute_asset_cache):
            if not param_info.use_geometry_AI:
                raise ValueError("Mirroring is only supported when Geometry AI is enabled.")
        return self

    def _register_assigned_entities(self, registry: EntityRegistry) -> EntityRegistry:
        """Recursively register all entities listed in EntityList to the asset cache."""
        registry.clear()
        register_entity_list(self, registry)
        return registry

    def _update_entity_private_attrs(self, registry: EntityRegistry) -> EntityRegistry:
        """
        Once the SimulationParams is set, extract and update information
        into all used entities by parsing the params.
        """
        if self.meshing is not None:
            volume_zones = None
            if isinstance(self.meshing, MeshingParams):
                volume_zones = self.meshing.volume_zones
            if isinstance(self.meshing, ModularMeshingWorkflow) and self.meshing.volume_meshing is not None:
                volume_zones = self.meshing.zones
            if volume_zones is not None:
                for volume in volume_zones:
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
                    if isinstance(volume, (RotationCylinder, RotationVolume, RotationSphere)):
                        pass

        return registry

    @property
    def base_length(self) -> Length.Float64:
        """Get base length unit for non-dimensionalization"""
        return self.private_attribute_asset_cache.project_length_unit.to("m")

    @property
    def base_temperature(self) -> AbsoluteTemperature.Float64:
        """Get base temperature unit for non-dimensionalization"""
        if self.operating_condition.type_name == "LiquidOperatingCondition":
            return 273 * u.K
        return self.operating_condition.thermal_state.temperature.to("K")

    @property
    def base_velocity(self) -> Velocity.Float64:
        """Get base velocity unit for non-dimensionalization"""
        if self.operating_condition.type_name == "LiquidOperatingCondition":
            if self.operating_condition._evaluated_velocity_magnitude.value != 0:
                return (self.operating_condition._evaluated_velocity_magnitude / LIQUID_IMAGINARY_FREESTREAM_MACH).to(
                    "m/s"
                )
            return (self.operating_condition.reference_velocity_magnitude / LIQUID_IMAGINARY_FREESTREAM_MACH).to("m/s")
        return self.operating_condition.thermal_state.speed_of_sound.to("m/s")

    @property
    def reference_velocity(self) -> Velocity.Float64:
        """
        This function returns the **reference velocity**.
        Note that the reference velocity is **NOT** the non-dimensionalization velocity scale.
        """
        reference_velocity_magnitude = getattr(self.operating_condition, "reference_velocity_magnitude", None)
        if reference_velocity_magnitude is not None:
            reference_velocity = reference_velocity_magnitude.to("m/s")
        elif self.operating_condition.type_name == "LiquidOperatingCondition":
            reference_velocity = self.base_velocity.to("m/s") * LIQUID_IMAGINARY_FREESTREAM_MACH
        else:
            reference_velocity = self.operating_condition.velocity_magnitude.to("m/s")
        return reference_velocity

    @property
    def base_density(self) -> Density.Float64:
        """Get base density unit for non-dimensionalization"""
        if self.operating_condition.type_name == "LiquidOperatingCondition":
            return self.operating_condition.material.density.to("kg/m**3")
        return self.operating_condition.thermal_state.density.to("kg/m**3")

    @property
    def base_mass(self) -> Mass.Float64:
        """Get base mass unit for non-dimensionalization"""
        return self.base_density * self.base_length**3

    @property
    def base_time(self) -> Time.Float64:
        """Get base time unit for non-dimensionalization"""
        return self.base_length / self.base_velocity

    @property
    def flow360_unit_system(self) -> u.UnitSystem:
        """Get the unit system for non-dimensionalization."""
        if self.operating_condition is None:
            return RestrictedUnitSystem("flow360_nondim", length_unit=self.base_length)
        return RestrictedUnitSystem(
            "flow360_nondim",
            length_unit=self.base_length,
            mass_unit=self.base_mass,
            time_unit=self.base_time,
            temperature_unit=self.base_temperature,
        )

    @property
    def used_entity_registry(self) -> EntityRegistry:
        """
        Get an entity registry that collects all the entities used in the simulation.
        """
        registry = EntityRegistry()
        registry = self._register_assigned_entities(registry)
        registry = self._update_entity_private_attrs(registry)
        return registry

    def _update_param_with_actual_volume_mesh_meta(self, volume_mesh_meta_data: dict):
        """
        Update the zone info from the actual volume mesh before solver execution.
        """
        lookup_table = BoundaryNameLookupTable.from_params(volume_mesh_meta_data, params=self)

        update_entities_in_model(self, lookup_table, _SurfaceEntityBase)

        _update_entity_full_name(self, _VolumeEntityBase, volume_mesh_meta_data)
        _update_zone_boundaries_with_metadata(self.used_entity_registry, volume_mesh_meta_data)

        post_process_rotation_volume_entities(self, lookup_table)
        post_process_wall_models_for_rotating(self, lookup_table)

        return self

    def is_steady(self):
        """
        returns True when SimulationParams is steady state
        """
        return isinstance(self.time_stepping, Steady)

    def has_solid(self):
        """
        returns True when SimulationParams has Solid model
        """
        if self.models is None:
            return False
        return any(isinstance(item, Solid) for item in self.models)

    def has_actuator_disks(self):
        """
        returns True when SimulationParams has ActuatorDisk disk
        """
        if self.models is None:
            return False
        return any(isinstance(item, ActuatorDisk) for item in self.models)

    def has_bet_disks(self):
        """
        returns True when SimulationParams has BET disk
        """
        if self.models is None:
            return False
        return any(isinstance(item, BETDisk) for item in self.models)

    def has_isosurfaces(self):
        """
        returns True when SimulationParams has isosurfaces
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, IsosurfaceOutput) for item in self.outputs)

    def has_monitors(self):
        """
        returns True when SimulationParams has monitors
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, (ProbeOutput, SurfaceProbeOutput, SurfaceIntegralOutput)) for item in self.outputs)

    def has_volume_output(self):
        """
        returns True when SimulationParams has volume output
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, VolumeOutput) for item in self.outputs)

    def has_aeroacoustics(self):
        """
        returns True when SimulationParams has aeroacoustics
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, AeroAcousticOutput) for item in self.outputs)

    def has_user_defined_dynamics(self):
        """
        returns True when SimulationParams has user defined dynamics
        """
        return self.user_defined_dynamics is not None and len(self.user_defined_dynamics) > 0

    def has_force_distributions(self):
        """
        returns True when SimulationParams has force distributions
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, ForceDistributionOutput) for item in self.outputs)

    def has_custom_forces(self):
        """
        returns True when SimulationParams has any ForceOutputs
        """
        if self.outputs is None:
            return False
        return any(isinstance(item, ForceOutput) for item in self.outputs)

    def display_output_units(self) -> None:
        """
        Display all the output units for UserVariables used in `outputs`.
        """
        if not self.outputs:
            return

        post_processing_variables = get_post_processing_variables(self)

        post_processing_variables = sorted(post_processing_variables)
        name_units_pair = batch_get_user_variable_units(post_processing_variables, self.unit_system.name)

        for output in self.outputs:
            if isinstance(output, SurfaceIntegralOutput):
                for field in output.output_fields.items:
                    if isinstance(field, UserVariable):
                        unit = compute_surface_integral_unit(
                            field,
                            unit_system_name=self.unit_system.name,
                            unit_system=self.unit_system.resolve(),
                        )
                        name_units_pair[f"{field.name} (Surface integral)"] = unit

        if not name_units_pair:
            return

        name_column_width = max(len("Variable Name"), max(len(name) for name in name_units_pair))
        unit_column_width = max(len("Unit"), max(len(str(unit)) for unit in name_units_pair.values()))

        name_column_width = max(name_column_width, 15)
        unit_column_width = max(unit_column_width, 10)

        header = f"{'Variable Name':<{name_column_width}} | {'Unit':<{unit_column_width}}"
        separator = "-" * len(header)

        logger.info("")
        logger.info("Units of output `UserVariables`:")
        logger.info(separator)
        logger.info(header)
        logger.info(separator)

        for name, unit in name_units_pair.items():
            logger.info(f"{name:<{name_column_width}} | {str(unit):<{unit_column_width}}")

        logger.info(separator)
        logger.info("")

    def pre_submit_summary(self):
        """
        Display a summary of the simulation params before submission.
        """
        self.display_output_units()
