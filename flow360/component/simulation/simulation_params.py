"""
Flow360 simulation parameters
"""

from __future__ import annotations

from typing import Annotated, List, Optional, Union

import pydantic as pd

from flow360.component.simulation.conversion import unit_converter
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
from flow360.component.simulation.framework.updater_utils import Flow360Version
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
    DimensionedTypes,
    LengthType,
    UnitSystem,
    UnitSystemType,
    is_flow360_unit,
    unit_system_manager,
    unyt_quantity,
)
from flow360.component.simulation.user_defined_dynamics.user_defined_dynamics import (
    UserDefinedDynamic,
)
from flow360.component.simulation.utils import model_attribute_unlock
from flow360.component.simulation.validation.validation_output import (
    _check_output_fields,
    _check_output_fields_valid_given_turbulence_model,
    _check_unsteadiness_to_use_aero_acoustics,
)
from flow360.component.simulation.validation.validation_simulation_params import (
    _check_and_add_noninertial_reference_frame_flag,
    _check_cht_solver_settings,
    _check_complete_boundary_condition_and_unknown_surface,
    _check_consistency_hybrid_model_volume_output,
    _check_consistency_wall_function_and_surface_output,
    _check_duplicate_entities_in_models,
    _check_duplicate_isosurface_names,
    _check_low_mach_preconditioner_output,
    _check_numerical_dissipation_factor_output,
    _check_parent_volume_is_rotating,
    _check_time_average_output,
    _check_unsteadiness_to_use_hybrid_model,
    _check_valid_models_for_liquid,
)
from flow360.component.utils import remove_properties_by_name
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
    get_validation_info,
)

ModelTypes = Annotated[Union[VolumeModelTypes, SurfaceModelTypes], pd.Field(discriminator="type")]


class _ParamModelBase(Flow360BaseModel):
    """
    Base class that abstracts out all Param type classes in Flow360.
    """

    version: str = pd.Field(__version__, frozen=True)
    unit_system: UnitSystemType = pd.Field(frozen=True, discriminator="name")
    model_config = pd.ConfigDict(include_hash=True)

    @classmethod
    def _init_check_unit_system(cls, **kwargs):
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

    @classmethod
    def _get_version_from_dict(cls, model_dict: dict) -> str:
        version = model_dict.get("version", None)
        if version is None:
            raise Flow360RuntimeError("Failed to find SimulationParams version from the input.")
        return version

    @classmethod
    def _update_param_dict(cls, model_dict, version_to=__version__):
        """
        1. Find the version from the input dict.
        2. Update the input dict to `version_to` which by default is the current version.
        3. If the simulation.json has higher version, then return the dict as is without modification.

        Returns
        -------
        dict
            The updated parameters dictionary.
        bool
            Whether the `model_dict` has higher version than `version_to` (AKA forward compatibility mode).
        """
        input_version = cls._get_version_from_dict(model_dict=model_dict)
        forward_compatibility_mode = Flow360Version(input_version) > Flow360Version(version_to)
        if not forward_compatibility_mode:
            model_dict = updater(
                version_from=input_version, version_to=version_to, params_as_dict=model_dict
            )
        return model_dict, forward_compatibility_mode

    @classmethod
    def _sanitize_params_dict(cls, model_dict):
        """
        Clean the redundant content in the params dict from WebUI
        """
        model_dict = remove_properties_by_name(model_dict, "_id")
        return model_dict

    def _init_no_unit_context(self, filename, file_content, **kwargs):
        """
        Initialize the simulation parameters without a unit context.
        """
        if unit_system_manager.current is not None:
            raise Flow360RuntimeError(
                f"When loading params from file: {self.__class__.__name__}(filename), "
                "unit context must not be used."
            )

        if filename is not None:
            model_dict = self._handle_file(filename=filename, **kwargs)
        else:
            model_dict = self._handle_dict(**file_content)

        model_dict = _ParamModelBase._sanitize_params_dict(model_dict)
        # When treating files/file like contents the updater will always be run.
        model_dict, _ = _ParamModelBase._update_param_dict(model_dict)

        unit_system = model_dict.get("unit_system")

        with UnitSystem.from_dict(**unit_system):  # pylint: disable=not-context-manager
            super().__init__(**model_dict)

    def _init_with_unit_context(self, **kwargs):
        """
        Initializes the simulation parameters with the given unit context.
        """
        # When treating dicts the updater is skipped.
        kwargs = _ParamModelBase._init_check_unit_system(**kwargs)
        super().__init__(unit_system=unit_system_manager.current, **kwargs)

    # pylint: disable=super-init-not-called
    # pylint: disable=fixme
    # TODO: avoid overloading the __init__ so IDE can proper prompt root level keys
    def __init__(self, filename: str = None, file_content: dict = None, **kwargs):
        if filename is not None or file_content is not None:
            self._init_no_unit_context(filename, file_content, **kwargs)
        else:
            self._init_with_unit_context(**kwargs)

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
    private_attribute_dict: Optional[dict] = pd.Field(None)

    # pylint: disable=arguments-differ
    def _preprocess(self, mesh_unit=None, exclude: list = None) -> SimulationParams:
        """Internal function for non-dimensionalizing the simulation parameters"""
        if exclude is None:
            exclude = []

        if mesh_unit is None:
            raise Flow360ConfigurationError("Mesh unit has not been supplied.")
        self._private_set_length_unit(LengthType.validate(mesh_unit))  # pylint: disable=no-member
        if unit_system_manager.current is None:
            # pylint: disable=not-context-manager
            with self.unit_system:
                return super().preprocess(params=self, exclude=exclude)
        return super().preprocess(params=self, exclude=exclude)

    def _private_set_length_unit(self, validated_mesh_unit):
        with model_attribute_unlock(self.private_attribute_asset_cache, "project_length_unit"):
            # pylint: disable=assigning-non-slot
            self.private_attribute_asset_cache.project_length_unit = validated_mesh_unit

    @pd.validate_call
    def convert_unit(
        self, value: DimensionedTypes, target_system: str, length_unit: Optional[LengthType] = None
    ):
        """
        Converts a given value to the specified unit system.

        This method takes a dimensioned quantity and converts it from its current unit system
        to the target unit system, optionally considering a specific length unit for the conversion.

        Parameters
        ----------
        value : DimensionedTypes
            The dimensioned quantity to convert. This should have units compatible with Flow360's
            unit system.
        target_system : str
            The target unit system for conversion. Common values include "SI", "Imperial", flow360".
        length_unit : LengthType, optional
            The length unit to use for conversion. If not provided, the method defaults to
            the project length unit stored in the `private_attribute_asset_cache`.

        Returns
        -------
        DimensionedTypes
            The converted value in the specified target unit system.

        Raises
        ------
        Flow360RuntimeError
            If the input unit system is not compatible with the target system, or if the required
            length unit is missing.

        Examples
        --------
        Convert a value from the current system to Flow360's V2 unit system:

        >>> simulation_params = SimulationParams()
        >>> value = unyt_quantity(1.0, "meters")
        >>> converted_value = simulation_params.convert_unit(value, target_system="flow360")
        >>> print(converted_value)
        1.0 (flow360_length_unit)
        """

        if length_unit is not None:
            # pylint: disable=no-member
            self._private_set_length_unit(LengthType.validate(length_unit))

        flow360_conv_system = unit_converter(
            value.units.dimensions,
            params=self,
            required_by=[f"{self.__class__.__name__}.convert_unit(value=, target_system=)"],
        )

        if target_system == "flow360":
            target_system = "flow360_v2"

        if is_flow360_unit(value) and not isinstance(value, unyt_quantity):
            converted = value.in_base(target_system, flow360_conv_system)
        else:
            value.units.registry = flow360_conv_system.registry  # pylint: disable=no-member
            converted = value.in_base(unit_system=target_system)
        return converted

    # pylint: disable=no-self-argument
    @pd.field_validator("models", mode="after")
    @classmethod
    def apply_default_fluid_settings(cls, v):
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

    @pd.field_validator("models", mode="after")
    @classmethod
    def check_valid_models_for_liquid(cls, models):
        """Ensure that all the boundary conditions used are valid."""
        return _check_valid_models_for_liquid(models)

    @pd.field_validator("user_defined_dynamics", "user_defined_fields", mode="after")
    @classmethod
    def _disable_expression_for_liquid(cls, value, info: pd.ValidationInfo):
        """Ensure that string expressions are disabled for liquid simulation."""
        validation_info = get_validation_info()
        if validation_info is None or validation_info.using_liquid_as_material is False:
            return value
        if value:
            raise ValueError(
                f"{info.field_name} cannot be used when using liquid as simulation material."
            )
        return value

    @pd.field_validator("outputs", mode="after")
    @classmethod
    def check_duplicate_isosurface_names(cls, outputs):
        """Check if we have isosurfaces with a duplicate name"""
        return _check_duplicate_isosurface_names(outputs)

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
    def check_unsteadiness_to_use_aero_acoustics(self):
        """Only allow Aero acoustics when using unsteady simulation"""
        return _check_unsteadiness_to_use_aero_acoustics(self)

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

    @pd.model_validator(mode="after")
    def check_output_fields_valid_given_turbulence_model(params):
        """Check output fields are valid given the turbulence model"""
        return _check_output_fields_valid_given_turbulence_model(params)

    @pd.model_validator(mode="after")
    def check_and_add_rotating_reference_frame_model_flag_in_volumezones(params):
        """Ensure that all volume zones have the rotating_reference_frame_model flag with correct values"""
        return _check_and_add_noninertial_reference_frame_flag(params)

    @pd.model_validator(mode="after")
    def check_time_average_output(params):
        """Only allow TimeAverage output field in the unsteady simulations"""
        return _check_time_average_output(params)

    def _register_assigned_entities(self, registry: EntityRegistry) -> EntityRegistry:
        """Recursively register all entities listed in EntityList to the asset cache."""
        # pylint: disable=no-member
        registry.clear()
        register_entity_list(self, registry)
        return registry

    def _update_entity_private_attrs(self, registry: EntityRegistry) -> EntityRegistry:
        """
        Once the SimulationParams is set, extract and update information
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
        registry = self._register_assigned_entities(registry)
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
        # Below includes the Ghost entities.
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
