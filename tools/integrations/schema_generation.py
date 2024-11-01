import json
import os
from typing import List, Literal, Optional, Type, Union, get_args

import pydantic.v1 as pd

import flow360.component.v1.modules as fl
from flow360.component.v1.flow360_params import (
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
)
from flow360.component.v1.initial_condition import (
    ExpressionInitialCondition,
    ModifiedRestartSolution,
)
from flow360.component.v1.params_base import Flow360BaseModel, Flow360SortableBaseModel
from flow360.component.v1.unit_system import DensityType, PressureType, TemperatureType
from flow360.component.v1.volume_zones import (
    ReferenceFrame,
    ReferenceFrameDynamic,
    ReferenceFrameExpression,
    VolumeZoneBase,
)

here = os.path.dirname(os.path.abspath(__file__))
version_postfix = "develop"


def write_to_file(name, content):
    with open(name, "w") as outfile:
        outfile.write(content)


def write_schemas(type_obj: Type[Flow360BaseModel], folder_name):
    data = type_obj.flow360_schema()
    schema = json.dumps(data, indent=2)
    name = type_obj.__name__
    if name.startswith("_"):
        name = name[1:]
    if not os.path.exists(os.path.join(here, "data", folder_name)):
        os.mkdir(os.path.join(here, "data", folder_name))
    write_to_file(
        os.path.join(here, "data", folder_name, f"json-schema-{version_postfix}.json"), schema
    )
    ui_schema = json.dumps(type_obj.flow360_ui_schema(), indent=2, sort_keys=True)
    if ui_schema is not None:
        write_to_file(
            os.path.join(here, "data", folder_name, f"ui-schema-{version_postfix}.json"), ui_schema
        )


if not os.path.exists(os.path.join(here, "data")):
    os.mkdir(os.path.join(here, "data"))


class _Freestream(Flow360BaseModel):
    freestream: Union[
        fl.FreestreamFromVelocity,
        fl.FreestreamFromMach,
        fl.FreestreamFromMachReynolds,
        fl.ZeroFreestream,
        fl.ZeroFreestreamFromVelocity,
    ] = pd.Field(
        options=[
            "Freestream from velocity",
            "Freestream from Mach number",
            "Freestream from Mach number and Reynolds number",
            "Zero freestream with reference Mach number",
            "Zero freestream with reference velocity",
        ]
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["*", "turbulentViscosityRatio"]
        field_properties = {
            "velocity": ("field", "unitInput"),
            "velocityRef": ("field", "unitInput"),
        }
        root_property = "properties/freestream/anyOf"
        optional_objects = ["anyOf/properties/turbulenceQuantities"]


class _TurbulenceModelSolver(Flow360BaseModel):
    solver: Union[fl.SpalartAllmaras, fl.KOmegaSST, fl.NoneSolver] = pd.Field(
        options=["Spalart-Allmaras", "kOmegaSST", "None"]
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["modelType", "*", "linearSolver"]
        optional_objects = ["anyOf/properties/linearSolver"]
        exclude_fields = ["anyOf/properties/linearSolver/default"]
        root_property = "properties/solver/anyOf"


class _TimeStepping(Flow360BaseModel):
    class _UnsteadyTimeStepping(fl.UnsteadyTimeStepping):
        class RampCFLUnsteady(fl.RampCFL):
            initial: Optional[pd.PositiveFloat] = pd.Field(
                default=fl.RampCFL.default_unsteady().initial
            )
            final: Optional[pd.PositiveFloat] = pd.Field(
                default=fl.RampCFL.default_unsteady().final
            )
            ramp_steps: Optional[int] = pd.Field(
                alias="rampSteps", default=fl.RampCFL.default_unsteady().ramp_steps
            )

        class AdaptiveCFLUnsteady(fl.AdaptiveCFL):
            max: Optional[pd.PositiveFloat] = pd.Field(
                default=fl.AdaptiveCFL.default_unsteady().max
            )
            convergence_limiting_factor: Optional[pd.PositiveFloat] = pd.Field(
                alias="convergenceLimitingFactor",
                default=fl.AdaptiveCFL.default_unsteady().convergence_limiting_factor,
            )
            max_relative_change: Optional[pd.PositiveFloat] = pd.Field(
                alias="maxRelativeChange",
                default=fl.AdaptiveCFL.default_unsteady().max_relative_change,
            )

        CFL: Optional[Union[RampCFLUnsteady, AdaptiveCFLUnsteady]] = pd.Field(
            displayed="CFL", options=["Ramp CFL", "Adaptive CFL"]
        )

    class _SteadyTimeStepping(fl.SteadyTimeStepping):
        class RampCFLSteady(fl.RampCFL):
            initial: Optional[pd.PositiveFloat] = pd.Field(
                default=fl.RampCFL.default_steady().initial
            )
            final: Optional[pd.PositiveFloat] = pd.Field(default=fl.RampCFL.default_steady().final)
            ramp_steps: Optional[int] = pd.Field(
                alias="rampSteps", default=fl.RampCFL.default_steady().ramp_steps
            )

        class AdaptiveCFLSteady(fl.AdaptiveCFL):
            max: Optional[pd.PositiveFloat] = pd.Field(default=fl.AdaptiveCFL.default_steady().max)
            convergence_limiting_factor: Optional[pd.PositiveFloat] = pd.Field(
                alias="convergenceLimitingFactor",
                default=fl.AdaptiveCFL.default_steady().convergence_limiting_factor,
            )
            max_relative_change: Optional[pd.PositiveFloat] = pd.Field(
                alias="maxRelativeChange",
                default=fl.AdaptiveCFL.default_steady().max_relative_change,
            )

        CFL: Optional[Union[RampCFLSteady, AdaptiveCFLSteady]] = pd.Field(
            displayed="CFL", options=["Ramp CFL", "Adaptive CFL"]
        )

    time_stepping: Union[_SteadyTimeStepping, _UnsteadyTimeStepping] = pd.Field(
        alias="timeStepping", options=["Steady", "Unsteady"]
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["*", "CFL"]
        field_properties = {
            "timeStepSize": ("field", "unitInput"),
        }
        root_property = "properties/timeStepping/anyOf"
        exclude_fields = [
            root_property + "/properties/CFL/default",
        ]


class _AirDensityTemperature(fl.AirDensityTemperature):
    density: DensityType.Positive = pd.Field(default={"value": 1.225, "units": "kg/m**3"})
    temperature: TemperatureType = pd.Field(default={"value": 288.15, "units": "K"})


class _AirPressureTemperature(fl.AirPressureTemperature):
    pressure: PressureType.Positive = pd.Field(default={"value": 101325, "units": "Pa"})


class _FluidProperties(Flow360BaseModel):
    fluid_properties: Union[_AirDensityTemperature, _AirPressureTemperature] = pd.Field(
        alias="fluidProperties",
        options=["From density and temperature", "From pressure and temperature"],
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["modelType", "temperature", "*"]
        field_properties = {
            "temperature": ("field", "unitInput"),
            "density": ("field", "unitInput"),
            "pressure": ("field", "unitInput"),
        }
        root_property = "properties/fluidProperties/anyOf"


class _InitialConditions(Flow360BaseModel):
    initial_conditions: Union[ModifiedRestartSolution, ExpressionInitialCondition] = pd.Field(
        alias="initialConditions", options=["ModifyRestart", "Expression"]
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        root_property = "properties/initialConditions/anyOf"


class _FluidDynamicsVolumeZone(VolumeZoneBase):
    """FluidDynamicsVolumeZone type"""

    model_type = pd.Field("FluidDynamics", alias="modelType", const=True)
    reference_frame: Optional[
        Union[
            ReferenceFrame,
            ReferenceFrameExpression,
            ReferenceFrameDynamic,
        ]
    ] = pd.Field(alias="referenceFrame")

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        optional_objects = ["properties/referenceFrame"]


class _GenericVolumeZonesWrapper(Flow360BaseModel):
    v: Union[_FluidDynamicsVolumeZone, fl.HeatTransferVolumeZone]


class _VolumeZones(Flow360SortableBaseModel):
    @classmethod
    def get_subtypes(cls) -> list:
        return list(get_args(_GenericVolumeZonesWrapper.__fields__["v"].type_))

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {
            "additionalProperties/referenceFrame/centerOfRotation": ("field", "unitInput"),
            "additionalProperties/referenceFrame/omega": ("field", "unitInput"),
            "additionalProperties/referenceFrame/axisOfRotation": ("widget", "vector3"),
        }


class _Geometry(fl.Geometry):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {
            "refArea": ("field", "unitInput"),
            "momentCenter": ("field", "unitInput"),
            "momentLength": ("field", "unitInput"),
            "meshUnit": ("field", "unitInput"),
        }


class _NavierStokesSolver(fl.NavierStokesSolver):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["*", "linearSolver"]
        optional_objects = ["properties/linearSolver"]
        exclude_fields = ["properties/linearSolver/default"]


class _TransitionModelSolver(fl.TransitionModelSolver):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["modelType", "*", "linearSolver"]
        optional_objects = ["properties/linearSolver"]
        exclude_fields = ["properties/linearSolver/default"]


class _HeatEquationSolver(fl.HeatEquationSolver):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["*", "linearSolver"]
        optional_objects = ["properties/linearSolver"]
        exclude_fields = ["properties/linearSolver/default"]


class _SlidingInterface(fl.SlidingInterface):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {"centerOfRotation": ("widget", "vector3")}


class _PorousMediumBox(fl.PorousMediumBox):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {
            "DarcyCoefficient": ("widget", "vector3"),
            "ForchheimerCoefficient": ("widget", "vector3"),
            "volumeZone/center": ("widget", "vector3"),
            "volumeZone/lengths": ("widget", "vector3"),
            "volumeZone/axes/items": ("widget", "vector3"),
            "volumeZone/axes": (
                "options",
                {"orderable": False, "addable": False, "removable": False},
            ),
            "volumeZone/windowingLengths": ("widget", "vector3"),
        }


class _ActuatorDisk(fl.ActuatorDisk):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {"center": ("widget", "vector3"), "axisThrust": ("widget", "vector3")}


class _BETDiskTwist(BETDiskTwist):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        displayed = "BET disk twist"


class _BETDiskChord(BETDiskChord):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        displayed = "BET disk chord"


class _BETDiskSectionalPolar(BETDiskSectionalPolar):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        displayed = "BET disk sectional polar"


class _BETDisk(fl.BETDisk):
    twists: List[_BETDiskTwist] = pd.Field(displayed="BET disk twists")
    chords: List[_BETDiskChord] = pd.Field(displayed="BET disk chords")
    sectional_polars: List[_BETDiskSectionalPolar] = pd.Field(
        displayed="Sectional polars", alias="sectionalPolars"
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {
            "centerOfRotation": ("widget", "vector3"),
            "axisOfRotation": ("widget", "vector3"),
            "initialBladeDirection": ("widget", "vector3"),
            "radius": ("field", "unitInput"),
            "omega": ("field", "unitInput"),
            "chordRef": ("field", "unitInput"),
            "thickness": ("field", "unitInput"),
            "bladeLineChord": ("field", "unitInput"),
        }
        displayed = "BET disk"


class _VolumeOutput(fl.VolumeOutput):
    output_format: List[Literal["paraview", "tecplot", "both"]]

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "outputFields", "*"]


class _AeroacousticOutput(fl.AeroacousticOutput):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {"observers/items": ("widget", "vector3")}


class _SliceOutput(fl.SliceOutput):
    # pylint: disable=protected-access, too-few-public-methods
    output_format: List[Literal["paraview", "tecplot", "both"]]

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "outputFields", "*", "slices"]
        field_properties = {
            "slices/additionalProperties/sliceNormal": ("widget", "vector3"),
            "slices/additionalProperties/sliceOrigin": ("widget", "vector3"),
        }
        swap_fields = {"slices": fl.Slices.flow360_schema()}
        exclude_fields = ["properties/slices/additionalProperties/title"]


class _MonitorOutput(fl.MonitorOutput):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFields", "*"]
        field_properties = {
            "monitors/additionalProperties/monitorLocations/items": ("widget", "vector3")
        }
        swap_fields = {"monitors": fl.Monitors.flow360_schema()}
        exclude_fields = ["properties/monitors/additionalProperties/title"]


class _SurfaceOutput(fl.SurfaceOutput):
    output_format: List[Literal["paraview", "tecplot", "both"]]

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "outputFields", "*", "surfaces"]
        swap_fields = {"surfaces": fl.Surfaces.flow360_schema()}
        exclude_fields = ["properties/surfaces/additionalProperties/title"]


class _IsoSurfaceOutput(fl.IsoSurfaceOutput):
    output_format: List[Literal["paraview", "tecplot", "both"]]

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "*", "isoSurfaces"]
        swap_fields = {"isoSurfaces": fl.IsoSurfaces.flow360_schema()}
        exclude_fields = ["properties/isoSurfaces/additionalProperties/title"]


class _Boundaries(fl.Boundaries):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        optional_objects = ["anyOf/properties/turbulenceQuantities"]
        field_properties = {
            "Velocity": ("field", "unitInput"),
            "velocityDirection": ("widget", "vector3"),
            "translationVector": ("widget", "vector3"),
            "axisOfRotation": ("widget", "vector3"),
        }

    @classmethod
    def _mark_const(cls, dictionary, const_key):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if key == const_key:
                dictionary[key]["readOnly"] = True
            elif isinstance(value, dict):
                cls._mark_const(value, const_key)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._mark_const(item, const_key)

        return dictionary

    @classmethod
    def _modify_boundary_types(cls, dictionary):
        if not isinstance(dictionary, dict):
            raise ValueError("Input must be a dictionary")

        for key, value in list(dictionary.items()):
            if key == "Velocity":
                dictionary[key].update(value["anyOf"][0])
                dictionary[key]["type"] = "object"
                del value["anyOf"]
            if key == "velocityDirection":
                dictionary[key].update(value["anyOf"][0])
                dictionary[key]["type"] = "array"
                del value["anyOf"]
            elif isinstance(value, dict):
                cls._modify_boundary_types(value)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, dict):
                        cls._modify_boundary_types(item)

        return dictionary

    @classmethod
    def flow360_schema(cls):
        root_schema = {"anyOf": []}

        models = cls.get_subtypes()

        for model in models:
            schema = model.flow360_schema()
            root_schema["anyOf"].append(schema)

        if cls.SchemaConfig.displayed is not None:
            root_schema["displayed"] = cls.SchemaConfig.displayed
        for item in cls.SchemaConfig.exclude_fields:
            cls._schema_remove(root_schema, item.split("/"))
        for item in cls.SchemaConfig.optional_objects:
            cls._schema_generate_optional(root_schema, item.split("/"))
        if cls.SchemaConfig.swap_fields is not None:
            for key, value in cls.SchemaConfig.swap_fields.items():
                value["title"] = root_schema["properties"][key]["title"]
                displayed = root_schema["properties"][key].get("displayed")
                if displayed is not None:
                    value["displayed"] = displayed
                root_schema["properties"][key] = value
        cls._schema_swap_key(root_schema, "title", "displayed")
        cls._schema_clean(root_schema)
        cls._mark_const(root_schema, "name")
        cls._modify_boundary_types(root_schema)

        definitions = {}

        cls._collect_all_definitions(root_schema, definitions)

        if definitions:
            root_schema["definitions"] = definitions

        root_schema["type"] = "object"
        root_schema["additionalProperties"] = False

        return root_schema


write_schemas(_Geometry, "geometry")
write_schemas(_NavierStokesSolver, "navier-stokes")
write_schemas(_TransitionModelSolver, "transition-model")
write_schemas(_HeatEquationSolver, "heat-equation")
write_schemas(_TurbulenceModelSolver, "turbulence-model")
write_schemas(_SlidingInterface, "sliding-interface")
write_schemas(_PorousMediumBox, "porous-media")
write_schemas(_ActuatorDisk, "actuator-disk")
write_schemas(_BETDisk, "bet-disk")
write_schemas(_VolumeOutput, "volume-output")
write_schemas(_AeroacousticOutput, "aeroacoustic-output")
write_schemas(_SliceOutput, "slice-output")
write_schemas(_MonitorOutput, "monitor-output")
write_schemas(_SurfaceOutput, "surface-output")
write_schemas(_IsoSurfaceOutput, "iso-surface-output")
write_schemas(_Freestream, "freestream")
write_schemas(_TimeStepping, "time-stepping")
write_schemas(_FluidProperties, "fluid-properties")
write_schemas(_InitialConditions, "initial-conditions")
write_schemas(_VolumeZones, "volume-zones")
write_schemas(_Boundaries, "boundaries")
