import json
import os
from typing import List, Optional, Type, Union, get_args

import pydantic as pd

import flow360 as fl
from flow360.component.flow360_params.flow360_params import (
    BETDiskChord,
    BETDiskSectionalPolar,
    BETDiskTwist,
)
from flow360.component.flow360_params.initial_condition import (
    ExpressionInitialCondition,
    FreestreamInitialCondition,
)
from flow360.component.flow360_params.params_base import (
    Flow360BaseModel,
    Flow360SortableBaseModel,
)
from flow360.component.flow360_params.volume_zones import (
    ReferenceFrame,
    ReferenceFrameDynamic,
    ReferenceFrameExpression,
    VolumeZoneBase,
)
from flow360.component.types import PositiveFloat

here = os.path.dirname(os.path.abspath(__file__))


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
    write_to_file(os.path.join(here, "data", folder_name, "json-schema.json"), schema)
    ui_schema = json.dumps(type_obj.flow360_ui_schema(), indent=2, sort_keys=True)
    if ui_schema is not None:
        write_to_file(os.path.join(here, "data", folder_name, "ui-schema.json"), ui_schema)


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


class _TurbulenceModelSolver(Flow360BaseModel):
    solver: Union[fl.SpalartAllmaras, fl.KOmegaSST, fl.NoneSolver] = pd.Field(
        options=["Spalart-Allmaras", "kOmegaSST", "None"]
    )

    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["modelType", "*", "linearSolverConfig"]
        optional_objects = ["anyOf/properties/linearSolverConfig"]
        exclude_fields = ["anyOf/properties/linearSolverConfig/default"]
        root_property = "properties/solver/anyOf"


class _TimeStepping(Flow360BaseModel):
    class _UnsteadyTimeStepping(fl.UnsteadyTimeStepping):
        class RampCFLUnsteady(fl.RampCFL):
            initial: Optional[PositiveFloat] = pd.Field(
                default=fl.RampCFL.default_unsteady().initial
            )
            final: Optional[PositiveFloat] = pd.Field(default=fl.RampCFL.default_unsteady().final)
            ramp_steps: Optional[int] = pd.Field(
                alias="rampSteps", default=fl.RampCFL.default_unsteady().ramp_steps
            )

        class AdaptiveCFLUnsteady(fl.AdaptiveCFL):
            max: Optional[PositiveFloat] = pd.Field(default=fl.AdaptiveCFL.default_unsteady().max)
            convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
                alias="convergenceLimitingFactor",
                default=fl.AdaptiveCFL.default_unsteady().convergence_limiting_factor,
            )
            max_relative_change: Optional[PositiveFloat] = pd.Field(
                alias="maxRelativeChange",
                default=fl.AdaptiveCFL.default_unsteady().max_relative_change,
            )

        CFL: Optional[Union[RampCFLUnsteady, AdaptiveCFLUnsteady]] = pd.Field(
            displayed="CFL", options=["Ramp CFL", "Adaptive CFL"]
        )

    class _SteadyTimeStepping(fl.SteadyTimeStepping):
        class RampCFLSteady(fl.RampCFL):
            initial: Optional[PositiveFloat] = pd.Field(default=fl.RampCFL.default_steady().initial)
            final: Optional[PositiveFloat] = pd.Field(default=fl.RampCFL.default_steady().final)
            ramp_steps: Optional[int] = pd.Field(
                alias="rampSteps", default=fl.RampCFL.default_steady().ramp_steps
            )

        class AdaptiveCFLSteady(fl.AdaptiveCFL):
            max: Optional[PositiveFloat] = pd.Field(default=fl.AdaptiveCFL.default_steady().max)
            convergence_limiting_factor: Optional[PositiveFloat] = pd.Field(
                alias="convergenceLimitingFactor",
                default=fl.AdaptiveCFL.default_steady().convergence_limiting_factor,
            )
            max_relative_change: Optional[PositiveFloat] = pd.Field(
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
        root_property = "properties/timeStepping/anyOf"
        exclude_fields = [
            root_property + "/properties/CFL/default",
        ]


class _FluidProperties(Flow360BaseModel):
    fluid_properties: Union[fl.AirDensityTemperature, fl.AirPressureTemperature] = pd.Field(
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
    initial_conditions: Union[FreestreamInitialCondition, ExpressionInitialCondition] = pd.Field(
        alias="initialConditions", options=["Freestream", "Expression"]
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
        field_order = ["*", "linearSolverConfig"]
        optional_objects = ["properties/linearSolverConfig"]
        exclude_fields = ["properties/linearSolverConfig/default"]


class _TransitionModelSolver(fl.TransitionModelSolver):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["modelType", "*", "linearSolverConfig"]
        optional_objects = ["properties/linearSolverConfig"]
        exclude_fields = ["properties/linearSolverConfig/default"]


class _HeatEquationSolver(fl.HeatEquationSolver):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["*", "linearSolverConfig"]
        optional_objects = ["properties/linearSolverConfig"]
        exclude_fields = ["properties/linearSolverConfig/default"]


class _SlidingInterface(fl.SlidingInterface):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {"centerOfRotation": ("widget", "vector3")}


class _PorousMedium(fl.PorousMedium):
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
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "outputFields", "*"]


class _AeroacousticOutput(fl.AeroacousticOutput):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {"observers/items": ("widget", "vector3")}


class _SliceOutput(fl.SliceOutput):
    # pylint: disable=protected-access, too-few-public-methods
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
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "outputFields", "*", "surfaces"]
        swap_fields = {"surfaces": fl.Surfaces.flow360_schema()}
        exclude_fields = ["properties/surfaces/additionalProperties/title"]


class _IsoSurfaceOutput(fl.IsoSurfaceOutput):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_order = ["outputFormat", "*", "isoSurfaces"]
        swap_fields = {"isoSurfaces": fl.IsoSurfaces.flow360_schema()}
        exclude_fields = ["properties/isoSurfaces/additionalProperties/title"]


class _Boundaries(fl.Boundaries):
    class SchemaConfig(Flow360BaseModel.SchemaConfig):
        field_properties = {
            "additionalProperties/velocity/value": ("widget", "vector3"),
            "additionalProperties/Velocity/value": ("widget", "vector3"),
            "additionalProperties/velocityDirection/value": ("widget", "vector3"),
            "additionalProperties/translationVector": ("widget", "vector3"),
            "additionalProperties/axisOfRotation": ("widget", "vector3"),
        }


write_schemas(_Geometry, "geometry")
write_schemas(_NavierStokesSolver, "navier-stokes")
write_schemas(_TransitionModelSolver, "transition-model")
write_schemas(_HeatEquationSolver, "heat-equation")
write_schemas(_TurbulenceModelSolver, "turbulence-model")
write_schemas(_SlidingInterface, "sliding-interface")
write_schemas(_PorousMedium, "porous-media")
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
