"""Utilities for processing solver outputs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

import numpy as np

from flow360.component.results.base_results import (
    _PHYSICAL_STEP,
    _PSEUDO_STEP,
    LocalResultCSVModel,
    ResultCSVModel,
)
from flow360.component.simulation.models.volume_models import BETDisk
from flow360.component.simulation.simulation_params import SimulationParams
from flow360.exceptions import Flow360ValueError
from flow360.log import log

# pylint:disable=invalid-name
_CL = "CL"
_CD = "CD"
_CFx = "CFx"
_CFy = "CFy"
_CFz = "CFz"
_CMx = "CMx"
_CMy = "CMy"
_CMz = "CMz"
_CL_PRESSURE = "CLPressure"
_CD_PRESSURE = "CDPressure"
_CFx_PRESSURE = "CFxPressure"
_CFy_PRESSURE = "CFyPressure"
_CFz_PRESSURE = "CFzPressure"
_CMx_PRESSURE = "CMxPressure"
_CMy_PRESSURE = "CMyPressure"
_CMz_PRESSURE = "CMzPressure"
_CL_SKIN_FRICTION = "CLSkinFriction"
_CD_SKIN_FRICTION = "CDSkinFriction"
_CFx_SKIN_FRICTION = "CFxSkinFriction"
_CFy_SKIN_FRICTION = "CFySkinFriction"
_CFz_SKIN_FRICTION = "CFzSkinFriction"
_CMx_SKIN_FRICTION = "CMxSkinFriction"
_CMy_SKIN_FRICTION = "CMySkinFriction"
_CMz_SKIN_FRICTION = "CMzSkinFriction"
_CL_VISCOUS = "CLViscous"
_CD_VISCOUS = "CDViscous"
_CFx_VISCOUS = "CFxViscous"
_CFy_VISCOUS = "CFyViscous"
_CFz_VISCOUS = "CFzViscous"
_CMx_VISCOUS = "CMxViscous"
_CMy_VISCOUS = "CMyViscous"
_CMz_VISCOUS = "CMzViscous"
_HEAT_TRANSFER = "HeatTransfer"
_HEAT_FLUX = "HeatFlux"
_X = "X"
_Y = "Y"
_CUMULATIVE_CD_CURVE = "Cumulative_CD_Curve"
_CD_PER_STRIP = "CD_per_strip"
_CFx_PER_SPAN = "CFx_per_span"
_CFz_PER_SPAN = "CFz_per_span"
_CMy_PER_SPAN = "CMy_per_span"


# Static utilities for aerodynamic coefficient computations.

# Provides helper methods for computing aerodynamic coefficients using
# reference geometry and operating condition information.


def _vector_to_np3(vec):
    try:
        return np.array([float(vec[0]), float(vec[1]), float(vec[2])], dtype=float)
    except Exception as exc:  # pylint:disable=broad-except
        raise Flow360ValueError(f"Invalid vector: {vec}") from exc


def _get_reference_geometry(params: SimulationParams):
    # pylint:disable=import-outside-toplevel, no-member, protected-access
    from flow360.component.simulation.primitives import ReferenceGeometry

    # Fill defaults using preprocessed params
    reference_geometry_filled = ReferenceGeometry.fill_defaults(params.reference_geometry, params)

    reference_geometry_filled_flow360: ReferenceGeometry = reference_geometry_filled.preprocess(
        params=params
    )
    # Extract dimensionless area (in Flow360 units)
    area_flow360 = float(reference_geometry_filled_flow360.area.value)

    # Extract dimensionless moment_length
    moment_length_flow360 = reference_geometry_filled_flow360.moment_length

    # Convert to numpy array properly - handle both arrays and scalars
    try:
        # Try to treat it as a vector-like object
        len(moment_length_flow360)
        moment_length_vec_flow360 = np.array(
            [
                float(moment_length_flow360[0]),
                float(moment_length_flow360[1]),
                float(moment_length_flow360[2]),
            ],
            dtype=float,
        )
    except (TypeError, AttributeError):
        # It's a scalar, replicate for all three directions
        scalar_val = float(moment_length_flow360)
        moment_length_vec_flow360 = np.array([scalar_val, scalar_val, scalar_val], dtype=float)

    # Extract dimensionless moment_center
    moment_center = reference_geometry_filled_flow360.moment_center
    moment_center_flow360 = np.array(
        [moment_center[0], moment_center[1], moment_center[2]],
        dtype=float,
    )

    return area_flow360, moment_length_vec_flow360, moment_center_flow360


def _get_lift_drag_direction(params: SimulationParams):
    oc = params.operating_condition

    if oc is None:
        raise Flow360ValueError("Operating condition is required for computing freestream vectors.")

    # Check if it's GenericReferenceCondition which doesn't have alpha/beta
    if oc.type_name == "GenericReferenceCondition":
        log.info(
            "Operating condition is `GenericReferenceCondition` without alpha/beta angles. "
            "Assuming lift direction = (0, 0, 1), drag direction = (1, 0, 0)."
        )
        lift_dir = np.array([0.0, 0.0, 1.0], dtype=float)
        drag_dir = np.array([1.0, 0.0, 0.0], dtype=float)
        return lift_dir, drag_dir

    alpha_rad = float(oc.alpha.to("rad").value)
    beta_rad = float(oc.beta.to("rad").value)

    u_inf = np.array(
        [
            np.cos(alpha_rad) * np.cos(beta_rad),
            -np.sin(beta_rad),
            np.sin(alpha_rad) * np.cos(beta_rad),
        ],
        dtype=float,
    )

    lift_dir = np.array([-np.sin(alpha_rad), 0.0, np.cos(alpha_rad)], dtype=float)

    drag_dir = u_inf
    return lift_dir, drag_dir


def _get_dynamic_pressure_in_flow360_unit(params: SimulationParams):
    # pylint:disable=protected-access
    oc = params.operating_condition
    using_liquid_op = oc.type_name == "LiquidOperatingCondition"

    if using_liquid_op:
        v_ref = params._liquid_reference_velocity
    else:
        v_ref = params.base_velocity

    Mach_ref = params.convert_unit(value=v_ref, target_system="flow360").value
    return 0.5 * Mach_ref * Mach_ref


def _build_coeff_env(params) -> Dict[str, Any]:
    # pylint:disable=protected-access
    area, moment_length_vec, moment_center_global = _get_reference_geometry(params)
    dynamic_pressure = _get_dynamic_pressure_in_flow360_unit(params)
    lift_dir, drag_dir = _get_lift_drag_direction(params)
    return {
        "moment_center_global": moment_center_global,
        "dynamic_pressure": dynamic_pressure,
        "area": area,
        "moment_length_vec": moment_length_vec,
        "lift_dir": lift_dir,
        "drag_dir": drag_dir,
    }


def _copy_time_columns(src: Dict[str, list]) -> Dict[str, list]:
    out: Dict[str, list] = {}
    out[_PSEUDO_STEP] = src[_PSEUDO_STEP]
    out[_PHYSICAL_STEP] = src[_PHYSICAL_STEP]
    return out


def _collect_disk_axes_and_centers(
    params: SimulationParams, model_type: str
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Collect normalized disk axes and centers for a given model type.

    Parameters
    ----------
    params : SimulationParams
        Simulation parameters including models and entities.
    model_type : str
        The type name of the disk model to match (e.g. "ActuatorDisk", "BETDisk").

    Returns
    -------
    (disk_axes, disk_centers) : Tuple[List[np.ndarray], List[np.ndarray]]
        Lists of axis vectors (normalized) and center positions (dimensionless Flow360 units).
    """

    disk_axes: List[np.ndarray] = []
    disk_centers: List[np.ndarray] = []

    for model in params.models:
        if model.type != model_type:
            continue
        for cyl in model.entities.stored_entities:
            # Axis is assumed normalized by the inputs
            # pylint:disable=protected-access
            axis = _vector_to_np3(cyl.axis)
            center_flow360 = params.convert_unit(value=cyl.center, target_system="flow360")
            center_np = np.array(
                [center_flow360[0].value, center_flow360[1].value, center_flow360[2].value],
                dtype=float,
            )
            disk_axes.append(axis)
            disk_centers.append(center_np)

    return disk_axes, disk_centers


class DiskCoefficientsComputation:
    # pylint:disable=too-few-public-methods
    """
    Static utilities for disk coefficient computations.

    This class provides only static methods and should not be instantiated or subclassed.
    All methods are self-contained and require explicit parameters.
    """

    @staticmethod
    def _iter_disks(params, disk_model_type: str, values: Dict[str, list]):
        disk_axes, disk_centers = _collect_disk_axes_and_centers(params, disk_model_type)
        # Extract disk names matching pattern "Disk<number>"
        disk_pattern = re.compile(r"^(Disk\d+)_")
        disk_names = np.unique(
            [match.group(1) for key in values.keys() if (match := disk_pattern.match(key))]
        )
        for disk_name in disk_names:
            # Extract disk index from pattern "Disk<number>"
            match = re.search(r"Disk(\d+)", disk_name)
            if match:
                idx = int(match.group(1))
                axis = disk_axes[idx]
                center = disk_centers[idx]
                yield disk_name, axis, center

    @staticmethod
    def _init_disk_output(out: Dict[str, list], disk_name: str) -> Dict[str, str]:
        keys = {
            "CFx": f"{disk_name}_{_CFx}",
            "CFy": f"{disk_name}_{_CFy}",
            "CFz": f"{disk_name}_{_CFz}",
            "CMx": f"{disk_name}_{_CMx}",
            "CMy": f"{disk_name}_{_CMy}",
            "CMz": f"{disk_name}_{_CMz}",
            "CL": f"{disk_name}_{_CL}",
            "CD": f"{disk_name}_{_CD}",
        }
        out[keys["CFx"]], out[keys["CFy"]], out[keys["CFz"]] = [], [], []
        out[keys["CMx"]], out[keys["CMy"]], out[keys["CMz"]] = [], [], []
        out[keys["CL"]], out[keys["CD"]] = [], []
        return keys

    @staticmethod
    def compute_coefficients_static(
        params: SimulationParams,
        values: Dict[str, list],
        disk_model_type: str,
        iterate_step_values_func,
        coefficients_model_class,
    ):
        """
        Compute disk coefficients from raw force/moment data.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters containing reference geometry and flow conditions
        values : Dict[str, list]
            Dictionary containing time series data (pseudo/physical step and disk forces/moments)
        disk_model_type : str
            Type of disk model (e.g., "ActuatorDisk" or "BETDisk")
        iterate_step_values_func : callable
            Function that yields (CF, CM, CL, CD) for each time step.
            Signature: func(disk_name, disk_ctx, env, values) -> Iterator[Tuple]
        coefficients_model_class : type
            Class to instantiate for the output coefficients model

        Returns
        -------
        coefficients_model_class instance
            Model containing computed coefficients
        """
        if not isinstance(params, SimulationParams):
            raise ValueError(
                "compute_coefficients() is not supported for legacy cases with Flow360Params."
            )

        # pylint:disable=protected-access
        env = _build_coeff_env(params)
        out = _copy_time_columns(values)

        for disk_name, axis, center in DiskCoefficientsComputation._iter_disks(
            params, disk_model_type, values
        ):
            DiskCoefficientsComputation._init_disk_output(out, disk_name)
            for CF, CM, CL_val, CD_val in iterate_step_values_func(  # pylint:disable=invalid-name
                disk_name, {"axis": axis, "center": center}, env, values
            ):
                out[f"{disk_name}_{_CFx}"].append(CF[0])
                out[f"{disk_name}_{_CFy}"].append(CF[1])
                out[f"{disk_name}_{_CFz}"].append(CF[2])
                out[f"{disk_name}_{_CMx}"].append(CM[0])
                out[f"{disk_name}_{_CMy}"].append(CM[1])
                out[f"{disk_name}_{_CMz}"].append(CM[2])
                out[f"{disk_name}_{_CD}"].append(CD_val)
                out[f"{disk_name}_{_CL}"].append(CL_val)

        return coefficients_model_class().from_dict(out)


class PorousMediumCoefficientsComputation:
    # pylint:disable=too-few-public-methods
    """Static utilities for porous medium coefficient computations."""

    @staticmethod
    def _copy_time_columns(src: Dict[str, list]) -> Dict[str, list]:
        out: Dict[str, list] = {}
        out[_PSEUDO_STEP] = src[_PSEUDO_STEP]
        out[_PHYSICAL_STEP] = src[_PHYSICAL_STEP]
        return out

    @staticmethod
    def _iter_zones(values: Dict[str, list]):
        zone_names = np.unique(
            [
                v.split("_")[0] + "_" + v.split("_")[1]
                for v in values.keys()
                if v.startswith("zone_")
            ]
        )
        yield from zone_names

    @staticmethod
    def _init_zone_output(out: Dict[str, list], zone_name: str) -> Dict[str, str]:
        keys = {
            "CFx": f"{zone_name}_{_CFx}",
            "CFy": f"{zone_name}_{_CFy}",
            "CFz": f"{zone_name}_{_CFz}",
            "CMx": f"{zone_name}_{_CMx}",
            "CMy": f"{zone_name}_{_CMy}",
            "CMz": f"{zone_name}_{_CMz}",
            "CL": f"{zone_name}_{_CL}",
            "CD": f"{zone_name}_{_CD}",
        }
        out[keys["CFx"]], out[keys["CFy"]], out[keys["CFz"]] = [], [], []
        out[keys["CMx"]], out[keys["CMy"]], out[keys["CMz"]] = [], [], []
        out[keys["CL"]], out[keys["CD"]] = [], []
        return keys

    @staticmethod
    def compute_coefficients_static(
        params: SimulationParams,
        values: Dict[str, list],
        iterate_step_values_func,
        coefficients_model_class,
    ):
        """
        Compute porous medium coefficients from raw force/moment data.

        Parameters
        ----------
        params : SimulationParams
            Simulation parameters containing reference geometry and flow conditions
        values : Dict[str, list]
            Dictionary containing time series data (pseudo/physical step and zone forces/moments)
        iterate_step_values_func : callable
            Function that yields (CF, CM, CL, CD) for each time step.
            Signature: func(zone_name, zone_ctx, env, values) -> Iterator[Tuple]
        coefficients_model_class : type
            Class to instantiate for the output coefficients model

        Returns
        -------
        coefficients_model_class instance
            Model containing computed coefficients
        """
        if not isinstance(params, SimulationParams):
            raise ValueError(
                "compute_coefficients() is not supported for legacy cases with Flow360Params."
            )

        # pylint:disable=protected-access
        env = _build_coeff_env(params)
        out = _copy_time_columns(values)

        for zone_name in PorousMediumCoefficientsComputation._iter_zones(values):
            PorousMediumCoefficientsComputation._init_zone_output(out, zone_name)
            for CF, CM, CL_val, CD_val in iterate_step_values_func(zone_name, {}, env, values):
                out[f"{zone_name}_{_CFx}"].append(CF[0])
                out[f"{zone_name}_{_CFy}"].append(CF[1])
                out[f"{zone_name}_{_CFz}"].append(CF[2])
                out[f"{zone_name}_{_CMx}"].append(CM[0])
                out[f"{zone_name}_{_CMy}"].append(CM[1])
                out[f"{zone_name}_{_CMz}"].append(CM[2])
                out[f"{zone_name}_{_CD}"].append(CD_val)
                out[f"{zone_name}_{_CL}"].append(CL_val)

        return coefficients_model_class().from_dict(out)


class BETDiskCSVHeaderOperation:
    # pylint:disable=too-few-public-methods
    """
    Static utilities for renaming BET disk csv output headers to include the name of the BET disk.

    This class provides only static methods and should not be instantiated or subclassed.
    All methods are self-contained and require explicit parameters.
    """

    @staticmethod
    def format_headers(
        BETCSVModel: ResultCSVModel,
        params: SimulationParams,
        pattern: str = "$BETName_$CylinderName",
    ) -> LocalResultCSVModel:
        """
        renames the header entries in a BET csv file from Disk{x}_ based on input pattern
        $Default option is $BETName_$CylinderName

        pattern can take [$BETName, $CylinderName, $DiskLocalIndex, $DiskGlobalIndex]
        Parameters
        ----------
        BETCSVModel : ResultCSVModle
            Model containing csv entries
        params : SimulationParams
            Simulation parameters
        pattern : str
            Pattern string to rename header entries. Available patterns
            [$BETName, $CylinderName, $DiskLocalIndex, $DiskGlobalIndex]
        Returns
        -------
        LocalResultCSVModel
            Model containing csv with updated header
        """
        # pylint:disable=too-many-locals
        bet_disks = []
        for model in params.models:
            if isinstance(model, BETDisk):
                bet_disks.append(model)
        if not bet_disks:
            raise ValueError("No BET Disks in params to rename header.")

        csv_data = BETCSVModel.values
        new_csv = {}

        disk_rename_map = {}

        diskCount = 0
        for disk in bet_disks:
            for disk_local_index, cylinder in enumerate(disk.entities.stored_entities):
                new_name = pattern.replace("$BETName", disk.name)
                new_name = new_name.replace("$CylinderName", cylinder.name)
                new_name = new_name.replace("$DiskLocalIndex", str(disk_local_index))
                new_name = new_name.replace("$DiskGlobalIndex", str(diskCount))
                disk_rename_map[f"Disk{diskCount}"] = new_name
                diskCount = diskCount + 1

        for header, values in csv_data.items():
            matched = False
            for default_prefix, new_prefix in disk_rename_map.items():
                if header.startswith(default_prefix):
                    new_csv[new_prefix + header[len(default_prefix) :]] = values
                    matched = True
                    break
            if not matched:
                new_csv[header] = values
        newModel = LocalResultCSVModel().from_dict(new_csv)
        return newModel
