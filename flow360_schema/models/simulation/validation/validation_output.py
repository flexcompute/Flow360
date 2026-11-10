"""
Validation for output parameters
"""

import math
from typing import Literal, Union, get_args, get_origin

from flow360_schema.framework.entity.entity_utils import subclass_predicate
from flow360_schema.framework.expression import Expression, UserVariable
from flow360_schema.framework.validation.context import add_validation_warning
from flow360_schema.models.entities.output_entities import Isosurface, PointGroup, Slice
from flow360_schema.models.entities.surface_entities import ImportedSurface
from flow360_schema.models.simulation.models.volume_models import Fluid, Rotation
from flow360_schema.models.simulation.outputs.outputs import (
    AeroAcousticOutput,
    ForceDistributionOutput,
    PROBE_OUTPUT_TYPE_NAMES,
    ProbeOutputTypes,
    RenderOutput,
    StreamlineOutput,
    SurfaceIntegralOutput,
    TimeAverageForceDistributionOutput,
    TimeAverageSliceOutput,
    TimeAverageVolumeOutput,
    TimeAverageSurfaceOutput,
    sanitize_file_name,
)
from flow360_schema.models.simulation.time_stepping.time_stepping import Steady
from flow360_schema.models.simulation.validation.validation_context import (
    get_validation_info,
)
from flow360_schema.models.simulation.validation.validation_utils import (
    customize_model_validator_error,
)


def _check_output_fields(params):
    """Check the specified output fields for each output item is valid."""

    if params.outputs is None:
        return params

    has_legacy_user_defined_field_in_surface_integral_output = False
    for output in params.outputs:
        if isinstance(output, SurfaceIntegralOutput):
            for output_field in output.output_fields.items:
                if isinstance(output_field, str):
                    has_legacy_user_defined_field_in_surface_integral_output = True
                    break
    has_user_defined_fields = len(params.user_defined_fields) > 0

    if has_legacy_user_defined_field_in_surface_integral_output and has_user_defined_fields is False:
        raise ValueError(
            "The legacy string output fields in `SurfaceIntegralOutput` must be used with `UserDefinedField`."
        )

    def extract_literal_values(annotation):
        origin = get_origin(annotation)
        if origin is Union:
            # Traverse each Union argument
            results = []
            for arg in get_args(annotation):
                result = extract_literal_values(arg)
                if result:
                    results.extend(result)
            return results
        if origin is list or origin is list:
            # Apply the function to the List's element type
            return extract_literal_values(get_args(annotation)[0])
        if origin is Literal:
            return list(get_args(annotation))
        return []

    additional_fields = [item.name for item in params.user_defined_fields]

    for output_index, output in enumerate(params.outputs):
        if output.output_type in (
            "AeroAcousticOutput",
            "StreamlineOutput",
            "ForceDistributionOutput",
            "TimeAverageForceDistributionOutput",
            "RenderOutput",
        ):
            continue
        # Get allowed output fields items:
        natively_supported = extract_literal_values(output.output_fields.__class__.model_fields["items"].annotation)
        allowed_items = natively_supported + additional_fields

        for item in output.output_fields.items:
            if isinstance(item, str) and item not in allowed_items:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}:, {item} is not a"
                    f" valid output field name. Allowed fields are {allowed_items}."
                )

        if output.output_type == "IsosurfaceOutput":
            # using the 1st item's allowed field as all isosurface have same field definition
            allowed_items = (
                extract_literal_values(output.entities.items[0].__class__.model_fields["field"].annotation)
                + additional_fields
            )
            for entity in output.entities.items:
                if isinstance(entity.field, str) and entity.field not in allowed_items:
                    raise ValueError(
                        f"In `outputs`[{output_index}] {output.output_type}:, {entity.field} is not a"
                        f" valid iso field name. Allowed fields are {allowed_items}."
                    )

    return params


def _check_output_fields_valid_given_turbulence_model(params):
    """Ensure that the output fields are consistent with the turbulence model used."""

    if not params.models or not params.outputs:
        return params

    turbulence_model = None

    # mutRatio is a derived function registered inside the turbulence solver
    # (TurbulenceSolver.h), so a laminar run emits no column for it -- mut is
    # different, the Navier-Stokes solver registers that one itself.
    invalid_output_fields = {
        "None": ("kOmega", "nuHat", "residualTurbulence", "solutionTurbulence", "mutRatio"),
        "SpalartAllmaras": ("kOmega",),
        "kOmegaSST": ("nuHat",),
    }
    for model in params.models:
        if isinstance(model, Fluid):
            turbulence_model = model.turbulence_model_solver.type_name
            break

    for output_index, output in enumerate(params.outputs):
        if output.output_type in (
            "AeroAcousticOutput",
            "StreamlineOutput",
            "ForceDistributionOutput",
            "TimeAverageForceDistributionOutput",
            "RenderOutput",
        ):
            continue
        for item in output.output_fields.items:
            if isinstance(item, str) and item in invalid_output_fields[turbulence_model]:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: {item} is not a valid"
                    f" output field when using turbulence model: {turbulence_model}."
                )

        if output.output_type == "IsosurfaceOutput":
            for entity in output.entities.items:
                if isinstance(entity.field, str) and entity.field in invalid_output_fields[turbulence_model]:
                    raise ValueError(
                        f"In `outputs`[{output_index}] {output.output_type}: {entity.field} is not a valid"
                        f" iso field when using turbulence model: {turbulence_model}."
                    )
    return params


def _check_output_fields_valid_given_transition_model(params):
    """Ensure that the output fields are consistent with the transition model used."""

    if not params.models or not params.outputs:
        return params

    transition_model = "None"
    for model in params.models:
        if isinstance(model, Fluid):
            transition_model = model.transition_model_solver.type_name
            break

    if transition_model != "None":
        return params

    transition_output_fields = [
        "residualTransition",
        "solutionTransition",
        "linearResidualTransition",
    ]

    for output_index, output in enumerate(params.outputs):
        if output.output_type in (
            "AeroAcousticOutput",
            "StreamlineOutput",
            "ForceDistributionOutput",
            "TimeAverageForceDistributionOutput",
            "RenderOutput",
        ):
            continue
        for item in output.output_fields.items:
            if isinstance(item, str) and item in transition_output_fields:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: {item} is not a valid"
                    f" output field when transition model is not used."
                )
    return params


def _check_unsteadiness_to_use_aero_acoustics(params):
    if not params.outputs:
        return params

    if isinstance(params.time_stepping, Steady):
        for output_index, output in enumerate(params.outputs):
            if isinstance(output, AeroAcousticOutput):
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}:"
                    "`AeroAcousticOutput` can only be activated with `Unsteady` simulation."
                )
    # Not running case or is using unsteady
    return params


def _check_local_cfl_output(params):
    """localCFL output is only valid for unsteady simulations."""

    if not params.outputs:
        return params

    if not isinstance(params.time_stepping, Steady):
        return params

    for output_index, output in enumerate(params.outputs):
        if output.output_type in (
            "AeroAcousticOutput",
            "StreamlineOutput",
            "ForceDistributionOutput",
            "TimeAverageForceDistributionOutput",
            "RenderOutput",
        ):
            continue
        for item in output.output_fields.items:
            if isinstance(item, str) and item == "localCFL":
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    "`localCFL` output is only supported for unsteady simulations."
                )

    return params


def _check_statistics_on_imported_surfaces(params):
    """Imported surfaces (only reachable through surface outputs) translate into a single
    merged, mean-only solver section, so a non-default statistics request on an instance
    containing one would be silently ignored; reject it. The translator additionally never
    emits ``averagingStatistics`` on that section, covering sibling regular-surface
    instances whose statistics are legal per-instance."""

    if not params.outputs:
        return params

    validation_info = get_validation_info()
    for output_index, output in enumerate(params.outputs):
        if not isinstance(output, TimeAverageSurfaceOutput):
            continue
        statistics = output.statistics
        if statistics == ["mean"]:
            continue
        if validation_info is not None:
            expanded_entities = validation_info.expand_entity_list(output.entities)
        else:
            expanded_entities = output.entities.stored_entities or []
        for entity in expanded_entities:
            if isinstance(entity, ImportedSurface):
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    f"`statistics` {statistics} is not supported on imported surface "
                    f"`{entity.name}`; only the default mean is available for imported surfaces."
                )

    return params


def _check_consistent_statistics_across_instances(params):
    """Volume and slice time-average instances merge into a single solver section whose
    statistics come from the first instance, so differing values would be silently dropped.
    Surface outputs translate to one solver config per instance and probe outputs to one
    monitor group per instance; both are exempt."""

    if not params.outputs:
        return params

    first_statistics_by_type = {}
    for output_index, output in enumerate(params.outputs):
        statistics = getattr(output, "statistics", None)
        if statistics is None or isinstance(output, (TimeAverageSurfaceOutput, *ProbeOutputTypes)):
            continue
        first = first_statistics_by_type.setdefault(output.output_type, (output_index, statistics))
        if statistics != first[1]:
            raise ValueError(
                f"In `outputs`[{output_index}] {output.output_type}: `statistics` {statistics} "
                f"differs from `outputs`[{first[0]}]'s {first[1]}; all `{output.output_type}` "
                "instances must request the same statistics."
            )

    return params


def _warn_assumed_vector_user_variables(params):
    """A user variable with three components is treated as a spatial vector and rotated when
    body-frame averaging is active. Warn so a user whose three components are unrelated
    scalars can split them up or opt that output out with ``averaging_reference_frame="global"``.

    A variable that is just a known solver vector (e.g. ``solution.velocity``, including the
    wrapper auto-created when one is passed in ``output_fields``) stays silent: its vectorness
    is not assumed, and the equivalent string request (``"velocity"``) does not warn either."""

    if not params.outputs or not params.models:
        return params

    if not any(isinstance(model, Rotation) for model in params.models):
        return params

    flagged = set()
    for output in params.outputs:
        if getattr(output, "averaging_reference_frame", None) != "body":
            continue
        output_fields = getattr(output, "output_fields", None)
        if output_fields is None:
            continue
        for item in output_fields.items:
            if not isinstance(item, UserVariable):
                continue
            try:
                is_three_component = getattr(item.value, "length", 0) == 3
            except ValueError:
                continue
            if not is_three_component:
                continue
            if isinstance(item.value, Expression) and item.value.expression.strip() in (
                item.value.solver_variable_names(recursive=False)
            ):
                continue
            flagged.add(item.name)

    if flagged:
        message = (
            f"Time-averaged user variable(s) {sorted(flagged)} have three components and are "
            "averaged as spatial vectors in the rotating frame of the geometry they are "
            "evaluated on. If they are not vectors, split them into separate scalar variables "
            'or set `averaging_reference_frame="global"`.'
        )
        add_validation_warning(message)

    return params


def _check_rms_output_field_name_collision(params):
    """With ``rms`` statistics every output field gains a ``<name>_rms`` sibling, so a field
    whose name already ends in ``_rms`` would collide with it."""

    if not params.outputs:
        return params

    for output_index, output in enumerate(params.outputs):
        statistics = getattr(output, "statistics", None)
        if statistics is None or "rms" not in statistics:
            continue
        for item in getattr(getattr(output, "output_fields", None), "items", []) or []:
            name = item if isinstance(item, str) else getattr(item, "name", "")
            if isinstance(name, str) and name.endswith("_rms"):
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: output field `{name}` "
                    "ends in `_rms`, which collides with the rms statistic's field naming."
                )

    return params


TIME_DERIVATIVE_OUTPUT_FIELDS = ("pressureTimeDerivative",)


def _check_time_derivative_output(params):
    """Time-derivative output fields (e.g. pressureTimeDerivative) are only valid for unsteady simulations."""

    if not params.outputs or not isinstance(params.time_stepping, Steady):
        return params

    for output_index, output in enumerate(params.outputs):
        for item in getattr(getattr(output, "output_fields", None), "items", []) or []:
            if isinstance(item, str) and item in TIME_DERIVATIVE_OUTPUT_FIELDS:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    f"`{item}` output is only supported for unsteady simulations."
                )

    return params


def _check_aero_acoustics_observer_time_step_size(params):
    if not params.outputs:
        return params

    for output_index, output in enumerate(params.outputs):
        if isinstance(output, AeroAcousticOutput):
            time_step_size = params.time_stepping.step_size
            if isinstance(params.time_stepping.step_size, Expression):
                time_step_size = params.time_stepping.step_size.evaluate(
                    raise_on_non_evaluable=True, force_evaluate=True
                )
            if output.observer_time_step_size and output.observer_time_step_size < time_step_size:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    f"`observer_time_size` ({output.observer_time_step_size}) is smaller than "
                    f"the time step size of CFD ({params.time_stepping.step_size})."
                )
    return params


def _get_first_aero_acoustic_output(params):
    """The only AeroAcousticOutput that reaches the solver, with its index. (None, None) if absent."""
    if not params.outputs:
        return None, None
    return next(
        ((index, item) for index, item in enumerate(params.outputs) if isinstance(item, AeroAcousticOutput)),
        (None, None),
    )


def _as_number_list(quantity) -> list[float]:
    """Magnitudes of a scalar or vector quantity as a flat list."""
    magnitude = quantity.value.tolist()
    return magnitude if isinstance(magnitude, list) else [magnitude]


def _quantities_match(value, parent_value) -> bool:
    """Compare two quantities of the same dimension, in the child's units."""
    magnitudes = _as_number_list(value)
    parent_magnitudes = _as_number_list(parent_value.to(value.units))
    if len(magnitudes) != len(parent_magnitudes):
        return False
    return all(
        math.isclose(magnitude, parent_magnitude, rel_tol=1e-9)
        for magnitude, parent_magnitude in zip(magnitudes, parent_magnitudes, strict=True)
    )


def _observers_match(observers, recorded_observers) -> bool:
    """Compare observers position by position, in order."""
    if len(observers) != len(recorded_observers):
        return False
    return all(
        observer.group_name == recorded.group_name and _quantities_match(observer.position, recorded.position)
        for observer, recorded in zip(observers, recorded_observers, strict=True)
    )


def _stored_entity_refs(entity_list):
    """(entity type name, id) pairs of the entities the list holds, or None when there is no list.

    Ids are only unique within an entity type, so identity is the pair.
    """
    if entity_list is None:
        return None
    return {
        (entity.private_attribute_entity_type_name, entity.private_attribute_id)
        for entity in entity_list.stored_entities
    }


def _fork_immutability_error(output_index, field_name, detail, may_be_unset=False):
    inherit_hint = ", leave it unset to inherit the parent's," if may_be_unset else ""
    return ValueError(
        f"In `outputs`[{output_index}] AeroAcousticOutput: `{field_name}` {detail}. It cannot be "
        "changed when a case is forked because the aeroacoustic solver continues the parent's "
        f"signal. Restore the parent's value{inherit_hint} or set `force_clean_start` to True to "
        "restart the aeroacoustic solver."
    )


def _check_aero_acoustics_fork_immutability(params):
    """A fork resumes the parent case's acoustic signal, so it cannot change what that signal is.

    The recorded settings are the parent's for a fork, and this case's own otherwise -- in
    which case they were filled from this very output at submission and match by
    construction. A case whose params predate the record has nothing to compare against and
    is left unconstrained.
    """
    output_index, output = _get_first_aero_acoustic_output(params)
    if output is None or output.force_clean_start:
        return params

    recorded = params.private_attribute_asset_cache.aeroacoustic_parent_case
    if recorded is None:
        return params

    if output.patch_type != recorded.recorded_patch_type:
        raise _fork_immutability_error(
            output_index,
            "patch_type",
            f"(`{output.patch_type}`) does not match the parent case (`{recorded.recorded_patch_type}`)",
        )

    recorded_surface_refs = (
        set(recorded.recorded_permeable_surfaces) if recorded.recorded_permeable_surfaces is not None else None
    )
    if _stored_entity_refs(output.permeable_surfaces) != recorded_surface_refs:
        raise _fork_immutability_error(output_index, "permeable_surfaces", "does not match the parent case")

    if not _observers_match(output.observers, recorded.recorded_observers):
        raise _fork_immutability_error(output_index, "observers", "does not match the parent case")

    # Unset inherits the parent's value, so only an explicit one can move it.
    if (
        output.observer_time_step_size is not None
        and recorded.recorded_observer_time_step_size is not None
        and not _quantities_match(output.observer_time_step_size, recorded.recorded_observer_time_step_size)
    ):
        raise _fork_immutability_error(
            output_index,
            "observer_time_step_size",
            f"({output.observer_time_step_size}) does not match the parent case "
            f"({recorded.recorded_observer_time_step_size})",
            may_be_unset=True,
        )

    if not _quantities_match(output.aeroacoustic_solver_start_time, recorded.recorded_aeroacoustic_solver_start_time):
        raise _fork_immutability_error(
            output_index,
            "aeroacoustic_solver_start_time",
            f"({output.aeroacoustic_solver_start_time}) does not match the parent case "
            f"({recorded.recorded_aeroacoustic_solver_start_time})",
        )

    return params


def _check_unique_surface_volume_probe_names(params):
    if not params.outputs:
        return params

    active_probe_names = set()

    for output_index, output in enumerate(params.outputs):
        if isinstance(output, ProbeOutputTypes):
            if output.name in active_probe_names:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    f"Output name {output.name} has already been used by another probe "
                    f"output. Output names must be unique across {PROBE_OUTPUT_TYPE_NAMES}."
                )
            active_probe_names.add(output.name)

    return params


def _check_unique_force_distribution_output_names(params):
    if not params.outputs:
        return params

    active_names = set()

    for output_index, output in enumerate(params.outputs):
        if isinstance(output, (ForceDistributionOutput, TimeAverageForceDistributionOutput)):
            if output.name in active_names:
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: "
                    f"Output name {output.name} has already been used for a `ForceDistributionOutput`. "
                    "Output names must be unique among all force distribution outputs."
                )
            active_names.add(output.name)

    return params


def _check_uniform_averaging_reference_frame(params):
    """Instances that share one solver-side accumulator must agree on the averaging reference frame:
    TimeAverageForceDistributionOutput shares per-wall accumulators, TimeAverageVolumeOutput
    shares the volume accumulator, TimeAverageSliceOutput shares the slice accumulator."""
    if not params.outputs:
        return params

    for output_class, shared_accumulator in (
        (TimeAverageForceDistributionOutput, "one averaged result per wall"),
        (TimeAverageVolumeOutput, "one averaged volume result"),
        (TimeAverageSliceOutput, "one averaged result per slice in a merged config section"),
    ):
        frames = {output.averaging_reference_frame for output in params.outputs if isinstance(output, output_class)}
        if len(frames) > 1:
            raise ValueError(
                f"All `{output_class.__name__}` instances must use the same "
                f"`averaging_reference_frame`; found {sorted(frames)}. They share "
                f"{shared_accumulator}, which can only be averaged in one frame."
            )

    return params


def _check_unique_surface_volume_probe_entity_names(params):
    if not params.outputs:
        return params

    for output_index, output in enumerate(params.outputs):
        if not isinstance(output, ProbeOutputTypes):
            continue
        # Compared as monitor keys rather than entity names: an entity standing for
        # several points expands into several `monitorLocations` keys, so `Point("Rake_0")`
        # and `PointArray("Rake")` carry distinct names and still write the same key,
        # of which the solver would keep only one.
        entity_producing = {}
        for entity in output.entities.stored_entities:
            for monitor_key in entity._monitor_key_names:
                if monitor_key not in entity_producing:
                    entity_producing[monitor_key] = entity.name
                    continue
                previous_entity_name = entity_producing[monitor_key]
                if previous_entity_name == entity.name:
                    raise ValueError(
                        f"In `outputs`[{output_index}] {output.output_type}: "
                        f"Entity name {entity.name} has already been used in the "
                        f"same `{output.output_type}`. Entity names must be unique."
                    )
                raise ValueError(
                    f"In `outputs`[{output_index}] {output.output_type}: monitor `{monitor_key}` "
                    f"is produced by both entity `{previous_entity_name}` and entity "
                    f"`{entity.name}`, so they would share one monitor. Please rename one of them."
                )

    return params


# Asked of every entity of every output, including the boundary lists that reach 140k.
_is_file_named_entity = subclass_predicate((Slice, Isosurface))


def _file_named_by(output):
    """(namespace, name) for each name this output turns into an output file name. The namespace
    is the file name pattern the name shares, so names in different namespaces cannot collide.
    An output named by its own `name` rather than by its entities has to be listed explicitly."""
    if isinstance(output, (*ProbeOutputTypes, SurfaceIntegralOutput)):
        return [("monitor", output.name)]  # all write `monitor_<name>_v2.csv`
    if isinstance(output, ForceDistributionOutput):
        return [("forceDistribution", output.name)]
    if isinstance(output, RenderOutput):
        return [("render", output.name)]  # writes `render_<name>[_time_<n>].png`
    # The rest are named by their entities. Entities that do not reach a file name (notably
    # boundaries) are excluded: the solver matches those against mesh patch names, which may
    # contain `/`.
    entities = getattr(output, "entities", None)
    items = getattr(entities, "stored_entities", None) or getattr(entities, "items", None) or []
    if isinstance(output, StreamlineOutput):
        # Streamline seed file names are based on the seed type and name, so seeds of one type can
        # collide; the solver merges same-key seeds into a single monitor group. The translator
        # writes a PointGroup's points as Point seeds, so the two share file names.
        return [
            (f"{output.output_type} {'Point' if isinstance(item, PointGroup) else type(item).__name__}", item.name)
            for item in items
        ]
    return [(output.output_type, item.name) for item in items if _is_file_named_entity(item)]


def _check_output_names_usable_in_file_names(params):
    """Report the path separators the translator rewrites in output file names (see
    `sanitize_file_name`), and reject names that differ only by a rewritten character, since
    those would resolve to a single output file."""

    claimed: dict[tuple[str, str], str] = {}
    for output in params.outputs or []:
        for namespace, name in _file_named_by(output):
            safe_name = sanitize_file_name(name)
            if safe_name != name:
                add_validation_warning(
                    f"`{name}` cannot be used in a file name; the output files for it will be "
                    f"named after `{safe_name}` instead."
                )
            claimed_by = claimed.setdefault((namespace, safe_name), name)
            if claimed_by != name:
                raise ValueError(
                    f"`{name}` and `{claimed_by}` both become `{safe_name}` when used in a file "
                    "name, so they would share one output file. Please rename one of them."
                )
    return params


def _check_moving_statistic_applicability(params):
    if not params.time_stepping:
        return params

    if not params.outputs:
        return params

    is_steady = isinstance(params.time_stepping, Steady)
    max_steps = params.time_stepping.max_steps if is_steady else params.time_stepping.steps

    for output_index, output in enumerate(params.outputs):
        if not hasattr(output, "moving_statistic") or output.moving_statistic is None:
            continue
        moving_window_size_in_step = (
            output.moving_statistic.moving_window_size * 10 if is_steady else output.moving_statistic.moving_window_size
        )
        start_step = (
            math.ceil(output.moving_statistic.start_step / 10) * 10 if is_steady else output.moving_statistic.start_step
        )
        if moving_window_size_in_step + start_step > max_steps:
            raise customize_model_validator_error(
                model_instance=params,
                relative_location=("outputs", output_index, "moving_statistic"),
                message="`moving_statistic`'s moving_window_size + start_step exceeds "
                "the total number of steps in the simulation.",
                input_value=output.moving_statistic.model_dump(),
            )

    return params
