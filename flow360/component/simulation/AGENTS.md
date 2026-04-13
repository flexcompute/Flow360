# Simulation Module Guidelines

This module (`flow360/component/simulation/`) is the V2 simulation framework. It defines the user-facing configuration API, validation pipeline, and translation to solver-native JSON.

## Architecture

| Directory | Purpose |
|---|---|
| `framework/` | Base classes, config, entity system, updater |
| `models/` | Physical models (materials, surface/volume models, solver numerics) |
| `meshing_param/` | Meshing parameter definitions (edge, face, volume, snappy) |
| `operating_condition/` | Operating condition definitions |
| `outputs/` | Output/monitor definitions |
| `time_stepping/` | Time stepping configuration |
| `run_control/` | Run control settings |
| `translator/` | SimulationParams → solver JSON conversion |
| `validation/` | Context-aware validation pipeline |
| `blueprint/` | Safe function/expression serialization and dependency resolution |
| `user_code/` | User-defined expressions, variables, and code |
| `user_defined_dynamics/` | User-defined dynamics definitions |
| `migration/` | V1 → V2 migration utilities |
| `web/` | Web/cloud integration utilities |
| `services.py` | Service facade (validate, translate, convert) |
| `simulation_params.py` | Top-level `SimulationParams` model |
| `primitives.py` | Entity types (Surface, Volume, Edge, Box, Cylinder, etc.) |
| `units.py` | Unit system definitions |

## Base Class Hierarchy

```
pd.BaseModel
  └── Flow360BaseModel                      # framework/base_model.py
        ├── EntityBase (ABCMeta)            # framework/entity_base.py
        │     ├── _VolumeEntityBase
        │     │     ├── GenericVolume, Box, Cylinder, Sphere, ...
        │     └── _SurfaceEntityBase
        │           ├── Surface, GhostSurface, MirroredSurface, ...
        ├── _ParamModelBase
        │     └── SimulationParams
        ├── BoundaryBase (ABCMeta)          # models/surface_models.py
        │     └── Wall, Freestream, Inflow, Outflow, ...
        └── SingleAttributeModel            # framework/base_model.py
```

### Flow360BaseModel

All models inherit from `Flow360BaseModel`. It provides:
- JSON/YAML file I/O (`from_file`, `to_file`)
- SHA-256 integrity hashing
- Recursive `preprocess()` for non-dimensionalization
- `require_one_of` / `conflicting_fields` mutual exclusion constraints
- `validate_conditionally_required_field` for pipeline-stage-aware required fields

### EntityBase

Abstract base for all simulation entities. Key attributes:
- `name` (frozen, immutable identifier)
- `private_attribute_entity_type_name` (discriminator for serialization)
- `private_attribute_id` (UUID for tracking)

Concrete entities are decorated with `@final` to prevent subclassing.

### EntityList and EntitySelector

- `EntityList[Surface, GhostSurface]` — generic container supporting direct entities and rule-based selectors
- `EntitySelector` — rule-based entity selection with glob/regex predicates
- Selectors are lazily expanded via `ParamsValidationInfo.expand_entity_list()`

### EntityRegistry

Central registry holding all entity instances. Populated from `EntityInfo` metadata. Supports type-filtered views and glob pattern access.

## Key Patterns

### Unit System Context

All dimensioned construction must happen inside a `UnitSystem` context manager:

```python
with SI_unit_system:
    params = SimulationParams(
        reference_geometry=ReferenceGeometry(
            moment_center=(1, 2, 1) * u.m,
            area=1.5 * u.m**2,
        ),
        ...
    )
```

When loading from file/dict, the unit system is auto-detected from the serialized data.

### Private Attribute Convention

Attributes prefixed with `private_attribute_` are pydantic fields (not `pd.PrivateAttr`). They participate in serialization but are considered internal to the framework. Common examples:
- `private_attribute_entity_type_name` — type discriminator
- `private_attribute_id` — generated UUID
- `private_attribute_zone_boundary_names` — mesh zone names

### Discriminated Unions

Polymorphic types use Pydantic discriminated unions:

```python
ModelTypes = Annotated[
    Union[VolumeModelTypes, SurfaceModelTypes],
    pd.Field(discriminator="type"),
]
```

Each variant has a `type` literal field (e.g., `type: Literal["Wall"] = pd.Field("Wall", frozen=True)`).

### Field Constructors

`SimulationParams` uses specialized field constructors from `validation_context.py`:
- `pd.Field()` — always present
- `CaseField()` — relevant only for solver case configuration
- `ConditionalField(context=[SURFACE_MESH, VOLUME_MESH])` — required only during specific pipeline stages

### Validators

Two categories of validators:

1. **Standard Pydantic** — always run:
   - `@pd.field_validator("field", mode="after")` with `@classmethod`
   - `@pd.model_validator(mode="after")`

2. **Contextual** — run only when a `ValidationContext` is active:
   - `@contextual_field_validator(...)` — wraps field validators, auto-skips outside context
   - `@contextual_model_validator(...)` — wraps model validators, can inject `param_info`

Extract complex validation logic into standalone functions in `validation/validation_simulation_params.py` or `validation/validation_output.py`. Keep the model class as a thin declarative shell.

## Validation Pipeline

Validation is context-aware, using `contextvars` to track the pipeline stage:

| Level | Constant | When |
|---|---|---|
| Surface meshing | `SURFACE_MESH` | Generating surface mesh params |
| Volume meshing | `VOLUME_MESH` | Generating volume mesh params |
| Case solve | `CASE` | Generating solver params |
| All stages | `ALL` | Full validation |

The `services.validate_model()` function orchestrates: updater → sanitize → materialize entities → initialize variables → contextual validation.

## Translation

Translators in `translator/` convert `SimulationParams` to solver-native JSON. The pipeline:

1. `SimulationParams._preprocess()` — non-dimensionalize physical units
2. Translator function (`get_solver_json()`, `get_surface_meshing_json()`, etc.) — map to flat JSON
3. Output JSON dict suitable for solver consumption

Translators are pure functions. Do not add `to_solver()` methods on model classes.

## Version Migration

The `framework/updater.py` system applies incremental dict transforms to migrate older `simulation.json` files to the current schema. Per-version functions follow the naming pattern `_to_<version>()` (e.g., `_to_25_2_0()`). The `_ParamModelBase._update_param_dict()` method orchestrates migration automatically on load.

When making breaking schema changes, add a new migration function to the updater.

## Adding New Features

1. **New model field:** Add to the appropriate model class with `pd.Field()`. Add validators if needed.
2. **New entity type:** Inherit from `_VolumeEntityBase` or `_SurfaceEntityBase`. Decorate with `@final`. Add `private_attribute_entity_type_name` literal discriminator.
3. **New boundary condition:** Inherit from `BoundaryBase`. Add to the `SurfaceModelTypes` union.
4. **New validator:** Prefer `@contextual_field_validator` if it depends on pipeline stage. Extract logic to `validation/` files.
5. **Schema migration:** Add a versioned migration function to `framework/updater.py`.
6. **Translator update:** Add translation logic to the appropriate `translator/*.py` file.

_Update this AGENTS.md when architectural patterns or conventions change._
