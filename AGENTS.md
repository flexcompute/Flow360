# Flow360 Python Client — Repository Guidelines

These guidelines reflect the conventions in the Flow360 Python client. Follow existing patterns in nearby code and these rules when contributing.

## Project Structure

| Directory | Contents |
|---|---|
| `flow360/component/simulation/` | V2 simulation framework (active development) |
| `flow360/component/v1/` | Legacy V1 API (maintenance only) |
| `flow360/cli/` | Click-based CLI (`flow360 configure`, etc.) |
| `flow360/plugins/` | Plugin system (e.g., `report`) |
| `flow360/examples/` | Internal example scripts |
| `examples/` | User-facing example scripts |
| `tests/simulation/` | V2 simulation tests |
| `tests/v1/` | V1 legacy tests |
| `docs/` | Sphinx documentation |

New features go in `flow360/component/simulation/`. Do not add to `flow360/component/v1/` unless fixing a bug.

## Workflow & Tooling

- **Package manager:** Poetry. Prefix every command with `poetry run` to match CI.
- **Setup:** `pip install poetry && poetry install`
- **Pre-commit hooks:** autohooks (configured in `pyproject.toml`). Runs black, isort, and pylint automatically. Install with `autohooks activate`.

### Check-in Procedure

Run these before opening a PR:

```sh
poetry run black .                        # auto-format
poetry run isort .                        # sort imports
poetry run pylint $(git ls-files "flow360/*.py") --rcfile .pylintrc  # lint
poetry run pytest -rA tests/simulation -vv                           # V2 tests
poetry run pytest -rA --ignore tests/simulation -vv                  # V1 tests
```

V1 and V2 tests must be run separately (not together).

## Coding Style

### Formatting

- **Black** with line-length 100, target Python 3.10 (`pyproject.toml [tool.black]`).
- **isort** with `profile = "black"` (`pyproject.toml [tool.isort]`).
- **pylint** with `.pylintrc` (max line-length 120, but black enforces 100).
- Do not reformat or re-indent lines you did not modify.

### Naming

| Element | Convention | Example |
|---|---|---|
| Functions/methods | `snake_case` | `compute_residual`, `get_solver_json` |
| Classes | `PascalCase` | `SimulationParams`, `Flow360BaseModel` |
| Variables/params | `snake_case` | `moment_center`, `mesh_unit` |
| Constants | `UPPER_CASE` | `CASE`, `SURFACE_MESH` |
| Private | leading underscore | `_preprocess`, `_update_param_dict` |
| Modules | `snake_case` | `simulation_params.py`, `surface_models.py` |

### Imports

Standard ordering enforced by isort:

1. Standard library
2. Third-party (pydantic, numpy, etc.)
3. Local/package imports (absolute paths)

Convention aliases:
- `import pydantic as pd`
- `import numpy as np`
- `import flow360.component.simulation.units as u`
- `from flow360.log import log`

Prefer top-level imports. Use lazy imports only to break circular dependencies, and scope them as narrowly as possible.

### Type Annotations

- Use modern typing from Python 3.10+: `Optional[X]`, `Union[X, Y]`, `Literal["value"]`.
- Use `Annotated[Type, pd.Field(...)]` for discriminated unions.
- Use `typing.final` decorator on concrete entity classes.
- Use `typing_extensions.Self` for return types of model validators.

### Docstrings

Numpy-style docstrings with Sphinx `:class:` cross-references. User-facing classes include an `Example` section with `>>>` code and a trailing `====` marker for doc rendering.

```python
class ReferenceGeometry(Flow360BaseModel):
    """
    :class:`ReferenceGeometry` class contains all geometrical related reference values.

    Example
    -------
    >>> ReferenceGeometry(
    ...     moment_center=(1, 2, 1) * u.m,
    ...     moment_length=(1, 1, 1) * u.m,
    ...     area=1.5 * u.m**2
    ... )

    ====
    """
```

Method docstrings use `Parameters`, `Returns`, `Raises`, and `Examples` sections.

## Pydantic Model Conventions

- **Pydantic V2** (`>= 2.8`). Always import as `import pydantic as pd`.
- All models inherit from `Flow360BaseModel` (`flow360.component.simulation.framework.base_model`).
- Model config: `extra="forbid"`, `validate_assignment=True`, camelCase serialization aliases via `alias_generator`.
- Private-by-convention attributes use `private_attribute_` prefix (these are regular pydantic fields, not `pd.PrivateAttr`).
- Discriminated unions use `pd.Field(discriminator="type")` or `pd.Field(discriminator="private_attribute_entity_type_name")`.
- Use `pd.Field()` with `frozen=True` for immutable fields, `description` for documentation, and `alias` for serialization overrides.
- Validator patterns:
  - `@pd.field_validator("field", mode="after")` with `@classmethod`
  - `@pd.model_validator(mode="after")`
  - `@contextual_field_validator(...)` / `@contextual_model_validator(...)` for pipeline-stage-aware validation

## Error Handling & Logging

- Use the custom exception hierarchy rooted at `Flow360Error` (`flow360/exceptions.py`). Every exception auto-logs on `__init__`.
- Common exceptions: `Flow360ValueError`, `Flow360TypeError`, `Flow360RuntimeError`, `Flow360ConfigurationError`, `Flow360ValidationError`.
- Use `Flow360DeprecationError` for deprecated features.
- **Logging:** Use `from flow360.log import log` — a custom `Logger` class backed by `rich.Console`. Do not use Python's stdlib `logging` module.

## Testing

See `tests/AGENTS.md` for detailed testing conventions.

Quick reference:
- **Framework:** pytest (no `unittest.TestCase`)
- **V2 tests:** `poetry run pytest -rA tests/simulation -vv`
- **V1 tests:** `poetry run pytest -rA --ignore tests/simulation -vv`
- **Coverage:** `pytest -rA tests/simulation --cov-report=html --cov=flow360/component/simulation`
- Warnings are treated as errors via `tests/pytest.ini`.

## CI/CD

- **Code style** (`codestyle.yml`): black → isort → pylint (Python 3.10, ubuntu)
- **Tests** (`test.yml`): runs after code style passes; matrix of Python 3.10–3.13 × ubuntu/macOS/Windows; V2 and V1 tests run separately
- **Docs** (`deploy-doc.yml`): Sphinx documentation build and deployment
- **Publishing** (`pypi-publish.yml`): PyPI release workflow

## Documentation

- Sphinx-based docs in `docs/`. Build with `poetry install -E docs` then `make html` in `docs/`.
- Update existing docs before creating new ones.
- Example scripts live in `examples/` (user-facing) and `flow360/examples/` (internal).

_Update this AGENTS.md whenever workflow, tooling, or conventions change._
