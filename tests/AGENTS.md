# Testing Guidelines

## Framework & Configuration

- **Framework:** pytest. No `unittest.TestCase`.
- **Config:** `tests/pytest.ini` — all warnings treated as errors except specific known deprecations.
- **Fixtures:** root `conftest.py` registers plugins via `pytest_plugins = ["tests.utils", "tests.mock_server"]`.
- V2 and V1 tests **must be run separately** (different conftest setups conflict):
  - V2: `poetry run pytest -rA tests/simulation -vv`
  - V1: `poetry run pytest -rA --ignore tests/simulation -vv`

## File & Function Naming

- Test files: `test_<topic>.py`
- Test functions: `test_<descriptive_name>` in `snake_case`
- Test classes (rare, for logical grouping only): `TestDescriptiveName`
- Updater tests use version-stamped names: `test_updater_to_<version>_<description>`

## Fixtures

### Autouse Patterns

Most test files define a `change_test_dir` fixture to enable relative paths to `data/` folders:

```python
@pytest.fixture(autouse=True)
def change_test_dir(request, monkeypatch):
    monkeypatch.chdir(request.fspath.dirname)
```

### Shared Fixtures

Root `conftest.py` provides:
- `mock_validation_context` / `mock_case_validation_context` — validation context stubs
- `mock_geometry` / `mock_surface_mesh` / `mock_volume_mesh` — local-storage-backed resource mocks

`tests/simulation/conftest.py` provides:
- `array_equality_override` — patches unyt array equality for test comparisons

### Inline Fixtures

Define test-specific fixtures inline in the test file. Keep them near the tests that use them:

```python
@pytest.fixture()
def constant_variable():
    return UserVariable(name="constant_variable", value=10)
```

## Mocking

### Primary: monkeypatch

Use pytest `monkeypatch` for most mocking needs:

```python
monkeypatch.setattr(http_util, "api_key_auth", lambda: {...})
monkeypatch.setattr(http, "session", MockRequests())
```

### API Mocking: MockResponse + mock_server

`tests/mock_server.py` provides the `mock_response` fixture:
- `MockResponse` classes load JSON from `tests/data/mock_webapi/`
- URL-to-response routing via `GET_RESPONSE_MAP` / `POST_RESPONSE_MAP`
- The `mock_response` fixture patches `http.session` and `http_util.api_key_auth`

### S3 Mocking

`tests/utils.py` provides `s3_download_override` fixture that redirects S3 downloads to local test data.

### unittest.mock.patch

Use sparingly, only when `monkeypatch` cannot handle the scenario (e.g., capturing call arguments with `patch`):

```python
from unittest.mock import patch

with patch("flow360.component.project.set_up_params_for_uploading", mock_fn):
    project.run_case(...)
```

## Assertions

- Plain `assert` statements (no `self.assertEqual`)
- `pytest.raises(ExceptionType, match=...)` for exception testing
- `capsys.readouterr()` for stdout/stderr capture
- `pytest.mark.parametrize` for data-driven test variants
- `pytest.mark.usefixtures("fixture_name")` to apply fixtures without injection

```python
with pytest.raises(Flow360ValueError, match=error_msg):
    project.get_case(asset_id=query_id)
```

## Test Utilities

`tests/utils.py` provides reusable helpers:

| Function | Purpose |
|---|---|
| `to_file_from_file_test(obj)` | Serialize → deserialize → assert equality |
| `compare_to_ref(obj, ref_path)` | Serialize and compare against reference file |
| `compare_dict_to_ref(data, ref_path)` | Compare dict against JSON reference |
| `file_compare(file1, file2)` | Unified diff comparison |

## Test Data Organization

```
tests/data/
├── mock_webapi/                # JSON responses for API mocking
├── <resource-id>/              # Per-resource local storage data
└── simulation/                 # Simulation JSON fixtures for updater tests

tests/simulation/data/          # Simulation-test-specific input data
tests/simulation/ref/           # Reference output files for comparison
tests/simulation/service/data/  # Service-layer test data
```

Reference files in `ref/` directories are used with `compare_to_ref()` for regression testing. When adding new test cases, create corresponding reference files.

## Import Conventions in Tests

```python
import flow360 as fl
from flow360 import SimulationParams, u, math, SI_unit_system
from flow360.component.simulation.services import validate_model
from flow360.exceptions import Flow360ValueError
from tests.utils import compare_to_ref, to_file_from_file_test
```

## Quick Commands

```sh
# Run all V2 tests
poetry run pytest -rA tests/simulation -vv

# Run a specific test file
poetry run pytest -rA tests/simulation/params/test_expressions.py -vv

# Run a specific test by keyword
poetry run pytest -rA -k "test_expression_operators" -vv

# Fast run (no coverage, mute warnings, fail fast)
poetry run pytest -rA tests/simulation --no-cov -W ignore --maxfail=1

# Coverage report
poetry run pytest -rA tests/simulation --cov-report=html --cov=flow360/component/simulation
```

_Update this AGENTS.md when testing conventions or infrastructure change._
