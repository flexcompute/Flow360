"""Enforce PEP 440 version specifiers in flow360 runtime dependencies.

Poetry caret/tilde (``^``, ``~``) are not PEP 440. The schema-inline merge
intersects the client and schema dependency ranges with
``packaging.SpecifierSet``, which only understands PEP 440 ``>=,<`` ranges,
so the source must stay caret/tilde-free.

Usage:
    python tools/check_pep440_deps.py   # exit 1 if any ^/~ in runtime deps
"""

import re
import sys
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"
# Poetry caret (^x) and Poetry tilde (~x). PEP 440's compatible-release operator
# (~=x) is valid and must NOT be flagged, hence the negative lookahead on `~`.
POETRY_OPERATOR_RE = re.compile(r"\^|~(?!=)")


def _load_toml(path):
    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib
        except ModuleNotFoundError:
            import toml

            return toml.load(path)
    with open(path, "rb") as handle:
        return tomllib.load(handle)


def _version_specs(value):
    """Yield every version specifier string a dependency entry carries.

    Bare string -> the string. Table -> its ``version`` (path/git/url tables
    have none). Array of tables -> each member's ``version``. The ``python``
    marker inside a table is a selector, not a version range, so it is skipped.
    """
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        if "version" in value:
            yield value["version"]
    elif isinstance(value, list):
        for member in value:
            if isinstance(member, dict) and "version" in member:
                yield member["version"]


def main():
    dependencies = _load_toml(PYPROJECT)["tool"]["poetry"]["dependencies"]
    violations = [
        (name, spec)
        for name, value in dependencies.items()
        if name != "python"
        for spec in _version_specs(value)
        if POETRY_OPERATOR_RE.search(spec)
    ]
    if violations:
        print(f"{PYPROJECT.name}: Poetry caret/tilde found in runtime dependencies.")
        print("Use explicit PEP 440 ranges instead (e.g. ^3.0.0 -> >=3.0.0,<4.0.0):")
        for name, spec in violations:
            print(f'  {name} = "{spec}"')
        sys.exit(1)
    print(f"{PYPROJECT.name}: runtime dependencies are PEP 440 (no ^/~).")


if __name__ == "__main__":
    main()
