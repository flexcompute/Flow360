"""
Flow360 plugin namespace.

Report-related functionality has moved to the standalone `flow360-report` package.
Install it via `pip install flow360-report` if you need reporting utilities.
"""

from typing import Any

_REPORT_HINT = (
    "flow360 reporting utilities now live in the `flow360-report` package. "
    "Install it via `pip install flow360-report` and import from `flow360_report.plugins.report`."
)


def __getattr__(name: str) -> Any:
    if name == "report":
        raise ImportError(_REPORT_HINT)
    raise AttributeError(f"module 'flow360.plugins' has no attribute '{name}'")


__all__: list[str] = []
