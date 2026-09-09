"""Simulation services function for report (legacy)."""

from flow360.plugins.report.report import get_default_report_summary_template


def get_default_report_config() -> dict:
    """
    Get the default report config
    Returns
    -------
    dict
        default report config
    """
    return get_default_report_summary_template().model_dump(
        exclude_none=True,
    )
