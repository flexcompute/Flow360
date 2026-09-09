"""utils for cloud operations"""

import re
from enum import Enum

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

from flow360.exceptions import Flow360ValueError


class _S3Action(Enum):
    """
    Enum for s3 action
    """

    UPLOADING = "[bold red]↑"
    DOWNLOADING = "[bold green]↓"
    COMPRESSING = "[cyan]Compressing..."
    NONE = ""


def _valid_resource_id(resource_id) -> bool:
    """
    Returns:
    1. Whether the resource_id is valid
    2. The content of the resource_id
    """
    if not isinstance(resource_id, str):
        raise ValueError(f"resource_id must be a string, but got {type(resource_id)}")

    pattern = re.compile(
        r"""
        ^                     # Start of the string
        ROOT\.FLOW360|        # accept root folder
        (?P<content>          # Start of the content group
        [0-9a-zA-Z,-]{16,}    # Content: at least 16 characters, alphanumeric, comma, or dash
        )$                    # End of the string
        """,
        re.VERBOSE,
    )

    match = pattern.match(resource_id)
    if not match:
        return False

    return True


# pylint: disable=redefined-builtin
def is_valid_uuid(id, allow_none=False):
    """
    Checks if id is valid
    """

    if id is None:
        if allow_none is True:
            return
        raise Flow360ValueError("None is not a valid id.")

    try:
        is_valid = _valid_resource_id(id)
        if is_valid is False:
            raise ValueError(f"{id} is not a valid UUID.")
    except ValueError as exc:
        raise Flow360ValueError(f"{id} is not a valid UUID.") from exc


def _get_progress(action: _S3Action = _S3Action.NONE):
    # if not action == _S3Action.NONE:
    #     description = TextColumn(action.value)
    # else:
    #     description = TextColumn("[text.bold.green]{task.fields[description]}"),

    return Progress(
        (
            TextColumn("{task.description}")
            if action == _S3Action.NONE
            else TextColumn(action.value)
        ),
        TextColumn("[bold blue]{task.fields[filename]}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.1f}%",
        "•",
        DownloadColumn(),
        "•",
        TransferSpeedColumn(),
        "•",
        TimeRemainingColumn(),
    )
