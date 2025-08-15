"""utils for cloud operations"""

from enum import Enum

from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)


class _S3Action(Enum):
    """
    Enum for s3 action
    """

    UPLOADING = "[bold red]↑"
    DOWNLOADING = "[bold green]↓"
    COMPRESSING = "[cyan]Compressing..."
    NONE = ""


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
