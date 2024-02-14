import os

import pytest

from flow360.cloud.s3_utils import S3TransferType
from flow360.exceptions import Flow360ValueError


def test_file_download():
    with pytest.raises(Flow360ValueError):
        S3TransferType.CASE.download_file("id", "file", to_file="to_file", to_folder="to_folder")
