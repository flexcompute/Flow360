import os

import pytest

from flow360.cloud.s3_utils import S3TransferType
from flow360.exceptions import ValueError


def test_file_download():
    with pytest.raises(ValueError):
        S3TransferType.CASE.download_file("id", "file", to_file="to_file", to_folder="to_folder")
