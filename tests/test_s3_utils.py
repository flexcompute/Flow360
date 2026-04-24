import pytest

from flow360.cloud import s3_utils
from flow360.cloud.s3_utils import S3TransferType
from flow360.exceptions import Flow360ValueError


def test_file_download():
    with pytest.raises(Flow360ValueError):
        S3TransferType.CASE.download_file("id", "file", to_file="to_file", to_folder="to_folder")


def test_read_text_uses_range_and_decodes(monkeypatch):
    token = s3_utils._S3STSToken.parse_obj(  # pylint: disable=protected-access
        {
            "cloudpathPrefix": "s3://bucket/prefix",
            "cloudpath": "s3://bucket/prefix/logs/file.log",
            "userCredentials": {
                "accessKeyId": "key",
                "secretAccessKey": "secret",
                "sessionToken": "token",
                "expiration": "2999-01-01T00:00:00Z",
                "region": "us-east-1",
            },
        }
    )

    class _Body:
        def read(self):
            return b"hello"

    class _Client:
        def get_object(self, **kwargs):
            assert kwargs["Range"] == "bytes=-6"
            return {"Body": _Body(), "ContentLength": 5}

    monkeypatch.setattr(s3_utils._S3STSToken, "get_client", lambda self: _Client())  # pylint: disable=protected-access
    monkeypatch.setattr(S3TransferType.CASE, "_get_s3_sts_token", lambda *_args, **_kwargs: token)

    text, metadata = S3TransferType.CASE.read_text("case-id", "logs/file.log", byte_range=(-6, None))

    assert text == "hello"
    assert metadata["body_length"] == 5


def test_get_file_size(monkeypatch):
    token = s3_utils._S3STSToken.parse_obj(  # pylint: disable=protected-access
        {
            "cloudpathPrefix": "s3://bucket/prefix",
            "cloudpath": "s3://bucket/prefix/logs/file.log",
            "userCredentials": {
                "accessKeyId": "key",
                "secretAccessKey": "secret",
                "sessionToken": "token",
                "expiration": "2999-01-01T00:00:00Z",
                "region": "us-east-1",
            },
        }
    )

    class _Client:
        def head_object(self, **kwargs):
            assert kwargs["Bucket"] == "bucket"
            assert kwargs["Key"] == "prefix/logs/file.log"
            return {"ContentLength": 123}

    monkeypatch.setattr(s3_utils._S3STSToken, "get_client", lambda self: _Client())  # pylint: disable=protected-access
    monkeypatch.setattr(S3TransferType.CASE, "_get_s3_sts_token", lambda *_args, **_kwargs: token)

    assert S3TransferType.CASE.get_file_size("case-id", "logs/file.log") == 123
