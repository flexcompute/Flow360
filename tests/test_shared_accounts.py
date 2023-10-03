import pytest

from flow360 import Accounts, Env

from .mock_server import mock_response


def test_shared_account(mock_response):
    Accounts.choose_shared_account("user1@test.com")

    assert Accounts.shared_account_info() == "user1@test.com"
    assert Env.impersonate == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    Accounts.leave_shared_account()

    assert Accounts.shared_account_info() is None
    assert Env.impersonate is None

    with pytest.raises(ValueError):
        Accounts.choose_shared_account("user3@test.com")
