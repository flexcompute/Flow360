import pytest

from flow360 import Accounts, Env

from .mock_server import mock_response


def test_shared_account(mock_response, monkeypatch):
    Accounts.choose_shared_account("user1@test.com")

    assert Accounts.shared_account_info() == "user1@test.com"
    assert Env.impersonate == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    assert not Accounts.shared_account_submit_is_confirmed()

    Accounts.shared_account_confirm_submit()

    assert Accounts.shared_account_submit_is_confirmed()

    Accounts.leave_shared_account()

    assert Accounts.shared_account_info() is None
    assert Env.impersonate is None

    monkeypatch.setattr('builtins.input', lambda _: "0")

    assert Accounts.shared_account_submit_is_confirmed()

    Accounts.choose_shared_account()

    assert not Accounts.shared_account_submit_is_confirmed()

    assert Accounts.shared_account_info() == "user1@test.com"
    assert Env.impersonate == "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"

    Accounts.leave_shared_account()
    Accounts.leave_shared_account()

    assert Accounts.shared_account_info() is None
    assert Env.impersonate is None

    Accounts.choose_shared_account("user1@test.com")

    assert Accounts.shared_account_info() == "user1@test.com"

    Env.impersonate = "cccccccc-cccc-cccc-cccc-cccccccccccc"

    assert Accounts.shared_account_info() is None

    monkeypatch.setattr('builtins.input', lambda _: "q")

    with pytest.raises(ValueError):
        Accounts.choose_shared_account()

    with pytest.raises(ValueError):
        Accounts.choose_shared_account("user5@test.com")

    Accounts.leave_shared_account()
