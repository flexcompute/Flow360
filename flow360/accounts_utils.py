"""
This module provides utility functions for managing access between interconnected accounts.

Functions:
- choose_shared_account(None) -> None - select account from the list of client and organization accounts interactively
- choose_shared_account(email: str, optional) -> None - select account matching the provided email (if exists)
- shared_account_info(None) -> str - return current shared account email address (if exists, None otherwise)
- leave_shared_account(None) -> None - leave current shared account
"""

from requests import HTTPError

from flow360.cloud.http_util import http
from flow360.environment import Env
from flow360.log import log

from .exceptions import Flow360WebError


class AccountsUtils:
    """
    Current account info and utility functions.
    """

    def __init__(self):
        self._current_email = None
        self._current_user_identity = None
        self._confirmed_submit = False

    @staticmethod
    def _interactive_selection(users):
        print(
            "Choosing account in interactive mode, please select email from the organization list: "
        )

        user_count = len(users)

        for i in range(0, user_count):
            print(f"{i}: {users[i]['userEmail']}")

        while True:
            try:
                value = input(
                    f"Enter address of the account to switch to [0 - {user_count - 1}] or 'q' to abort: "
                )
                if value == "q":
                    return None
                if int(value) in range(0, user_count):
                    return int(value)
                print(f"Value out of range [0 - {user_count - 1}]")
                continue
            except ValueError:
                print("Invalid input type, please input an integer value:")
                continue

    @staticmethod
    def _get_supported_users():
        try:
            response = http.portal_api_get("auth/credential")
            supported_users = response["guestUsers"]
            if supported_users:
                for entry in supported_users:
                    entry["userEmail"] = entry.pop("email")
                    entry["userIdentity"] = entry.pop("identity")
                return supported_users
            return []
        except HTTPError as error:
            raise Flow360WebError("Failed to retrieve supported user data from server") from error

    @staticmethod
    def _get_company_users():
        try:
            response = http.get("flow360/account")
            company_users = response["tenantMembers"]
            if company_users:
                return company_users
            return []
        except HTTPError as error:
            raise Flow360WebError("Failed to retrieve company user data from server") from error

    def _check_state_consistency(self):
        if Env.impersonate != self._current_user_identity:
            log.warning(
                f"Environment impersonation ({Env.impersonate}) does "
                f"not match current account ({self._current_user_identity}), "
                "this may be caused by explicit modification of impersonation "
                "in the environment, use choose_shared_account() instead."
            )
            self._current_email = None
            self._current_user_identity = None

    def shared_account_submit_is_confirmed(self):
        """check if the user confirmed that he wants to submit resources to a shared account"""
        return self._confirmed_submit

    def shared_account_confirm_submit(self):
        """confirm submit for the current session"""
        self._confirmed_submit = True

    def choose_shared_account(self, email=None):
        """choose a shared account to impersonate

        Parameters
        ----------
        email : str, optional
           user email to impersonate (if email exists among shared accounts),
           if email is not provided user can select the account interactively
        """
        shared_accounts = self._get_company_users() + self._get_supported_users()

        shared_accounts.sort(key=lambda user: user["userEmail"])

        if len(shared_accounts) == 0:
            log.info("There are no accounts shared with the current user")
            return

        selected = None

        addresses = [user["userEmail"] for user in shared_accounts]

        if email is None:
            selected = self._interactive_selection(shared_accounts)
        elif email in addresses:
            selected = addresses.index(email)

        if selected is None:
            raise ValueError("Invalid or empty email address selected, cannot change account.")

        user_email = shared_accounts[selected]["userEmail"]
        user_id = shared_accounts[selected]["userIdentity"]

        Env.impersonate = user_id

        self._confirmed_submit = False
        self._current_email = user_email
        self._current_user_identity = user_id

    def shared_account_info(self):
        """
        retrieve current shared account name, if possible
        """
        self._check_state_consistency()

        return self._current_email

    def leave_shared_account(self):
        """
        leave current shared account name, if possible
        """
        self._check_state_consistency()

        if Env.impersonate is None:
            log.warning("You are not currently logged into any shared account")
        else:
            log.info(f"Leaving shared account {self._current_email}")
            self._current_email = None
            self._current_user_identity = None
            Env.impersonate = None


Accounts = AccountsUtils()
