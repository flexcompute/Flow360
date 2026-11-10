.. _python_api_change_account:

.. currentmodule:: flow360

**************************
Change Account and Submit
**************************

This example demonstrates how to utilize the shared account feature within the Flow360 Python API.
It covers the process of interactively selecting a shared account, submitting a pre-existing volume mesh
to initiate a project under that account, and subsequently reverting to the original account context.

.. literalinclude:: _snippets/change_account.py
   :language: python

Notes
=====

- ``Accounts.leave_shared_account()`` is used to exit the shared account context and return to the user's primary account.
