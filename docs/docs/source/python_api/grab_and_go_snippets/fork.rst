.. _python_api_fork:

.. currentmodule:: flow360

*****************
Fork a Case
*****************

This example demonstrates the procedure for continue from an existing simulation case stored in the cloud.
It shows how to retrieve a parent case, modify specific simulation parameters (in this instance, the angle of attack),
and subsequently initiate a new case based on these adjusted parameters while referencing the original case.

.. literalinclude:: _snippets/fork.py
   :language: python

Notes
-----

- Existing projects and cases are accessed via ``Project.from_cloud`` and ``Case`` respectively.
- Simulation parameters are retrieved from the parent case using ``parent_case.params`` and subsequently modified.
- The ``fork_from`` argument within ``project.run_case`` specifies the parent case from which the new simulation is derived.
