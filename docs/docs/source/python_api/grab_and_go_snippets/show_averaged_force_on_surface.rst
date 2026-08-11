.. _python_api_show_averaged_force_on_surface:

.. currentmodule:: flow360

*************************************
Get Averaged Forces from a Boundary
*************************************

This example demonstrates how to retrieve lift and drag coefficients from a specific boundary and average them over the last 10% of iterations.

.. literalinclude:: _snippets/show_averaged_force_on_surface.py
   :language: python

Notes
=====

- Use ``Case.from_cloud(case_id="...")`` to retrieve a completed case from the cloud.
- The ``surface_forces`` result contains force coefficients broken down by boundary.
- Use ``filter(include="...")`` to select specific boundaries or ``filter(exclude="...")`` to exclude them. Both support wildcard patterns (e.g., ``"*wing*"``).
- The ``get_averages(fraction)`` method computes the average over the last ``fraction`` of pseudo steps (0.1 = last 10%).
- Available force coefficients include: ``CL``, ``CD``, ``CFx``, ``CFy``, ``CFz``, ``CMx``, ``CMy``, ``CMz``, and their pressure/skin friction components.
