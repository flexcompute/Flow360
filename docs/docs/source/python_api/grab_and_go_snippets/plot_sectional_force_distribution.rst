.. _python_api_plot_sectional_force_distribution:

.. currentmodule:: flow360

**********************************
Plot Sectional Force Distribution
**********************************

This example demonstrates how to extract the sectional force distribution from a completed case and report it as a **dimensional** quantity: the cumulative drag build-up along the X axis and the spanwise load distribution along the Y axis. The solver writes these distributions as coefficients, which are scaled by the dynamic pressure and reference area (``q * A``) to recover dimensional values.

.. literalinclude:: _snippets/plot_sectional_force_distribution.py
   :language: python

.. figure:: Figures/sectional_force_distribution.png
   :width: 100%
   :align: center

   Example output: the dimensional cumulative drag along X and the spanwise lift loading along Y.

Notes
=====

- Use ``Case.from_cloud(case_id="...")`` to retrieve a completed case from the cloud.
- ``x_slicing_force_distribution`` gives the cumulative drag coefficient along X; ``y_slicing_force_distribution`` gives the per-span force and moment coefficients along Y.
- The raw distributions are non-dimensional. Multiply by ``q * A`` (with ``q = 0.5 * density * reference_velocity**2`` and ``A = reference_geometry.area``) to dimensionalize, following the same scaling as :doc:`Calculate Dimensional Forces <calculate_dimensional_forces>`.
- Mind the difference in what each distribution represents. ``totalCumulative_CD_Curve`` is an *integrated* (cumulative) drag coefficient, so scaling it gives a force in Newtons. ``totalCFz_per_span`` is a *per-span* coefficient, so scaling it gives a sectional loading (force per unit span, N/m), not a total force.
- Sectional distributions are produced by post-processing, which can finish after the case has converged. Call ``wait()`` before reading the data.
- ``filter(include="...")`` and ``filter(exclude="...")`` restrict the distribution to a subset of surfaces (both support wildcard patterns such as ``"*Wing*"`` and explicit surface names).
- The X distribution column is ``totalCumulative_CD_Curve``; the Y distribution columns are ``totalCFx_per_span``, ``totalCFz_per_span`` and ``totalCMy_per_span``.
