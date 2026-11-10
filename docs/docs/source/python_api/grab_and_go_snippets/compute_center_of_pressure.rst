.. _python_api_compute_center_of_pressure:

.. currentmodule:: flow360

******************************
Compute the Center of Pressure
******************************

Flow360 reports the total force and the total moment about
:attr:`~ReferenceGeometry.moment_center`, but no center of pressure. This example
recovers it from those two vectors for any chosen set of surfaces. The
:class:`UserDefinedDynamic` tracks it live while the case runs; the block after
the run gives the single converged number.

.. literalinclude:: _snippets/compute_center_of_pressure.py
   :language: python

Notes
=====

- A load reduces to a force along a **line**, not through a point, so choose which point on that line to quote. ``x_cp, y_cp, z_cp`` is the one closest to the moment center. ``x_cp_plane`` is where the line crosses ``z = z_mc``, which in symmetric flow is the textbook ``x_cp = x_mc - CMy * Ly / CFz``.
- ``couple_arm`` is the part of the moment that no point can cancel. Compare it with a reference length of the geometry: if it is not small, the load does not reduce to a point at all.
- The result diverges as the resultant force goes to zero. Near zero lift, quote the aerodynamic center instead.
- Surfaces are selected in **two** matching places: ``input_boundary_patches`` for the monitor, ``filter()`` for the post-processing. ``show_available_groupings()`` lists the names. To include BET, actuator disk and porous contributions, read ``case.results.total_forces`` instead and drop the ``total`` prefix from the column names.
- Monitor traces are named positionally, ``state[0]`` to ``state[3]``, following the order of the ``update_law`` entries: here ``x_cp``, ``y_cp``, ``z_cp``, ``couple_arm``. They are in grid units, the post-processed values in metres.
- No dynamic pressure or reference area is needed, the ``q * A`` factor cancels in the ratio. See :ref:`force_moment_physical_conversion`.
- For an unsteady case, use ``as_dataframe()`` in place of ``get_averages()`` to get a time history instead of one point.

.. seealso::

   :ref:`python_api_calculate_dimensional_forces` converts the same coefficients
   into forces and moments in newtons.
