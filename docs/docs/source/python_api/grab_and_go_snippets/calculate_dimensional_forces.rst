.. _python_api_calculate_dimensional_forces:

.. currentmodule:: flow360

****************************
Calculate Dimensional Forces
****************************

Two snippets will demonstrate how to obtain dimensional forces in the following scenarios:

- total forces on the wall boundaries
- forces excluding selected boundaries

Total Forces on the Wall Boundaries
-----------------------------------

.. literalinclude:: _snippets/calculate_dimensional_forces_total.py
   :language: python


Forces Excluding Selected Boundaries
------------------------------------

.. literalinclude:: _snippets/calculate_dimensional_forces_excluded.py
   :language: python

.. note::

   The ``filter()`` method also supports an ``include`` argument for selecting only specific boundaries
   (e.g., ``forces_by_surface.filter(include="*wing*")``). Both ``include`` and ``exclude`` accept
   wildcard patterns.
