.. _legacy_surface_mesher_user_guide:

*********************
Legacy Surface Mesher
*********************

The legacy surface mesher is the initial mesher workflow in Flow360. It will be deprecated in a future release, so it is recommended to use the new surface mesher instead.

Main characteristics:

- requires a watertight geometry at the input,
- generates a spherical or half-spherical automated farfields,
- allows for the generation of quasi-3d meshes.

Examples
========

Examples of mesh generation can be found in the :ref:`Python API Example Library <python_api_example_library>`:

- :doc:`2D CRM airfoil example <../../python_api/example_library/notebooks/2D_CRM_airfoil>`,
- :doc:`DARPA AD example <../../python_api/example_library/notebooks/DARPA_SUBOFF_AD>`.

And in the `WebUI Example Library <https://flow360.simulation.cloud/examples>`_.:

- `Simple airplane example <https://flow360.simulation.cloud/workbench/prj-261fdfa7-025e-40d7-9c5d-52c0f206f782>`_,
- `eVTOL with BET line example <https://flow360.simulation.cloud/workbench/prj-207be482-26c0-4211-9834-753ae5ff3149>`_,
- `Isolated propeller example <https://flow360.simulation.cloud/workbench/prj-a0dd6591-7f9e-4436-a676-a9b80d68d103>`_.



.. _fig_legacy_surface_mesher_example:
.. figure:: Figures/legacy_sm_example.png
    :align: center
    :width: 70%

    Example of a mesh generated using the **Legacy Surface Mesher**.
