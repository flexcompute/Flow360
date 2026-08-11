.. _legacy_volume_mesher_user_guide:

********************
Legacy Volume Mesher
********************

The legacy volume mesher is the original volume meshing algorithm in Flow360 and is currently the default volume mesher. While still fully functional, it is being phased out in favor of the new volume mesher.

Main characteristics:

- uses an unstructured tetrahedral mesh in the farfield region,
- supports spherical and half-spherical farfield configurations,
- allows for the generation of quasi-3D meshes for 2D simulations.

Available features:

- Boundary layers:

  - first layer thickness specification,
  - growth rate control,
  - automatic layer count calculation based on target wall spacing.

- Refinements:

  - uniform volumetric refinements specified using cylinders or boxes,
  - axisymmetric volumetric refinements specified using cylinders.

- Zones:

  - rotating volume zones for simulating rotating machinery,
  - automatic farfield zone generation.

- Farfield:

  - spherical farfield generation with configurable radius,
  - half-sphere farfield for symmetric configurations.

- Quasi-3D meshing:

  - automatic handling of quasi-3D geometries,
  - boundary condition setup for quasi-3D simulations.

Examples
========

Examples of mesh generation can be found in the :ref:`Python API Example Library <python_api_example_library>`:

- :doc:`2D CRM airfoil example <../../python_api/example_library/notebooks/2D_CRM_airfoil>`,
- :doc:`DARPA AD example <../../python_api/example_library/notebooks/DARPA_SUBOFF_AD>`.

And in the `WebUI Example Library <https://flow360.simulation.cloud/examples>`_.:

- `Simple airplane example <https://flow360.simulation.cloud/workbench/prj-261fdfa7-025e-40d7-9c5d-52c0f206f782>`_,
- `eVTOL with BET line example <https://flow360.simulation.cloud/workbench/prj-207be482-26c0-4211-9834-753ae5ff3149>`_,
- `Isolated propeller example <https://flow360.simulation.cloud/workbench/prj-a0dd6591-7f9e-4436-a676-a9b80d68d103>`_.


.. _fig_legacy_volume_mesher_example:
.. figure:: Figures/legacy_vm_example.png
    :align: center
    :width: 70%

    Clip of an example mesh generated using the **Legacy Volume Mesher**.
