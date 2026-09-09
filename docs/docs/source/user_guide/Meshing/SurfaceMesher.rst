.. _new_surface_mesher_user_guide:

**************
Surface Mesher
**************

The *Surface Mesher* produces high-quality anisotropic surface meshes from watertight CAD geometries. It is currently opt-in: enable it via the *beta mesher* toggle in both the WebUI and the Python API.

Main characteristics:

- requires a clean, watertight input geometry
- generates anisotropic meshes, allowing finer refinement near leading and trailing edges
- retains feature edges.

.. _fig_new_surface_mesher_example:
.. figure:: Figures/new_sm_example.png
    :align: center
    :width: 70%

    Example of a mesh generated using the **Surface Mesher**.

Surface refinements
===================

User-defined refinements locally tighten the surface mesh beyond the global edge length and growth rate. One refinement type is available on the *Surface Mesher*:

- *Surface refinement* (per-face): override the maximum edge length or the curvature-resolution angle on selected faces. Useful for surfaces that need finer or coarser resolution than the global setting provides, or where the geometry curvature warrants a tighter angular limit.

.. seealso::

   Surface refinement setup: :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/06.refinements/02.surface-refinement>` · :doc:`Python </python_api/API_reference/meshing/meshing_index>`.

Reducing node count
===================

The *Surface Mesher* automatically refines the mesh near feature edges, which can make the surface mesh denser than needed. Two adjustments help reduce the overall node count:

- **Coarsen the default edge spacing**: set a high global *surface max edge length* and specify per-patch refinements only on the patches that need them. The mesher clusters nodes around feature edges subject to growth-rate and maximum-aspect-ratio constraints, so a larger global edge length lets cells stretch further *along* the edge while still satisfying those constraints, which reduces the number of nodes deposited along the edge. Per-patch refinements then restore the resolution where it matters.
- **Increase the edge growth rate**: the default growth rate from feature edges can be conservative. Higher values produce a faster transition away from edges; up to **1.5** is recommended.

The combined effect of both adjustments is illustrated below for a simple 1 m cube test case. Each image shows a close-up view of a corner of the cube:

.. list-table::
   :widths: 33 33 33
   :header-rows: 0

   * - .. figure:: Figures/sm_growth_1p2_max_0p05.png
          :width: 100%

          Growth rate **1.2**, max edge length **0.05 m**, 103,031 surface nodes.
     - .. figure:: Figures/sm_growth_1p5_max_0p05.png
          :width: 100%

          Growth rate **1.5**, max edge length **0.05 m**, 48,804 surface nodes.
     - .. figure:: Figures/sm_growth_1p5_max_0p1.png
          :width: 100%

          Growth rate **1.5**, max edge length **0.1 m**, 24,061 surface nodes.

.. seealso::

   Setup: :doc:`WebUI mesh parameters </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :doc:`WebUI surface refinement </gui_guide/02.simulation-setup/02.mesh/06.refinements/02.surface-refinement>` · :doc:`Python </python_api/API_reference/meshing/meshing_index>`.

Examples
========

Examples of meshing using *Surface Mesher* can be found in the :ref:`Python API Example Library <python_api_example_library>`:

- :doc:`Isolated wing with BET Disk Model <../../python_api/example_library/notebooks/isolated_wing_BET>`.
- :doc:`eVTOL Vehicle Mesh Generation <../../python_api/example_library/notebooks/BET_eVTOL>`.

.. admonition:: Recommended for:
   :class: tip

   - watertight aerospace geometries,
   - simple geometries of aerodynamic devices.
