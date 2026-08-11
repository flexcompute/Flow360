.. _GAI_surface_mesher_user_guide:

**************************
GeometryAI Surface Mesher
**************************

GeometryAI Surface Mesher (GAI) combines a geometry processing engine with a surface mesher. It produces mathematically watertight and manifold surface meshes by design, working directly from imperfect or non-watertight CAD without manual geometry repair.

.. _fig_GAI_surface_mesher_example:
.. figure:: ../Figures/gai_sm_example.png
    :align: center
    :width: 50%

    Example of a mesh generated using the **GeometryAI Surface Mesher**.

Main characteristics:

- produces mathematically watertight and manifold surface meshes by design,
- robustly handles non-watertight geometries and patches holes automatically,
- handles intersecting solid bodies well,
- supports basic spatial transformations and boolean operations on solids (to use these features, define the solids in separate CAD files),
- can combine geometries from different sources, including CAD (B-rep) files and discrete surface meshes (STL), into a single watertight surface,
- offers the option to set up an analytical wind tunnel farfield, so a farfield body does not need to be present in the input CAD,
- can build a half-model from a full-body geometry, or a full-body mesh from a half-body geometry, about the symmetry plane, for more efficient straight-line simulations,
- always produces a valid mesh, which makes it well suited to automated, agentic meshing workflows: a first attempt can be generated automatically and then refined only where it matters.

Unlike a standard wrapper, GeometryAI produces a watertight, manifold surface with no intersections or non-manifold elements, directly from the raw CAD input (for a side-by-side illustration, see :ref:`gai_wrapper_comparison`).

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: How it works
      :link: gai_how_it_works
      :link-type: ref

      The geometry-processing engine and every meshing parameter it exposes.

   .. grid-item-card:: Best practices
      :link: gai_best_practices
      :link-type: ref

      Recommended settings, and how to define the fluid domain for external-aerodynamics, internal-flow, wind-tunnel, and porous-media cases.

.. toctree::
   :hidden:

   HowItWorks
   BestPractices


Examples
========

Examples of meshing using *GeometryAI* can be found in the :ref:`Python API Example Library <python_api_example_library>`. Specifically look at:

- :doc:`Windsor Body with Wheels </python_api/example_library/notebooks/windsor_body>`.
