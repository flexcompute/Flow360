.. _GAI_surface_mesher_user_guide:

**************************
GeometryAI Surface Mesher
**************************

The GeometryAI Surface Mesher (GAI) takes the CAD you have, including CAD that is not watertight, has overlapping parts, or comes from several files, and turns it into a surface mesh that is mathematically watertight and manifold by design. You do not repair the geometry first; you tell the mesher how faithfully to capture it, and it does the rest.

The output is a mesh, not geometry. GeometryAI does not hand back a repaired CAD model you can open and edit: your input geometry is untouched, and everything the mesher seals, merges, or trims exists only in the surface mesh it produces.

.. _fig_GAI_surface_mesher_example:
.. figure:: ../Figures/gai_sm_example.png
    :align: center
    :width: 50%

    Example of a mesh generated using the **GeometryAI Surface Mesher**.

When to choose it
=================

Reach for GeometryAI when you want robust, reproducible, automated meshing. It gives consistent results on problematic CAD: production models with small gaps and leaks between panels, assemblies with intersecting solids, models that mix B-rep CAD with STL parts, and anything else that would otherwise send you into a manual repair loop. Because it always returns a valid mesh, it is particularly useful for automated workflows, where meshing has to behave consistently across a variety of CAD.

It represents sharp edges implicitly through the reconstructed surface rather than conforming mesh edges to feature lines. If your accuracy hinges on mesh edges that follow sharp feature lines on an already watertight model, the :ref:`Surface Mesher <new_surface_mesher_user_guide>` or the :ref:`Snappy Surface Mesher <snappy_surface_mesher_user_guide>` can be a better fit or a good complement. How the mesher toggles select each workflow is explained under :ref:`meshing`.

The three decisions you make
============================

Setting up a GeometryAI mesh comes down to three questions, and the two pages below follow them in that order:

1. **How faithfully should the geometry be captured?** One required setting, ``Geometry accuracy``, answers this globally, and local refinements tighten it only where small features need it.
2. **How should imperfect CAD be cleaned up?** Options to seal gaps and holes, keep thin parts, drop geometry hidden from the flow, and handle zero-thickness baffles.
3. **Which region is the fluid?** Which side of each body the fluid is on, where the fluid region starts, and whether the farfield boundary is already in your CAD or needs to be added.

.. grid:: 2
   :gutter: 3

   .. grid-item-card:: How it works
      :link: gai_how_it_works
      :link-type: ref

      This section explains what the mesher does with your geometry, then takes each control in the order you need it: geometry accuracy, geometry cleanup, local refinements, surface mesh density, farfield.

   .. grid-item-card:: Best practices
      :link: gai_best_practices
      :link-type: ref

      This section is your first GeometryAI mesh step by step, including what to reach for when something looks wrong, and how to define the fluid domain for external-aerodynamics, internal-flow, wind-tunnel, and porous-media cases.

.. toctree::
   :hidden:

   HowItWorks
   BestPractices

Capabilities at a glance
========================

- produces mathematically watertight and manifold surface meshes by design,
- robustly handles non-watertight geometries and patches holes automatically,
- correctly handles intersecting solid bodies,
- can move, rotate, scale, and mirror solid bodies, and apply boolean operations to them, without editing the CAD (to use these features, define the solids in separate CAD files), see :ref:`gai_transformations`,
- can combine geometries from different sources, including CAD (B-rep) files and discrete surface meshes (STL), into a single watertight surface,
- lets a rotating zone cut through the geometry, splitting each surface it crosses into the part inside the zone and the part outside,
- offers the option to set up an analytical wind tunnel farfield, so a farfield body does not need to be present in the input CAD,
- can build a half-model from a full-body geometry, or a full-body mesh from a half-body geometry, for more efficient simulation tool chains,
- always produces a valid mesh, which makes it well suited to automated, agentic meshing workflows.

Unlike a standard wrapper, GeometryAI produces a watertight, manifold surface with no intersections or non-manifold elements, directly from the raw CAD input. For a side-by-side illustration, see :ref:`gai_wrapper_comparison`.

Examples
========

Examples of meshing using *GeometryAI* can be found in the :ref:`Python API Example Library <python_api_example_library>`. Specifically look at:

- :doc:`Windsor Body with Wheels </python_api/example_library/notebooks/windsor_body>`,
- :doc:`Rotating Zone Cutting Through the Geometry </python_api/example_library/notebooks/gai_propeller_nacelle>`.
