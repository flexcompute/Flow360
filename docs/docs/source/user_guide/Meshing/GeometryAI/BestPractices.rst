.. _gai_best_practices:

**************
Best practices
**************

Recommended settings for GeometryAI meshing, followed by how to define the fluid domain for the common simulation scenarios. The parameters themselves are described in :ref:`gai_how_it_works`.

Recommended settings
====================

Start from the defaults, generate a first mesh, and adjust locally from there. The table gives a sensible starting point for each of the main parameters and when to change it.

.. list-table::
   :header-rows: 1
   :widths: 30 26 44

   * - Parameter
     - Recommended starting point
     - When to adjust
   * - ``Geometry accuracy``

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :py:attr:`~flow360.MeshingDefaults.geometry_accuracy` · :ref:`details <gai_geometry_accuracy>`
     - Default global value
     - Refine locally with a geometry refinement rather than lowering it globally. If the process log reports early termination, coarsen the global value or move the fine accuracy into a local refinement.
   * - ``Surface max edge length``

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :py:attr:`~flow360.MeshingDefaults.surface_max_edge_length` · :ref:`details <gai_surface_mesh_controls>`
     - Generous (large) global value
     - Keep it generous so it does not silently dominate the node-count target; tighten it locally with a surface refinement only where finer elements are needed.
   * - ``Target surface node count``

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :py:attr:`~flow360.MeshingDefaults.target_surface_node_count` · :ref:`details <gai_surface_mesh_controls>`
     - The primary lever for overall mesh size
     - For a refinement study, scale it by ``x`` and ``Surface max edge length`` by ``1 / sqrt(x)`` together (see :ref:`gai_mesh_refinement_study`).
   * - ``Preserve thin geometry``

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :py:attr:`~flow360.MeshingDefaults.preserve_thin_geometry` · :ref:`details <gai_geometry_cleanup>`
     - Off; enable only where needed
     - Turn on, ideally through a local geometry refinement, for thin parts such as trailing edges and fins, keeping the rest of the mesh coarse.
   * - ``Sealing size`` and ``Remove hidden geometry``

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>` · :py:attr:`~flow360.MeshingDefaults.sealing_size` · :py:attr:`~flow360.MeshingDefaults.remove_hidden_geometry` · :ref:`details <gai_geometry_cleanup>`
     - Use for dirty or non-watertight CAD
     - Prefer these cleanup options over manual CAD repair; together they handle most production geometries.
   * - ``Symmetry plane`` (half-model)

       :doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/01.farfield>` · :ref:`details <gai_farfield_half_model>`
     - Use for symmetric, zero-yaw cases
     - Roughly halves the mesh size.

For general meshing guidance that applies across all workflows, see the :ref:`meshing knowledge base <knowledge_base_meshing>`.


.. _gai_mesh_refinement_study:

Running a mesh refinement study
===============================

To study mesh convergence, scale the whole surface mesh up or down consistently rather than changing one parameter in isolation. The recommended approach uses ``Target surface node count`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.target_surface_node_count`) as the single driver:

- Multiply ``Target surface node count`` by a refinement factor ``x`` (for example ``2`` for a finer level, ``0.5`` for a coarser one).
- Scale ``Surface max edge length`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.surface_max_edge_length`) by ``1 / sqrt(x)`` at the same time. Node density on a surface scales with the inverse square of the edge length, so this keeps the edge-length cap consistent with the requested node count and prevents it from binding unexpectedly between levels.

Keeping the two in step means the curvature scaling continues to drive the mesh at every level, so successive meshes differ by a controlled, roughly uniform factor. If ``Surface max edge length`` is left fixed while only the node count changes, it will eventually become the binding constraint and the meshes will stop refining as expected (see ``Target surface node count`` under :ref:`gai_surface_mesh_controls`).


.. _defining_fluid_domain:

Defining the fluid domain
=========================

When the *GeometryAI* (GAI) workflow meshes an assembly, it has to decide which surfaces bound the fluid and which sit outside it. Three controls drive that decision, and they are independent of one another:

- ``Mesh exterior`` (:doc:`WebUI </gui_guide/04.entities-browser/01.geometry/README>`), set per body group, declares which side of a body the fluid is on (outside or inside).
- ``Remove hidden geometry`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.remove_hidden_geometry`), a single global toggle, decides whether occluded interior faces are discarded.
- ``Seed point`` volumes (:doc:`WebUI </gui_guide/04.entities-browser/04.volumes/05.seed-point-volume>`, :py:class:`~flow360.SeedpointVolume`) select which connected region is the fluid; the flood starts from them.

This section explains what each control does, how they combine, and which combination to use for the common simulation scenarios. The mechanics of generating the zones themselves (custom volumes, custom zones, the flood-fill workflow) are covered in :ref:`new_volume_mesher_user_guide`; this section is about choosing the settings that delimit the domain.

.. note::

   These controls apply to the *GAI surface mesher* workflow. See :ref:`meshing` for how the *beta mesher* and *GAI* toggles select each workflow.

Mesh exterior
-------------

``Mesh exterior`` (:doc:`WebUI </gui_guide/04.entities-browser/01.geometry/README>`), written ``mesh_exterior`` in the body-group settings, is a per-body-group flag (``true`` by default) that declares **which side of a body the fluid is on**:

- ``true`` (the default): the fluid is *outside* the body. The body is a solid the flow passes around, such as the object of study (a car, an aircraft, a building).
- ``false``: the fluid is *inside* the body. The body is an enclosure whose interior is meshed and whose faces bound the domain from the outside, such as a wind tunnel or a user-defined farfield.

The fluid domain is the region that is **outside every** ``mesh_exterior=true`` **body and inside every** ``mesh_exterior=false`` **body**. For a car inside a wind tunnel, that is exactly the gap between them: outside the car and inside the tunnel.

Because it defaults to ``true``, most cases never set it explicitly: an external-aerodynamics model, and the walls that bound an internal-flow cavity, all keep the default. You only set ``false`` on a body that should *contain* the fluid, such as an enclosing wind tunnel or user-defined farfield. Setting such an enclosing body to ``true`` instead would tell the mesher the fluid is *outside* it, which is not what you want, and a seed point cannot make up for it (see :ref:`the seed does not replace the flag <gai_seed_points>` below).

Set ``mesh_exterior`` on each body group from the geometry browser; see :doc:`GUI Guide: Geometry </gui_guide/04.entities-browser/01.geometry/README>` for the control.

.. note::

   ``mesh_exterior`` is a beta feature and its interface may change.

Remove hidden geometry
----------------------

``Remove hidden geometry`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.remove_hidden_geometry`) is a single global toggle (off by default, *GAI surface mesher* only). When it is on, face classification runs on the ``mesh_exterior=true`` bodies and drops the faces that the flow can never reach, for example the seats inside a closed car body or the components inside an engine bay. Bodies set to ``mesh_exterior=false`` are not classified and are therefore unaffected.

With the toggle off you still obtain a valid mesh, but the occluded interior surfaces are retained, which adds elements that do not contribute to the external solution. Turn it on when the geometry carries interior parts that are hidden from the flow.

The parameter itself, together with ``Min passage size`` (:py:attr:`~flow360.MeshingDefaults.min_passage_size`) and ``Remove baffle faces`` (:py:attr:`~flow360.MeshingDefaults.remove_baffle_faces`), is described under :ref:`gai_geometry_cleanup`.

.. _gai_seed_points:

Seed points and flooding
------------------------

Within the fluid side defined by ``mesh_exterior``, the mesher picks the connected region to mesh by *flooding* it from one or more seed points. When no seed point is supplied, the default seeds are the corners of the geometry bounding box, which is what makes ordinary external-aerodynamics cases work without any explicit seeding.

The seed **selects and orients** the fluid region; it does not change which side of a body is fluid. Place the seed *outside* the object for external aerodynamics, *inside* the enclosed region for internal flow (for example the cabin of an HVAC model), and *in the gap* between the model and an enclosing wind tunnel.

.. important::

   A seed point is not a substitute for ``mesh_exterior``. Seeding inside a body that is left at ``mesh_exterior=true`` does *not* turn its interior into the fluid domain; the flag, not the seed, decides which side of each body is fluid. An enclosing wind tunnel must therefore be ``mesh_exterior=false`` regardless of where the seed is placed.

Defining a ``Seed point`` volume explicitly (:doc:`WebUI </gui_guide/04.entities-browser/04.volumes/05.seed-point-volume>`, :py:attr:`~flow360.SeedpointVolume.point_in_mesh`) is covered on the seed point volume page of the WebUI guide and, on the Python side, in :ref:`new_volume_mesher_user_guide`.

Choosing settings per scenario
------------------------------

The table below summarises the typical combination for each common scenario. *Default seeding* means the bounding-box corner seeds, with no explicit seed point.

.. list-table::
   :header-rows: 1
   :widths: 24 18 22 36

   * - Scenario
     - ``Mesh exterior``
     - ``Remove hidden geometry``
     - Seeding
   * - External aerodynamics (car, aircraft) with an automated farfield
     - ``true`` on the model
     - On if the model has hidden interior parts; otherwise off
     - Default seeding (outside the model)
   * - Internal flow (HVAC, duct, manifold)
     - ``true`` on the bodies bounding the cavity
     - Usually on, to discard surfaces outside the cavity
     - Seed point inside the cavity
   * - Wind tunnel or user-defined farfield enclosing a model
     - ``false`` on the tunnel/farfield body, ``true`` on the model inside it
     - As needed on the model
     - Seed point in the gap between the model and the enclosure
   * - Porous media
     - ``true`` on the model
     - As needed
     - Seed so the porous medium lies inside the meshed domain

.. _fig_gai_fluid_domain_scenarios:
.. figure:: ../Figures/gai_fluid_domain_scenarios.svg
   :align: center
   :width: 100%

   Fluid-domain classification for the common scenarios. ``mesh_exterior`` sets which side of each body is fluid (outside a ``true`` body, inside a ``false`` body), and the seed selects the connected region to mesh.

.. seealso::

   - Body-group setup: :doc:`GUI Guide: Geometry </gui_guide/04.entities-browser/01.geometry/README>`
   - Hidden-geometry and mesh defaults: :doc:`mesh parameters </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`
   - Farfield types: :doc:`farfield </gui_guide/02.simulation-setup/02.mesh/01.farfield>`
   - Seed-point volumes: :doc:`seed point volume </gui_guide/04.entities-browser/04.volumes/05.seed-point-volume>`
   - Zone generation and the flood-fill workflow: :ref:`new_volume_mesher_user_guide`
   - Python API: :doc:`meshing reference </python_api/API_reference/meshing/meshing_index>`


.. _gai_wrapper_comparison:

Comparison with a standard wrapper
==================================

The comparisons below show the input geometry, the result of a standard wrapper, and the GeometryAI surface for the same detailed regions. The red arrows highlight regions where the standard wrapper produces defects (intersections or non-manifold elements) that GeometryAI avoids.

.. list-table:: Wing mirror and A-pillar: input geometry, a standard wrapper, and GeometryAI.
   :widths: 33 33 33
   :header-rows: 1

   * - Input geometry
     - Standard wrapper
     - GeometryAI
   * - .. image:: ../Figures/gai_mirror_input.png
          :width: 100%
     - .. image:: ../Figures/gai_mirror_wrapper.png
          :width: 100%
     - .. image:: ../Figures/gai_mirror_geometryai.png
          :width: 100%

.. list-table:: Engine bay: input geometry, a standard wrapper, and GeometryAI.
   :widths: 33 33 33
   :header-rows: 1

   * - Input geometry
     - Standard wrapper
     - GeometryAI
   * - .. image:: ../Figures/gai_enginebay_input.png
          :width: 100%
     - .. image:: ../Figures/gai_enginebay_wrapper.png
          :width: 100%
     - .. image:: ../Figures/gai_enginebay_geometryai.png
          :width: 100%
