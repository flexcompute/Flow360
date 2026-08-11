.. _gai_how_it_works:

*********************
How GeometryAI works
*********************

This page describes the GeometryAI engine and the meshing parameters it exposes. For recommended settings and how to define the fluid domain per scenario, see :ref:`gai_best_practices`.

Meshing Process
===============

The surface mesher performs two main steps:

1. **Geometry Processing**

   Through the use of *GeometryAI*, the geometry is processed and prepared for surface meshing. This includes:

   - applying basic spatial transformations and boolean operations to solids (defined in separate CAD files),
   - resolving intersections,
   - patching holes in the geometry,
   - constructing the farfield geometry,
   - splitting the geometry to create a half-model.

2. **Surface Meshing**

   The geometry is meshed using the surface mesher which enables the user to specify all kinds of refinements, also per face of the input geometry.

The behaviour of both steps is driven by a small set of meshing parameters described in the sections below. Each parameter is exposed both in the WebUI (:doc:`Mesh parameters </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`) and in the Python API (:ref:`Meshing reference <python_api_meshing>`), so the same workflow applies regardless of how the case is set up.

The geometry-processing step also decides which surfaces bound the fluid, controlled by ``Mesh exterior`` (:doc:`WebUI </gui_guide/04.entities-browser/01.geometry/README>`), ``Remove hidden geometry`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.remove_hidden_geometry`), and ``Seed point`` volumes (:doc:`WebUI </gui_guide/04.entities-browser/04.volumes/05.seed-point-volume>`, :py:class:`~flow360.SeedpointVolume`). See :ref:`defining_fluid_domain` for how to combine them for external aerodynamics, internal flow, wind-tunnel, and porous-media cases.

.. admonition:: Built for automation
   :class: tip

   Because GeometryAI always returns a valid, watertight mesh, even from dirty input, it lends itself to automated and agentic workflows. A first mesh can be produced with default parameters, evaluated, and then improved by adding local refinements only where the geometry or the flow requires them, without the manual repair loop that traditional surface meshers need.

.. note::

   The GeometryAI mesher may stop before fully achieving the requested ``Geometry accuracy`` if system resource constraints are encountered during surface meshing. When this occurs, a message indicating early termination will appear in the process log, and it reports the finest geometry accuracy that was actually reached. If you observe this, consider increasing the ``Geometry accuracy`` value (coarser) globally, or, preferably, apply a finer accuracy only where it is needed with a local geometry refinement (see :ref:`gai_local_refinements`).


.. _gai_geometry_accuracy:

Geometry accuracy
=================

``Geometry accuracy`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.geometry_accuracy`) is the central parameter of the GeometryAI Surface Mesher, and it is **required** whenever GeometryAI is enabled. It is the smallest length scale that the geometry processing step resolves accurately: it controls how faithfully the reconstructed surface follows the input CAD, particularly around small features and sharp edges.

A key idea behind GeometryAI is that **geometry accuracy is decoupled from the surface mesh resolution**. Geometry accuracy governs how well the geometry itself is captured, while the surface mesh density is controlled separately (see :ref:`gai_surface_mesh_controls`). A smaller value therefore resolves finer geometric detail at the cost of longer geometry processing; it is not a direct mesh-density control. It can still raise the node count indirectly and locally, because finer detail (for example sharper edges and higher local curvature) needs more nodes to represent, but for a smooth shape the effect on the overall count is small.

``Geometry accuracy`` is an absolute length: it corresponds to the smallest edge you want to capture. It does not scale with the overall size of the model, so the same trailing edge needs the same geometry accuracy whether it sits on a small or a large body.

Resolution carries a cost: the finer the geometry accuracy, the longer the geometry processing and meshing take. Lowering the global value to capture one small local feature refines the *whole* model and inflates both the mesh node count and the meshing time. The recommended workflow is therefore:

#. Start from the **default** global ``Geometry accuracy``. It is set automatically when GeometryAI is enabled and is a sensible baseline for the bulk of the geometry.
#. Add **local geometry refinements** on the specific faces that carry small features (thin trailing edges, small gaps, tight radii), to set a finer local accuracy only there (see :ref:`gai_local_refinements`).

This keeps the global geometry accuracy, and the meshing time, moderate while still resolving the small features that matter.

The mesher accepts a wide range of values. The main guard is a sanity floor: if the value is finer than ``1e-5`` times the bounding box diagonal, the mesher warns that it is unreasonably fine for the model and recommends increasing it.

.. note::

   In the WebUI, ``Geometry accuracy`` and the mesh input lengths (such as ``Surface max edge length``, ``Sealing size``, and ``Min passage size``) use the project length unit by default, but you can also enter an explicit unit. Make sure the unit is correct: a value given in the wrong length unit (for example one far smaller than intended) can produce a much finer mesh than expected and lead to very long meshing times. In the Python API each value carries its own explicit unit.


.. _gai_geometry_cleanup:

Cleaning up the geometry
========================

Since GeometryAI reconstructs a watertight surface from the input, it exposes several controls for cleaning up imperfect or overly detailed CAD before meshing. All of the options below are specific to the GeometryAI Surface Mesher.

- ``Sealing size`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.sealing_size`): closes *physical* gaps and holes in the geometry that are smaller than this threshold (for example the central hole of a torus, or a deliberate slot you do not want the mesh to pass through). These physical openings do not usually prevent a watertight surface, so this parameter is mainly about user intent: it decides which real openings should be sealed shut. It can be set to zero to disable this sealing. Even at zero, GeometryAI still closes the non-physical gaps (small cracks and leaks between faces) that would otherwise break watertightness, because producing a watertight surface does not depend on this parameter.

- ``Preserve thin geometry`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.preserve_thin_geometry`): controls how aggressively thin features (for example trailing edges, fins, or thin walls) are retained. With this option off, thin features need a thickness of roughly three times the ``Geometry accuracy`` to survive; with it on, thin features down to a scale of the ``Geometry accuracy`` are preserved accurately, at the cost of a finer local mesh.

- ``Remove hidden geometry`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.remove_hidden_geometry`): removes internal geometry that is not visible to the flow (for example components sealed inside a closed body). The companion ``Min passage size`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.min_passage_size`) sets the smallest opening through which an internal region is still considered connected to the exterior; internal regions reachable only through passages smaller than this are treated as hidden. If not specified, it is derived from ``Geometry accuracy`` and ``Sealing size``.

- ``Remove baffle faces`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.remove_baffle_faces`): when hidden geometry removal is active, faces that are exposed to the exterior on both sides (baffles, zero-thickness walls) are removed. Turning this option off instead thickens such faces into a closed shell so they are retained.

.. admonition:: Best practice
   :class: tip

   Reach for these cleanup options before resorting to manual CAD repair. ``Sealing size`` and ``Remove hidden geometry`` are usually enough to turn a dirty, non-watertight production model into a meshable one, which is the main reason to choose this mesher for automotive and other detailed geometries.


.. _gai_surface_mesh_controls:

Controlling the surface mesh
============================

Once the geometry is processed, the surface mesh density is governed by the following parameters:

- ``Curvature resolution angle`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.curvature_resolution_angle`): the maximum angular deviation allowed between a surface element and the underlying surface. Lower values capture curvature more accurately but increase the node count.

- ``Surface max edge length`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.surface_max_edge_length`): an upper bound on surface cell edge length. It is enforced regardless of the other surface controls, so it acts as a hard cap on element size. Because the surface mesh nodes are always projected onto the geometry, a small ``Surface max edge length`` also yields good curvature resolution on its own (a fine edge length cannot leave a curved surface under-resolved). The recommended practice is to keep a generous (large) global value and tighten it only where needed through a local surface refinement (see :ref:`gai_local_refinements`).

- ``Surface max aspect ratio`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.surface_max_aspect_ratio`): the maximum aspect ratio of surface cells. Lower values force more isotropic cells.

- ``Surface edge growth rate`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.surface_edge_growth_rate`): the rate at which surface edges grow in length away from regions of high curvature or small max edge length constraints. Typical values are ``1.1`` to ``1.25``.

- ``Resolve face boundaries`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.resolve_face_boundaries`): when enabled, the boundaries between adjacent faces are resolved accurately with anisotropic refinement. Enable this when face-to-face boundaries (for example between a body and an appendage) must be captured sharply.

- ``Target surface node count`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/02.mesh-parameters>`, :py:attr:`~flow360.MeshingDefaults.target_surface_node_count`): when set, the mesher scales the **curvature** metric so that the surface mesh reaches approximately this number of nodes, then enforces ``Surface max edge length`` on top of the result. This is a quick way to control overall mesh density without tuning individual length scales. The target is approximate: it is not a guaranteed minimum or maximum. Internally the value is converted to a target mesh complexity and the curvature metric is rescaled uniformly to match it, so the realised node count can land somewhat above or below the requested number. Two consequences follow from this:

  - The actual node count is *higher* than the target whenever ``Surface max edge length`` is the binding constraint. If you request a low target node count together with a fine ``Surface max edge length``, the edge length dominates, the node count overshoots the target, and the curvature scaling has limited influence.
  - It does not change the geometry-resolving behaviour, which remains governed by ``Geometry accuracy``.

  To let the target node count drive the mesh, keep ``Surface max edge length`` generous so it does not bind.


.. _gai_local_refinements:

Local refinements
=================

Most tuning of a GeometryAI mesh should be done locally, on the specific faces that need it, rather than by changing global values. Two per-face refinement types are available:

- ``Geometry Refinement`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/06.refinements/07.geometry-refinement>`, :py:class:`~flow360.GeometryRefinement`) overrides ``Geometry accuracy`` (and, optionally, ``Preserve thin geometry``, ``Sealing size``, and ``Min passage size``) on a selected set of faces. This is the recommended way to resolve a small or detailed feature without refining the whole model.

- ``Surface Refinement`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/06.refinements/02.surface-refinement>`, :py:class:`~flow360.SurfaceRefinement`) overrides surface mesh controls such as ``Surface max edge length``, ``Curvature resolution angle``, and ``Resolve face boundaries`` on selected faces.

Both refinement types are listed with the other refinements in the :ref:`Meshing reference <python_api_meshing>`.

.. note::

   With the GeometryAI surface mesher a *uniform refinement* (a volumetric refinement region) also refines the surface mesh inside the region. See :ref:`Effect on the surface mesh <uniform_refinement_surface_effect>` for how each workflow treats this.


.. _gai_farfield_half_model:

Farfield and half-model setup
=============================

GeometryAI can build the farfield analytically as part of geometry processing, so a separate farfield body is not required in the input CAD. Two analytic styles are available, selected through ``Type`` on the farfield form (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/01.farfield>`):

- ``Automated farfield`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/01.farfield>`, :py:class:`~flow360.AutomatedFarfield`) generates a (semi)sphere or cylinder sized relative to the geometry bounding box. This is the general-purpose choice for external aerodynamics.

- ``Wind tunnel`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/01.farfield>`, :py:class:`~flow360.WindTunnelFarfield`) builds a rectangular tunnel and is intended for ground-vehicle and wind-tunnel-correlation work. It supports several floor treatments (static floor, fully moving floor, central belt, and wheel belts) to represent moving-ground conditions.

For symmetric configurations, the mesher can split the geometry into a half-model about the symmetry plane, which roughly halves the mesh size and is well suited to straight-line (zero-yaw) simulations. It is enabled with ``Symmetry plane``, and ``Volume to include`` selects which side of the plane is kept (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/01.farfield>`). The half-model option requires the GeometryAI Surface Mesher together with the beta volume mesher.

The full set of farfield classes is listed under :ref:`Farfield zone specifications <python_api_meshing_farfield_specs>`.


Working with sharp edges and thin features
==========================================

GeometryAI represents sharp edges implicitly through the reconstructed surface rather than conforming mesh edges to feature lines, so edge-based refinements do not apply: ``Surface Edge Refinement`` (:doc:`WebUI </gui_guide/02.simulation-setup/02.mesh/06.refinements/01.surface-edge-refinement>`, :py:class:`~flow360.SurfaceEdgeRefinement`) is not supported with the GeometryAI mesher. To sharpen the representation of an edge, refine the ``Geometry accuracy`` (a smaller value), locally where possible. Likewise, features thinner than the active thickness threshold (see ``Preserve thin geometry`` above) are best resolved with a local geometry refinement or by enabling ``Preserve thin geometry`` on those parts.

For geometries whose accuracy hinges on conforming the mesh to sharp feature edges, the :ref:`Surface Mesher <new_surface_mesher_user_guide>` (watertight input) or the :ref:`Snappy Surface Mesher <snappy_surface_mesher_user_guide>` can be a good complement.
