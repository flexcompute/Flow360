.. _asset_drafts_userGuide:

************
Asset Drafts
************

In Flow360, a **Draft** is an isolated, in-memory snapshot of an asset's configuration. It provides a workspace where you can inspect and modify entities—surfaces, edges, volumes, body groups, and coordinate systems—without affecting the cloud asset directly. Drafts are essential for preparing simulation configurations before submission.

What is a Draft?
================

When you work with assets (geometries, surface meshes, or volume meshes) in Flow360, you often need to configure how entities are grouped, named, and transformed before running a simulation. A draft acts as a local editing environment that:

- **Captures** the current state of an asset's entity information
- **Isolates** your modifications from the cloud-stored asset
- **Tracks** all entities (surfaces, edges, body groups) for use in simulation setup
- **Manages** advanced operations like mirroring and coordinate system assignments

Drafts are the bridge between your assets and the simulation configuration. When you create a draft from a geometry or mesh, you gain access to all its entities and can configure them for boundary conditions, mesh refinement zones, and output settings.

Draft Lifecycle
---------------

A typical draft workflow follows these steps:

1. **Create** a draft from an existing asset (geometry, surface mesh, or volume mesh)
2. **Configure** entity groupings, naming, and transformations
3. **Reference** entities in your simulation setup (boundary conditions, refinement zones, outputs)
4. **Submit** the simulation case, which uses the draft configuration

Tracked Entities
----------------

A draft tracks various entity types that can be used in simulation setup:

- **Surfaces**: Face groups from geometry, used for boundary conditions and surface outputs
- **Edges**: Edge groups from geometry, used for mesh refinement control
- **Body groups**: Solid body components, used for transformations and mirroring
- **Volumes**: Volume zones in the computational domain, used for assigning 3D models (e.g., fluid, solid, porous media)
- **Boxes**: User-defined box regions for mesh refinement or flow initialization
- **Cylinders**: User-defined cylindrical regions for mesh refinement or flow initialization
- **Mirrored surfaces**: Surfaces created through mirror operations
- **Mirrored body groups**: Body groups created through mirror operations
- **Imported geometries**: Additional geometry components added to the project
- **Imported surfaces**: Additional surface mesh components added to the project

The draft also provides managers for:

- **Coordinate systems**: For applying geometric transformations (translation, rotation, scaling) to bodies
- **Mirror operations**: For creating symmetric geometry by mirroring body groups

Entity Grouping
===============

Entity grouping is a fundamental concept in Flow360 that determines how geometric features (faces, edges, and bodies) are organized for simulation setup. The way entities are grouped directly affects:

- Which surfaces are available for boundary condition assignment
- How mesh refinement zones can be defined
- What surface output names appear in your results

.. admonition:: Warning
   :class: danger

   Grouping entities outside of the draft context is being deprecated. Please use the draft context to group entities.

.. admonition:: Important
   :class: important

   **Grouping applies only to geometry-based workflows.** When starting a project from a surface mesh or volume mesh file, the boundary assignments are taken directly from the mesh file itself. The grouping functionality described below is not available for mesh-based projects.

Face Grouping
-------------

Faces are geometric surfaces that form parts of your model. Face grouping allows you to organize these surfaces based on metadata embedded in your CAD geometry file.

**CAD Face Metadata**

CAD geometry files contain metadata associated with each face that Flow360 can use for grouping:

- **name** (optional): The face name as defined in CAD software, mapped to ``builtinName`` in Flow360
- **color** (optional): Face color information from CAD
- **material** (optional): Material information associated with the face
- **attributes** (optional): Key-value pairs defined by the designer, such as:
  
  - ``groupName``: e.g., "wing"
  - ``faceName``: e.g., "fuselage"
  - ``boundaryName``: e.g., "tail"

.. admonition:: Note
   :class: note

   CAD face metadata (name, color, material, attributes) is only available for B-rep files (e.g., STEP, IGES, SolidWorks, CATIA). This metadata is not available when using STL or UGrid files as geometry input.

**Flow360-Specific Grouping Concepts**

In addition to CAD metadata, Flow360 provides its own grouping identifiers:

- **faceId**: A Flow360-generated identifier indicating the face's position in the CAD model structure. For B-rep files, it follows the format ``body00002_face00004`` (4th face of the 2nd body). For discrete files like STL, this is the actual face/solid name from the file.
- **groupByBodyId**: Groups faces by their parent body identifier within the CAD structure.

**Available Grouping Options**

When configuring face grouping, you can choose from various attributes depending on what metadata is present in your CAD file:

- Grouping by ``faceId`` (Flow360-generated)
- Grouping by ``groupByBodyId`` (Flow360-generated)
- Grouping by CAD attribute (read from CAD file)
- Grouping by CAD metadata (read from CAD file)  

The chosen grouping determines which face groups are available throughout your simulation setup and affects the names used in boundary conditions and surface outputs.

The available grouping options can be seen easily in the Entities Browser in the WebUI or by calling :py:meth:`~flow360.component.geometry.Geometry.show_available_groupings` on the geometry asset in python API.

Edge Grouping
-------------

Edges are the line segments that form boundaries between faces. Edge grouping enables you to organize these geometric features for:

- Defining mesh refinement along specific edges (e.g., leading/trailing edges of wings)
- Controlling anisotropic layer growth
- Marking important geometric features

**Edge Grouping Options**

Edges can be grouped based on attributes assigned in your CAD system:

- **edgeId**: Flow360-generated edge identifier
- **edgeName**: Edge name from CAD (if assigned)

Like faces, edge attributes must be properly assigned in your CAD software before importing the geometry to take full advantage of grouping capabilities.

Body Grouping
-------------

Bodies represent distinct solid components within your geometry. Body grouping is useful for:

- Performing geometric transformations
- Creating mirrored bodies
- Geometry boolean operations

**Body Grouping Options**

Bodies can be grouped by:

- **bodyId**: Flow360-generated body identifier

Mesh-Based Workflows
====================

When starting a project from a surface mesh or volume mesh file rather than CAD geometry, 
the draft still provides access to its stored entities, 
but the entity grouping behavior differs significantly:

**Surface Mesh Projects**

- Boundary names are read directly from the mesh file
- For UGrid files, boundary names come from the associated ``.mapbc`` file (must have the same name and be in the same directory)
- For CGNS files, boundary conditions are read from the file structure
- No grouping configuration is needed or available

**Volume Mesh Projects**

- Boundary names and volume zone names are taken directly from the volume mesh file
- The mesh must include properly named boundary patches
- Grouping operations are not applicable

Geometry Analysis: Oriented Bounding Box
========================================

Beyond managing entities, a draft created from a **geometry** asset also gives
access to a geometry-analysis tool (rather than a draft concept of its own):
fitting an **oriented bounding box (OBB)** to a selected set of surfaces. The
box is aligned with the natural axes of those surfaces and gives their center,
principal axes and extents. From that box a rotation axis and radius can be
derived, which is convenient for setting up rotating components such as wheels
without measuring the geometry by hand. It is the same computation the WebUI
offers through its "Estimate rotation from selected faces" helper on a rotating
wall.

.. admonition:: Note
   :class: note

   This tool is available only for drafts created from a geometry asset. Drafts
   created from a surface mesh or volume mesh do not carry the tessellation data
   the bounding box is computed from.


.. seealso::

   - :ref:`Draft API Reference <python_api_draft>` — Detailed documentation of the ``DraftContext`` class and draft management functions
   - :doc:`Estimate a wheel rotation axis with an OBB </python_api/grab_and_go_snippets/compute_obb>` (runnable snippet using the oriented bounding box)
   - :doc:`Rotating wall boundary condition </gui_guide/02.simulation-setup/03.flow-solver/01.boundary-conditions/01.wall>` (configuring wall rotation in the WebUI)
   - :doc:`Faces in the Entities Browser </gui_guide/04.entities-browser/01.geometry/02.faces>` — Face selection and grouping in the WebUI
   - :doc:`Edges in the Entities Browser </gui_guide/04.entities-browser/01.geometry/01.edges>` — Edge selection and grouping in the WebUI
   - :doc:`Windsor Body with GeometryAI </python_api/example_library/notebooks/windsor_body>` — Example demonstrating draft usage with geometry workflows
