.. _automatedMeshing:

.. currentmodule:: flow360

Automated Meshing
*****************

Overview
========

Flow360 offers automated meshing, from a CAD geometry to a surface mesh and finally to a volume mesh. The supported CAD formats are CSM and EGADS. The output volume meshes are in CGNS format.

.. _Meshing settings:

Meshing settings
================

Mesh in flow360 can be defined using settings, which fall into one of the three categories visible below:

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Setting
     - Default
     - Description
   * - :py:attr:`~MeshingParams.defaults`
     - REQUIRED
     - Default/global settings for meshing.
   * - :py:attr:`~MeshingParams.volume_zones`
     - REQUIRED
     - Creation of new volume_zones. Can be of type :class:`RotationCylinder` or :class:`AutomatedFarfield`.
   * - :py:attr:`~MeshingParams.refinements`
     - OPTIONAL
     - Additional fine-tunning for the mesh on top of default settings.

More details about each of those categories can be found here:

- :ref:`Meshing defaults`
- :ref:`Volume zones`
- :ref:`Refinements`

Surface Meshing
===============

The surface mesher takes the geometry file and user defined settings as inputs to generate a surface mesh.
Here, settings can be defined either in WebUI or through Python API, and are used to control the surface mesh resolution.
Example of defining surface meshing settings in Python API:

.. code-block:: python

  import flow360 as fl
  with fl.SI_unit_system:
      params = fl.SimulationParams(
          meshing=fl.MeshingParams(
              defaults=fl.MeshingDefaults(
                  surface_max_edge_length=1,
                  surface_edge_growth_rate=1.2,
                  curvature_resolution_angle=12 * fl.u.deg
              )
          ),
          ...
      )

The surface mesh can be created by submitting the project with a geometry file and providing previously set up :class:`SimulationParams` containing :class:`MeshingParams` via the Python API:

.. code-block:: python

  import flow360 as fl
  project = fl.Project.from_file("path/to/geometry.csm", "MyProject")
  project.generate_surface_mesh(params, "MySurfaceMesh")
  print(project.surface_mesh.id)

Inputs
    - WebUI project that contains an uploaded geometry
    - params: :py:class:`SimulationParams`

Outputs
    - surface mesh on cluster
    - return the :code:`surface_mesh.id`

.. _fig_surfMesh:
.. figure:: Figures/surfaceMesh.png
   :width: 80%
   :align: center

   Auto-generated surface mesh. An extra refinement is applied to the top face of the box.

.. _fig_surfMesh_edges:
.. figure:: Figures/surfaceMesh_firstLayerThickness.png
   :width: 50%
   :align: center

   Surface mesh on a wing. Anisotropic layers are growing from leading and trailing edges.

.. _fig_surfMesh_edges_projectAnisoSpacing:
.. figure:: Figures/projectAnisoSpacing.png
   :width: 70%
   :align: center

   Surface mesh on a hub. Anisotropic layers are growing from :code:`hubCircle` (green). The anisotropic spacing on the neighboring patches is "projected" to :code:`hubSplitEdge` (red) and updates the nodes distribution along :code:`hubSplitEdge`.

.. _VolumeMeshSection:

Volume Meshing
==============

The volume mesher takes a surface mesh and user defined settings to generate a volume mesh. These settings are defined either through WebUI or Python API and determine the volume mesh resolution. 
They contain :py:attr:`~MeshingDefaults.boundary_layer_first_layer_thickness` and :py:attr:`~MeshingDefaults.boundary_layer_growth_rate` of 3D anisotropic layers, the size/location/resolution of :py:attr:`~MeshingParams.refinements` and etc. 
Below is an example of defining volume meshing settings in Python API:

.. code-block:: python

  import flow360 as fl
  with fl.SI_unit_system:
      cylinder1 = fl.Cylinder(name="Cylinder1", axis=(0, 1, 0), center=(1, 0.2, 0), outer_radius=3, height=2)
      farfield = fl.AutomatedFarfield(name="farfield", method="auto")
      params = fl.SimulationParams(
          meshing=fl.MeshingParams(
              defaults=fl.MeshingDefaults(
                  boundary_layer_growth_rate=1.15,
                  boundary_layer_first_layer_thickness=3e-6,
              ),
              refinement_factor=1.25,
              gap_treatment_strength=0.7,
              volume_zones=[farfield],
              refinements=[
                  fl.UniformRefinement(name="refinement1", spacing=0.2, entities=[cylinder1])
              ]
          ),
          ...
      )

In the above Python script, there is a cylindrical volume mesh refinement zone centered at (1, 0.2, 0).
This refinement will have a uniform spacing of 0.2 m in this volume zone.
The volume mesh can be created from an existing surface mesh and parameters for the volume meshing, for example, using the Python API:

.. code-block:: python

  import flow360 as fl
  project = fl.Project.from_file("path/to/geometry.csm", "MyProject")
  project.generate_volume_mesh(params, "MyVolumeMesh")
  print(project.volume_mesh.id)

Inputs
    - WebUI project that contains an uploaded geometry
    - params: :class:`SimulationParams`

Outputs
  - volume mesh on cluster (in CGNS format)
  - Flow360Mesh.json on cluster
  - return the :code:`volume_mesh.id`

.. _fig_volMesh3D:
.. figure:: Figures/volumeMesh3DView.svg
   :width: 80%
   :align: center

   Left\: WebUI showing the refinement zone. Right\: auto-generated volume mesh.

.. _fig_volMeshSlices:
.. figure:: Figures/volumeMeshSlices.svg
   :width: 80%
   :align: center

   Auto-generated volume mesh. Left\: sliced at y=1. Right: sliced at y=-1.


.. _Meshing defaults:

:py:attr:`~MeshingParams.defaults`
==================================

This section describes the default settings for both surface and volume meshing.

.. list-table::
   :widths: 20 10 10 60
   :header-rows: 1

   * - Option
     - Default
     - Type
     - Description
   * - :py:attr:`~MeshingDefaults.surface_edge_growth_rate`
     - 1.2
     - SURFACE
     - [float] Growth rate of the anisotropic layers grown from the edges.
   * - :py:attr:`~MeshingDefaults.surface_max_edge_length`
     - REQUIRED
     - SURFACE
     - [float] Default maximum edge length for surface cells. This can be overridden with :class:`SurfaceRefinement`.
   * - :py:attr:`~MeshingDefaults.curvature_resolution_angle`
     - 12
     - SURFACE
     - | [float] Default maximum angular deviation in degrees. This value will restrict:
       | 1. The angle between a cell's normal and its underlying surface normal.
       | 2. The angle between a line segment's normal and its underlying curve normal.
       | This can not be overridden per face.
   * - :py:attr:`~MeshingDefaults.boundary_layer_growth_rate`
     - 1.2
     - VOLUME
     - [float] Default growth rate for volume prism layers. This can not be overridden per face.
   * - :py:attr:`~MeshingDefaults.boundary_layer_first_layer_thickness`
     - REQUIRED
     - VOLUME
     - [float] Default first layer thickness for volumetric anisotropic layers. This can be overridden with :class:`BoundaryLayer`.
   * - :py:attr:`~MeshingParams.refinement_factor`
     - 1
     - VOLUME (ADVANCED)
     - Adjusts all spacings in refinement regions and first layer thickness to generate `r`-times finer mesh, where r i s the :py:attr:`~MeshingParams.refinement_factor` value.
   * - :py:attr:`~MeshingParams.gap_treatment_strength`
     - 0
     - VOLUME (ADVANCED)
     - | Narrow gap treatment strength used when two surfaces are in close proximity.
       | Takes a value between 0 and 1, where 0 is no treatment and 1 is the most conservative treatment.
       | This parameter has a global impact where the anisotropic transition into the isotropic mesh.
       | Its impact on regions without close proximity is negligible.

Example of defining :class:`MeshingDefaults` in Python API\:

.. code-block:: python

  import flow360 as fl
  with fl.SI_unit_system:
      params = fl.SimulationParams(
          meshing=fl.MeshingParams(
              defaults=fl.MeshingDefaults(
                  surface_max_edge_length=1,
                  surface_edge_growth_rate=1.2,
                  curvature_resolution_angle=12 * fl.u.deg,
                  boundary_layer_growth_rate=1.1,
                  boundary_layer_first_layer_thickness=1e-5
              ),
              refinement_factor=1.1,
              gap_treatment_strength=0.2
          )
          ...
      )

Gap treatment examples\:

.. _fig_gaptreatment:
.. figure:: Figures/gap.svg
   :width: 100%
   :align: center

   Different :py:attr:`~MeshingParams.gap_treatment_strength` strength. The larger :py:attr:`~MeshingParams.gap_treatment_strength` is, the more space is dedicated to isotropic mesh than anisotropic mesh in narrow gaps.

.. _Volume zones:

:py:attr:`~MeshingParams.volume_zones`
======================================

Settings used for defining :class:`AutomatedFarfield` and :class:`RotationCylinder` volume zones.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Volume zone
     - Default
     - Description
   * - :class:`AutomatedFarfield`
     - REQUIRED
     - Automatically created farfield volume zone.
   * - :class:`RotationCylinder`
     - OPTIONAL
     - Rotation cylinder volume zone.

Example of defining :py:attr:`~MeshingParams.volume_zones` in Python API\:

.. code-block:: python

  import flow360 as fl
  with fl.SI_unit_system:
      farfield = fl.AutomatedFarfield(name="Farfield", method="auto")
      cylinder = fl.Cylinder(name="Cylinder", axis=[0, 1, 0], center=[1, 1, 1], outer_radius=10, height=3)
      rotation_cylinder = fl.RotationCylinder(
          name="RotationCylinder",
          spacing_axial=0.5,
          spacing_circumferential=0.3,
          spacing_radial=1.5,
          entities=cylinder
      )
      params = fl.SimulationParams(
          meshing=fl.MeshingParams(
              ...
              volume_zones=[farfield, rotation_cylinder]
          ),
          ...
      )

:class:`AutomatedFarfield`
--------------------------
Settings for automatic farfield volume zone generation.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~AutomatedFarfield.name`
     - Farfield
     - Name of the :class:`AutomatedFarfield`.
   * - :py:attr:`~AutomatedFarfield.method`
     - auto
     - By default, the farfield shape will be automatically determined by the volume mesher. The user can prescribe the farfield shape if needed.

       - :code:`auto`: The mesher will generate a Sphere or a semi-sphere based on the bounding box of the geometry

         - Full sphere if min{Y} < 0 and max{Y} > 0
         - +Y semi sphere if min{Y} = 0 and max{Y} > 0
         - -Y semi sphere if min{Y} <> 0 and max{Y} = 0

       - :code:`quasi-3d`: Thin disk will be generated for quasi 3D cases. Both sides of the farfield disk will be treated as "symmetric plane".

:class:`RotationCylinder`
-------------------------
- The mesh on :class:`RotationCylinder` is guaranteed to be concentric.
- The :class:`RotationCylinder` is designed to enclose other objects, but it can’t intersect with other objects.
- Users could create a donut-shape :class:`RotationCylinder` and put their stationary centerbody in the middle.
- This type of volume zone can be used to generate volume zone compatible with :class:`~flow360.Rotation` model.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~RotationCylinder.name`
     - Rotation cylinder
     - Name of the :class:`RotationCylinder`.
   * - :py:attr:`~RotationCylinder.spacing_axial`
     - REQUIRED
     - [float] Spacing along the axial direction.
   * - :py:attr:`~RotationCylinder.spacing_radial`
     - REQUIRED
     - [float] Spacing along the radial direction.
   * - :py:attr:`~RotationCylinder.spacing_circumferential`
     - REQUIRED
     - [float] Spacing along the circumferential direction.
   * - :py:attr:`~RotationCylinder.entities` (:class:`Cylinder`)
     - REQUIRED
     - List of volumes where :py:class:`RotationCylinder` will be applied.
   * - :py:attr:`~RotationCylinder.enclosed_entities` (:class:`Cylinder`, :class:`Surface`)
     - OPTIONAL
     - List of entities enclosed by :class:`RotationCylinder`, can be a :class:`Cylinder` and/or :class:`Surface`.

.. _Refinements:

:py:attr:`~MeshingParams.refinements`
=====================================
Describes the settings used for defining additional settings for refining the mesh on top of :class:`MeshingDefaults`.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Refinement
     - Default
     - Description
   * - :class:`SurfaceEdgeRefinement`
     - OPTIONAL
     - Setting for growing anisotropic layers orthogonal to the specified :class:`Edge` (s).
   * - :class:`SurfaceRefinement`
     - OPTIONAL
     - Setting for refining surface elements for given :class:`Surface` (s).
   * - :class:`BoundaryLayer`
     - OPTIONAL
     - Setting for growing anisotropic layers orthogonal to the specified :class:`Surface` (s).
   * - :class:`PassiveSpacing`
     - OPTIONAL
     - Setting for controlling the mesh spacing either through adjacent :class:`Surface`'s meshing settings or retaining the current surface mesh. 
   * - :class:`UniformRefinement`
     - OPTIONAL
     - Setting for uniform spacing refinement inside specified region of mesh.
   * - :class:`AxisymmetricRefinement`
     - OPTIONAL
     - Setting for axisymmetric refinement inside specified region of mesh.

Example of defining :py:attr:`~MeshingParams.refinements` in Python API\:

.. code-block:: python

  import flow360 as fl
  with fl.SI_unit_system:
      farfield = fl.AutomatedFarfield(name="Farfield", method="auto")
      cylinder = fl.Cylinder(name="Cylinder", axis=[0, 1, 0], center=[1, 1, 1], outer_radius=10, height=3)
      rotation_cylinder = fl.RotationCylinder(
          name="RotationCylinder",
          spacing_axial=0.5,
          spacing_circumferential=0.3,
          spacing_radial=1.5,
          entities=cylinder
      )
      params = fl.SimulationParams(
          meshing=fl.MeshingParams(
              ...
              volume_zones=[farfield, rotation_cylinder]
          ),
          ...
      )

.. _SurfaceEdgeRefinement:

:class:`SurfaceEdgeRefinement`
------------------------------
Setting for growing anisotropic layers orthogonal to the specified :class:`Edge` (s).

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~SurfaceEdgeRefinement.name`
     - SurfaceEdgeRefinement
     - Name of the :class:`SurfaceEdgeRefinement`.
   * - :py:attr:`~SurfaceEdgeRefinement.method`
     - REQUIRED
     - Method for determining the spacing. Can be one of:

       - :class:`AngleBasedRefinement`: Surface edge refinement by specifying curvature resolution angle.
       - :class:`HeightBasedRefinement`: Surface edge refinement by specifying first layer height of the anisotropic layers.
       - :class:`AspectRatioBasedRefinement`: Surface edge refinement by specifying maximum aspect ratio of the anisotropic cells.
       - :class:`ProjectAnisoSpacing`: Project the anisotropic spacing from neighboring faces to the edge.

   * - :py:attr:`~SurfaceEdgeRefinement.entities` (:class:`Edge`)
     - REQUIRED
     - List of :class:`Edge` (s), which will be used for :class:`SurfaceEdgeRefinement`.

.. _SurfaceRefinement:

:class:`SurfaceRefinement`
--------------------------
Setting for refining surface elements for given :class:`Surface`.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~SurfaceRefinement.name`
     - SurfaceRefinement
     - Name of the :class:`SurfaceRefinement`.
   * - :py:attr:`~SurfaceRefinement.max_edge_length`
     - REQUIRED
     - [float] Maximum edge length of surface cells.
   * - :py:attr:`~SurfaceRefinement.entities` (:class:`Surface`)
     - REQUIRED
     - List of :class:`Surface` (s), which will be used for :class:`SurfaceRefinement`.

.. _BoundaryLayer:

:class:`BoundaryLayer`
----------------------
Setting for growing anisotropic layers orthogonal to the specified :class:`Surface` (s).

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~BoundaryLayer.name`
     - BoundaryLayerRefinement
     - Name of the :class:`BoundaryLayer`.
   * - :py:attr:`~BoundaryLayer.first_layer_thickness`
     - REQUIRED
     - [float] First layer thickness for volumetric anisotropic layers grown from given :class:`Surface` (s).
   * - :py:attr:`~BoundaryLayer.entities` (:class:`Surface`)
     - REQUIRED
     - List of :class:`Surface` (s), which will be used for :class:`BoundaryLayer`.

.. _PassiveSpacing:

:class:`PassiveSpacing`
-----------------------
Passively control the mesh spacing either through the adjacent :class:`Surface`'s meshing settings or doing nothing to change existing surface mesh at all.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~PassiveSpacing.name`
     - PassiveSpacingRefinement
     - Name of the :class:`PassiveSpacing`.
   * - :py:attr:`~PassiveSpacing.type`
     - REQUIRED
     - [str] Type of the :class:`PassiveSpacing` refinement. Can be one of:

       - projected: Turns off anisotropic layers growing for selected :class:`Surface` (s) and projects anisotropic spacing from the neighboring volumes to this face.
       - unchanged: Turns off anisotropic layers growing for selected :class:`Surface` (s), the surface mesh will remain unaltered when populating the volume mesh.

   * - :py:attr:`~PassiveSpacing.entities` (:class:`Surface`)
     - REQUIRED
     - List of :class:`Surface` (s), which will be used for :class:`PassiveSpacing`.

.. _UniformRefinement:

:class:`UniformRefinement`
--------------------------
Uniform spacing refinement inside specified region of mesh.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~UniformRefinement.name`
     - UniformRefinement
     - Name of the :class:`UniformRefinement`.
   * - :py:attr:`~UniformRefinement.spacing`
     - REQUIRED
     - The required refinement spacing.
   * - :py:attr:`~UniformRefinement.entities` (:class:`Cylinder`, :class:`Box`)
     - REQUIRED
     - List of :class:`Cylinder` (s) and/or :class:`Box` (s), which will be used for :class:`UniformRefinement`.

.. _AxisymmetricRefinement:

:class:`AxisymmetricRefinement`
-------------------------------
- The mesh inside the :class:`AxisymmetricRefinement` is semi-structured.
- The :class:`AxisymmetricRefinement` cannot enclose/intersect with other objects.
- Users could create a donut-shaped :class:`AxisymmetricRefinement` and place their hub/centerbody in the middle.
- :class:`AxisymmetricRefinement` can be used for resolving the strong flow gradient along the axial direction for actuator or BET disks.
- The spacings along the axial, radial and circumferential directions can be adjusted independently.

.. list-table::
   :widths: 20 10 70
   :header-rows: 1

   * - Option
     - Default
     - Description
   * - :py:attr:`~AxisymmetricRefinement.name`
     - AxisymmetricRefinement
     - Name of the :class:`AxisymmetricRefinement`.
   * - :py:attr:`~AxisymmetricRefinement.spacing_axial`
     - REQUIRED
     - [float] Spacing along the axial direction.
   * - :py:attr:`~AxisymmetricRefinement.spacing_radial`
     - REQUIRED
     - [float] Spacing along the radial direction.
   * - :py:attr:`~AxisymmetricRefinement.spacing_circumferential`
     - REQUIRED
     - [float] Spacing along the circumferential direction.
   * - :py:attr:`~AxisymmetricRefinement.entities` (:class:`Cylinder`)
     - REQUIRED
     - List of :class:`Cylinder` (s), which will be used for :class:`AxisymmetricRefinement`.