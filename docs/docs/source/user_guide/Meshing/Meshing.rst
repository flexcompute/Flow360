.. _meshing:

*******
Meshing
*******

Flow360 offers different approaches to meshing, which are developed to suit different needs and workflows. The meshing algorithms can be divided into surface and volume meshers, each having its own advantages. They are available in four combinations forming full meshing workflows.

Available workflows
===================

Flow360 supports multiple meshing workflows. This section provides a high-level map of the available options and links to the workflows.
There are currently four mesher workflows available in Flow360 summarised in :ref:`fig-mesher-workflows` below:

.. figure:: ./Figures/mesher_workflows.png
   :name: fig-mesher-workflows
   :align: center
   :width: 80%

   Mesher workflows

Go to :ref:`Example Library <python_api_example_library>` to see more available examples for each mesher workflow. Here is a high-level overview of the mesher workflows advantages and disadvantages:

.. list-table:: Mesher workflow overview
   :header-rows: 1
   :widths: 25 65

   * - Workflow
     - Characteristics
   * - Legacy mesher
     - - First mesher workflow in Flow360
       - Supports automated Quasi-3D meshing
       - No longer maintained
       - Needs watertight geometry
   * - New mesher
     - - Current standard workflow
       - Ongoing improvements
       - Recommended starting point for new users/projects
       - Needs watertight geometry
   * - GeometryAI mesher
     - - More robust setup with automated/intelligent meshing decisions
       - More robust CAD import and geometry processing
       - Can reduce manual tuning for common cases
       - Represents sharp edges implicitly rather than conforming the mesh to feature lines
   * - Snappy mesher
     - - Leverages a widely used OpenFOAM meshing utility
       - Flexible for complex geometries; many established best practices exist
       - Handles dirty CAD models with intersections and gaps

Selecting a workflow
====================

Which surface and volume mesher run is controlled by two toggles, the *beta mesher* toggle and the *GAI* toggle, available in both the WebUI and the Python API:

.. list-table::
   :header-rows: 1
   :widths: 20 15 35 30

   * - Beta mesher toggle
     - GAI toggle
     - Surface mesher
     - Volume mesher
   * - Off
     - n/a
     - Legacy mesher
     - Legacy mesher
   * - On
     - Off
     - Beta surface mesher
     - Volume Mesher
   * - On
     - On
     - GAI surface mesher
     - Volume Mesher

The volume stage is the *Volume Mesher* whenever the *beta mesher* toggle is on, regardless of the *GAI* toggle; the *GAI* toggle only selects which surface mesher feeds it. The *snappy* workflow is selected through its own surface-mesher option and also feeds the *Volume Mesher*.

Pick a mesher
=============

Each column below corresponds to a full meshing workflow. Click any card to open its dedicated page.

.. _surface_meshers_user_guide:
.. _volume_meshers_user_guide:

.. grid:: 4
   :gutter: 3

   .. grid-item::

      .. container:: mesher-workflow-title

         Legacy mesher

   .. grid-item::

      .. container:: mesher-workflow-title

         New mesher

   .. grid-item::

      .. container:: mesher-workflow-title

         GeometryAI mesher

   .. grid-item::

      .. container:: mesher-workflow-title

         Snappy mesher

.. grid:: 4
   :gutter: 3

   .. grid-item-card:: Legacy Surface Mesher
      :link: legacy_surface_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: Surface Mesher
      :link: new_surface_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: GeometryAI Surface Mesher
      :link: gai_surface_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: Snappy Surface Mesher
      :link: snappy_surface_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

.. grid:: 4
   :gutter: 3

   .. grid-item::
      :class: mesher-workflow-arrow

      ↓

   .. grid-item::
      :class: mesher-workflow-arrow

      ↓

   .. grid-item::
      :class: mesher-workflow-arrow

      ↓

   .. grid-item::
      :class: mesher-workflow-arrow

      ↓

.. grid:: 4
   :gutter: 3

   .. grid-item-card:: Legacy Volume Mesher
      :link: legacy_volume_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: Volume Mesher
      :link: new_volume_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: Volume Mesher
      :link: new_volume_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

   .. grid-item-card:: Volume Mesher
      :link: new_volume_mesher_user_guide
      :link-type: ref
      :class-card: mesher-card

.. toctree::
    :hidden:

    SurfaceMesher
    GeometryAI/index
    SnappySurfaceMesher
    LegacySurfaceMesher
    VolumeMesher
    LegacyVolumeMesher
