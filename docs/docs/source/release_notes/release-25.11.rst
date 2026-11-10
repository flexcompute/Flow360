release-25.11
==============

Released: 9 September 2026

A new version of Flow360, release-25.11, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features/Improvements*

#. Support for spatially varying ``velocity_direction`` in ``Inflow`` boundary conditions via user defined expressions

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/spatially-varying-inflow-velocity-direction.png
      :align: center
      :class: release-figure

#. For vGPU and on-prem users: more accurate GPU memory estimation for resource allocation of the Flow360 flow solver
#. Minor improvement to the accuracy of ``Symmetry`` BC to better match to full span results
#. Improved performance of mesh processor for models with a large number of faces
#. Improved runtime performance by reducing the overhead of enforcing a large number of boundary conditions

*Bug Fixes*

#. Fixed bug where small values in initial and boundary conditions lost precision

Post-processing
---------------

*New features*

#. Added capability to get outputs with the set frequency of ``pseudoSteps`` in ``Steady`` simulations
#. Added time-derivative of pressure to the list of supported outputs
#. Added support for RMS output
#. Added support for limiting isosurface to a user-specified box
#. Added volumetric field rendering to the ``RenderOutput`` module

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/custom-video-thumbnails/volumetric-field-rendering-rotor-thumb.jpeg?width=1120&amp;name=volumetric-field-rendering-rotor-thumb.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/volumetric-field-rendering-rotor.mp4"></video>
      </div>

*Bug Fixes*

#. Added validation for hybrid model output on slices which previously caused illegal memory access error

Geometry and Meshing
--------------------

*GeometryAI Surface Mesher*

#. 2x runtime and memory reduction of end-to-end GeometryAI workflow
#. Added support for specifying boxes under ``GeometryRefinement``
#. Improved sealing capability to handle cases where a physical gap size is greater than the geometry thickness surrounding the gap. Supported only through the V2 CAD importer, and requires faces surrounding the hole to have distinct face IDs.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/gap-sealing-requires-distinct-face-ids.png
      :align: center
      :class: release-figure

#. New tessellation algorithm that produces coarser initial tessellations for GeometryAI, with roughly 100x reduction in tessellation node count and runtime. Currently only supports the V1 CAD importer.
#. Introduced a new baffle face handling option, with three options now supported: **Remove** deletes the baffle face, **Thicken** inflates the baffle face into a thin closed shell, and **Default** leaves the baffle face unmodified. Baffle face handling can now also be specified locally per face.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/baffle-face-treatment-options.png
      :align: center
      :class: release-figure

#. The y = 0 symmetry plane now follows ``geometry_accuracy`` rather than ``planar_face_tolerance``, which no longer affects GeometryAI surface meshing. A face that lies on the plane but is captured coarsely is recognized as symmetric, so a half body whose CAD is not exactly planar no longer needs the tolerance tuned by hand. A geometry standing off the plane by less than its geometry accuracy is now detected as a half body where it previously was not; set ``domain_type`` explicitly to force full-body treatment. Surface meshes are regenerated once on upgrade rather than reused.

*New Volume Mesher*

#. Improved robustness of stitching axisymmetric sliding interfaces to intersecting geometry
#. New volume mesher now honors the ``AutomatedFarfield``'s ``domainType`` if it is specified by the user, instead of always auto-detecting based on bounding box

*Bug Fixes*

#. Fixed a bug that severely degraded CGNS reader performance for volume meshes with a large number of boundaries

Python Client
-------------

*New features*

#. Added ``fl.diagnose()`` and ``flow360 diagnose``, which check connectivity and authentication to a deployment one layer at a time, and ``EnvironmentConfig.from_on_premises_url()``, which derives every endpoint from a single on-premises deployment URL
#. Added support for creating projects in the shared company workspace, through the new ``fl.Workspace`` component
#. Added support for setting a project description on creation, plus ``Project.update()`` for changing project metadata afterwards
#. Added support for downloading asset files by glob pattern, matched against the full cloud path and keeping the cloud subfolder layout locally
#. ``ReferenceGeometry.area`` can now be a ``ProjectedArea`` recipe recomputed at submission, and the new ``fl.measure`` functions expose the same measurements directly. Selections can be audited with ``draft.preview_unselected()`` before they are used. ``draft.compute_obb()`` is deprecated and will be removed in release 26; use ``fl.measure.oriented_bounding_box()`` instead.
#. Added support for assigning boundary conditions and surface outputs to the zone-to-zone interfaces of an uploaded volume mesh, so a passage can be closed off in a mesh that cannot be regenerated
#. Added CHARM as a BET disk input source
#. Added a warning when a minor solver release such as ``release-25.8.7`` is pinned; minor releases are for internal testing, so use the major release ``release-25.8`` instead
#. Improved validation, translation and submission performance on large geometries

*Bug Fixes*

#. Fixed a bug where forking a case could move it onto a different solver release; forks now inherit the parent case's release
#. Fixed a bug where a bare mass value was accepted on a molar quantity field, silently off by Avogadro's number (``fl.Gas.species_transport_model.species[i].molecular_weight``)

Web UI
------

*New features*

#. Solid/Transparent render mode -- Added a Solid/Transparent render mode for Geometry.

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/custom-video-thumbnails/solid-transparent-render-mode-thumb.jpeg?width=1120&amp;name=solid-transparent-render-mode-thumb.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/solid-transparent-render-mode.mp4"></video>
      </div>

#. Volume zone from interior seed points -- Added support for defining a volume zone via interior seed points in the GeometryAI workflow.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/volume-zone-from-interior-seed-points.png
      :align: center
      :class: release-figure

#. Inflow arrow -- Added an inflow arrow showing Alpha and Beta angles in the 3D viewer.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/inflow-arrow-with-alpha-and-beta-angles.png
      :align: center
      :class: release-figure

#. Shared workspace -- Introduced the Flow360 shared workspace, a collaborative hub that makes it easier for your team to work together.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/shared-workspace.gif
      :align: center
      :class: release-figure

#. Structured box refinement -- Added structured box refinement, which can be used for meshing porous media.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/structured-box-refinement.png
      :align: center
      :class: release-figure

#. Required and optional field indicators -- Added visual indicators of required/optional fields.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/required-and-optional-field-indicators.png
      :align: center
      :class: release-figure

#. Solution value probe -- Added a solution value probe in the visualization.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/solution-value-probe.png
      :align: center
      :class: release-figure

#. Consistent display units -- Improved unit consistency across the platform for more reliable simulation values.

   .. raw:: html

      <div class="release-video">
         <video controls preload="metadata" poster="https://hs.flexcompute.com/hs-fs/hubfs/custom-video-thumbnails/consistent-display-units-thumb.jpeg?width=1120&amp;name=consistent-display-units-thumb.jpeg" src="https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/consistent-display-units.mp4"></video>
      </div>

#. Global search -- Added global search with type filters, exact/partial match, and cross-workspace search.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.11/global-search-with-type-filters.png
      :align: center
      :class: release-figure

Retirement Notice
-----------------

Flow360 V1 (``release-24.2`` and older) will be retired in the next couple of
release cycles. If you are a Python user, you will be able to fetch the legacy
cases but you will not be able to submit new cases. You will need Python API
``release-25.x`` or lower to fetch Flow360 V1 cases, and Flow360 Python API
``release-26.x`` will drop Flow360 V1 support completely. If you are a Web UI
user, you will be able to see V1 cases but you will not be able to submit new
cases. Your legacy data will not be removed, but we recommend you download
those legacy cases to save on cloud storage cost.

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.11.*"

This will automatically select the latest patch version of 25.11.
