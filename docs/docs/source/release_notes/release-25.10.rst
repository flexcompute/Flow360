release-25.10
==============

Released: 22 June 2026

A new version of Flow360, release-25.10, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features/Improvements*

#. Improved performance and capabilities of the Krylov solver for steady state simulations
#. Support for the SLAU2 flux discretization scheme suitable for transonic and supersonic simulations
#. Support for prescribing turbulence by specifying time-varying velocity vectors on a plane, suitable for adding upstream turbulence (available through the Python client only)
#. Support for prescribing ``velocity_direction`` with respect to a non-inertial frame of reference in boundary conditions
#. Enabled more flexibility for selecting porous jump interfaces, specifying a list rather than having to specify pairs
#. Improved solver initialization time for cases with expression specifications in boundary conditions
#. Improved robustness of the kOmegaSST solver in multi-zone automotive cases

*Bug Fixes*

#. Reduced the discontinuity of physical quantities across multi-zone interfaces for kOmegaSST mode
#. Fixed a bug where the F1 blending factor was not communicated between ranks in the kOmegaSST model. This bug only impacted the diffusion term of the SST model. Testing across our full suite of test cases showed no significant differences.
#. Fixed a bug where user-specified boundary quantities were not set correctly for turbulence solvers in some limited scenarios

Post-processing
---------------

*New features*

#. Support for having the same boundary under multiple ``SurfaceOutput`` instances
#. Significant reduction in memory footprint of cases with many instantaneous slices
#. Tabular export of BET coefficient distributions
#. Added VTK-HDF and Ensight Gold output formats for volume, surface, slice, and isosurface outputs

*Bug Fixes*

#. Fixed a bug where a volume output was necessary to get slice output of the turbulence solver's ``hybridModel`` field

Geometry and Meshing
--------------------

*Geometry Import and Visualization*

#. Improved geometry visualization robustness, improved loading performance, and reduced memory requirements
#. Introduced a new CAD importer version for more robust ingestion of native CAD files. It is supported by the legacy mesher and the GeometryAI mesher.
#. Improved low-resolution visualization for slices and large face counts

*New Surface Mesher*

#. ``target_surface_node_count`` implemented for multibody meshes by globally scaling mesh size across bodies

*GeometryAI Surface Mesher*

#. Decreased influence of surface mesh crease resolution for total count target
#. Added seed-point support in GeometryAI: mark a point inside an enclosed region to designate the internal flow domain (applies when ``remove_hidden_geometry = True``)
#. Improved automatic cleanup of hidden geometry for internal flow, so only the wetted surfaces are meshed
#. Added a geometry-classification view to see how GeometryAI labeled exterior, interior, and removed geometry
#. Added feature to thicken or remove baffle faces
#. Faster surface adaptation with improved metric conformity

*New Volume Mesher*

#. Symmetry plane meshing is 4x faster on average
#. Volumes can be identified with a seed point
#. Exposed farfield mesh size growth rate and improved gradation
#. Structured refinement zones can intersect geometry
#. Axisymmetric refinement zones now accept a mesh spacing distribution along the generating profile (for example finer near the nose and coarser along the body), so each segment can be refined independently

*Snappy Surface Mesher*

#. ``BodyGroup`` replaces ``SnappyBody`` to make the interface common with other meshers
#. Added sphere ``UniformRefinement``
#. Improved logging, progress reporting, and total time

*Bug Fixes*

#. Symmetry plane meshing crashes fixed
#. Symmetry plane projection bug is fixed
#. Rotating zones with faces intersecting the body are fixed
#. Nested-rotation enclosed bodies in GeometryAI are fixed
#. GeometryAI input surface termination bug is fixed
#. Face names across multiple STL files are now unique
#. Low-resolution option memory bug is fixed

Python Client
-------------

*New features*

#. New function (``draft.compute_obb()``) for computing the oriented bounding box of selected patches
#. New function (``surface_mesh.stats``) for exposing surface mesh statistics
#. Added a new flow360 CLI for managing projects and assets, creating and running drafts, monitoring logs, downloading case results, and signing in through an interactive login workflow. Most commands return JSON for scripting and CI workflows.

*Bug Fixes*

#. Fixed a bug where projects starting from a surface mesh did not default to the beta volume mesher

Web UI
------

*New features*

#. Billing page -- See your storage usage, Flex Credit balance, and per-user usage breakdown on one page.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Billing%20page.png
      :align: center
      :class: release-figure

#. Resume interrupted mesh upload -- Resume an interrupted mesh upload instead of deleting the project and starting over.

   .. raw:: html

      <div class="release-video">
         <iframe src="https://hs.flexcompute.com/share/hubspotvideo/214927942431" allowfullscreen></iframe>
      </div>

#. In-page navigation -- Actions now open in the current page instead of a new browser tab, so you no longer accumulate tabs. A preference lets you choose the behavior.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/In-page%20navigation.png
      :align: center
      :class: release-figure

#. Expected job completion time -- Once a case starts running, the Run Status panel shows its estimated completion time.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Expected%20job%20completion%20time.png
      :align: center
      :class: release-figure

#. Visualization pipeline status -- See whether each completed case's visualization is still processing or ready, across all views.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Visualization%20pipeline%20status.png
      :align: center
      :class: release-figure

#. Automatic wheel rotation setup -- Set up wheel rotation for vehicle aerodynamics automatically, instead of configuring each wheel by hand.

   .. raw:: html

      <div class="release-video">
         <iframe src="https://hs.flexcompute.com/share/hubspotvideo/214927942411" allowfullscreen></iframe>
      </div>

#. Automatic viewpoint -- The 3D viewer frames the geometry for you, with no manual camera positioning.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Automatic%20viewpoint.png
      :align: center
      :class: release-figure

#. Actuator Disk setup -- Enter a uniform force directly and see the integrated thrust and torque while setting up an Actuator Disk.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Actuator%20Disk%20setup.png
      :align: center
      :class: release-figure

#. CSV download for the compare table -- Export the case comparison table to CSV for your own analysis and reporting.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/CSV%20download%20for%20the%20compare%20table.png
      :align: center
      :class: release-figure

#. Surface mesh statistics panel -- Check surface mesh node and triangle counts directly in the UI.

   .. image:: https://hs.flexcompute.com/hs-fs/hubfs/Flow360%20release%20notes/Flow360%20release-25.10/Surface%20mesh%20statistics%20panel.png
      :align: center
      :class: release-figure

#. Screen-normal rotation in the 3D viewer -- Rotate the model around the axis pointing out of the screen for finer control of its orientation.

   .. raw:: html

      <div class="release-video">
         <iframe src="https://hs.flexcompute.com/share/hubspotvideo/214927942410" allowfullscreen></iframe>
      </div>

#. BET disk visualization -- After importing a BET source, the 3D viewer renders each disk's blade geometry, rotation axis, and rotation-direction arrow.

   .. raw:: html

      <div class="release-video">
         <iframe src="https://hs.flexcompute.com/share/hubspotvideo/214920454197" allowfullscreen></iframe>
      </div>

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.10.*"

This will automatically select the latest patch version of 25.10.
