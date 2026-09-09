release-25.9
=============

Released: 25 March 2026

A new version of Flow360, release-25.9, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features*

#. Thermally Perfect Gas (TPG) simulation (supported only through Python client)
#. Krylov linear solver for steady simulations
#. Support for shear-layer adapted (SLA) functionality in DDES/ZDES
#. Allow for including external acceleration (like gravity) in flow simulations
#. New wall function based on inner layer laws of the wall
#. Support to read structured mesh in CGNS format

*Improvements*

#. Up to 3x performance improvement in computation of geometric quantities in mesh preprocessing
#. Stability improvement of the Navier-Stokes limiter for low dissipation kappa MUSCL simulations
#. Robustness improvement of wall model and kOmegaSST solver for automotive applications

*Fixes*

#. Bug fix for error faced in ``SymmetryPlane`` boundaries composed of more than one patch

Post-processing
---------------

*New features*

#. ``write_single_file`` capability for Paraview outputs
#. Support for selecting faces and setting the number of segments in Force Distribution Output

Geometry and Meshing
--------------------

*Geometry AI Surface Mesher*

#. Added control for target surface mesh node count
#. Automatic removal of hidden geometry parts irrelevant to external flow simulations
#. Improved detection and resolution of thin geometry features
#. ~5x faster auto symmetry plane meshing
#. Improved accuracy of surface mesh near thin geometries
#. Automatic early termination of GeometryAI to prevent resource exhaustion
#. Show CAD topological edges in web UI geometry view
#. Reduced execution time

*New Surface Mesher*

#. Added control for target surface mesh node count
#. Improved user-facing logs and progress updates

*New Volume Mesher*

#. Added support for spherical uniform refinement zones in volume meshing
#. Added support for spherical sliding interfaces for rotating machinery
#. Added support for axisymmetric body uniform refinements
#. Added support for base spacing control of farfield octree mesh
#. Added new volume mesh diagnostic metrics:

   a. Equiangular skewness
   b. Minimum/maximum included angle
   c. Volume ratio

#. Improved resource logging

*Snappy Surface Mesher*

#. Improved corner and edge reconstruction

*Fixes*

#. Fixed boundary layer edge splitting bug for nodes at projection faces

Python Client
-------------

*Possible breaking change*

- We deprecated support for Python 3.9 to apply security patches.

Web UI
------

*New features*

#. Virtual GPU Scheduler -- Eligible users now have access to a dedicated Virtual GPU (vGPU) queue. Skip the general queue entirely and submit runs with guaranteed priority execution. Job priority can be managed directly from the Virtual GPU Scheduler tab.

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/vGPU%20selection.png
      :align: center
      :class: release-figure

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/vGPU%20scheduler.png
      :align: center
      :class: release-figure

#. Company-wide storage -- Storage usage is now shown on the dashboard for users with Company-wide Storage. Warnings appear at 90%, 95%, and 100% usage. After the limit is reached, the normal Flex Credit storage rate will be applied.

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/Company%20storage%20limit.png
      :align: center
      :class: release-figure

#. Box selection in the 3D viewer -- Use Shift + drag to select multiple entities by drawing a box, or Ctrl + Shift + drag to deselect from an existing selection.

   .. raw:: html

      <div class="release-video">
         <iframe src="https://hs.flexcompute.com/share/hubspotvideo/209604737591" allowfullscreen></iframe>
      </div>

#. Default mesher preference -- Users can set a preferred default mesher in the Preferences screen.

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/Default%20mesher%20selection.png
      :align: center
      :class: release-figure

#. Keyboard shortcut palette -- A shortcut reference palette is now accessible directly from the Project tree.

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/Keyboard%20shortcut.png
      :align: center
      :class: release-figure

#. Geometry accuracy warnings -- The meshing form now flags geometry accuracy values that are too coarse or too fine, helping catch misconfiguration before running.

   .. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/Geometry%20accuarcy%20warning.png
      :align: center
      :class: release-figure

#. Compressed geometry file upload -- ``.zst``, ``.gz``, and ``.bz2`` geometry files can now be uploaded directly without manual decompression.
#. ``TimeAverageSurfaceProbeOutput`` -- Now available in the outputs configuration panel.
#. CustomVolume for Solid models -- CustomVolume can now be selected when configuring Solid model zones.
#. Wind tunnel surfaces in custom volumes -- Wind tunnel surfaces can now be selected when defining custom volumes, enabling multizone mesh workflows.

*Improvements*

#. BET input: units and doc links -- All BET parameters now show their units, with ``?`` links to relevant documentation.

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.9/BET%20input.png
   :align: center
   :class: release-figure

#. Slice plane visibility when zooming -- The slice plane boundary stays visible at any zoom level, making it easier to understand slice placement.

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/209614051126" allowfullscreen></iframe>
   </div>

#. Force/moment plot titles -- Titles now consistently say "coefficients" (e.g. "Total force coefficients"), making clear the values are non-dimensional coefficients.
#. Geometry and Mesh step timing -- Elapsed time is now shown for the Geometry and Mesh processing steps.
#. Surface mesh export as ``.cgns`` -- ``.ugrid`` surface meshes can now be exported as ``.cgns`` for Tecplot compatibility.
#. Entity tags in project export/import -- Entity tag data is now preserved when importing or exporting a project.
#. User-defined coordinate system form -- The coordinate system configuration form has been improved for usability.

*Bug Fixes*

#. Postprocessing clip -- Clip filter now works correctly for both above and below threshold.
#. Logarithmic min/max slider -- The min/max range slider now also uses logarithmic scale when logarithmic color mapping is active.
#. Case compare shows all cases -- Fixed: case comparison view was not displaying all selected cases.
#. Color card range on LOD switch -- The color card min/max values no longer reset when switching between levels of detail.
#. Mode switching while loading -- Users can now switch viewer modes (geometry, mesh, results) before the project finishes loading.
#. BET disk translator output -- BET disk translators now produce results consistent with V1 translators.
#. mapbc file rename on upload -- The ``.mapbc`` file is now correctly renamed to match the uploaded ``.ugrid`` filename.

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.9.*"

This will automatically select the latest patch version of 25.9.
