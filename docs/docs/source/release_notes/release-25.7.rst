release-25.7
=============

Released: 9 November 2025

A new version of Flow360, release-25.7, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features / Improvements*

1. Added support for porous-jump boundary condition (see reference :class:`~flow360.PorousJump`)
2. Added support for low Reynolds number correction in SA model to improve skin friction estimation (see reference :class:`~flow360.SpalartAllmaras`)
3. Added support for allowing Actuator Disk to be in a rotation zone

*Bugfixes*

1. Fixed a bug associated with cases having both CHT and rotating zones

Post-processing
---------------

*New features / Improvements*

1. Added support for clipping of iso-surface based on a given wall distance
2. Added support for including monitor names in the output file

Geometry and Meshing
--------------------

*Geometry Processing:*

1. Support for extracting built-in face name of uploaded CAD files and making “builtinName” a key in “Face Grouping“

*Default Surface Mesher:*

1. Enabled conformal quasi-3D mesh for periodic boundary condition (Python API only)
2. Enabled user control of farfield size (Python API only)
3. Improved robustness of the default surface mesher

*Beta Volume Mesher:*

1. Added option to create multi-zone meshes with custom volumes
   - Currently supported in Python client. Web UI integration pending.
2. Added support for axisymmetric volumes (bodies of revolution) as rotation zones
   - Currently supported in Python client. Web UI implemented and under testing.
3. Improved robustness of symmetry plane handling
4. Implemented volume meshing progress updates for Web UI
5. Added option to repair patch IDs of near-planar faces used for boundary layer projection

Python Client
-------------

*New features / Improvements*

1. Added v2 folder and a *folder parameter* for project creation
2. Added *rename* method for v2 cloud resources
3. Added *tag filtering* option for ``Project.get_case_ids``, ``Project.get_project_ids``, and ``metadata``
4. Allow grouping surface forces by bodyGroup or boundary models
5. Added option to compute BET/AD/PM force coefficients
6. Added option to reformat BETDisk output CSV headers
7. Added local/global options for ``geometry_accuracy`` and ``preserve_thin_geometry``
8. Allow ``CompressibleIsentropic`` solver type for ``LiquidOperatingCondition``
9. Added dry-run support for ``run_case()``

Web UI
------

*New features / Improvements*

1. Added a Probe tool to show or find the point location

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.7/Probe%20tool%20to%20show%20or%20find%20the%20point%20location.png
   :align: center
   :class: release-figure

2. View multiple simulation results in interactive viewers

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199440400741" allowfullscreen></iframe>
   </div>  

3. Added search, sort, and filter to the Entities list

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199440544682" allowfullscreen></iframe>
   </div>

4. Added time-averaging isosurface output — users can calculate averages from a specific physical step and output results periodically.

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.7/Time-averaging%20isosurface%20output.png
   :align: center
   :class: release-figure
   
5. Implemented an intuitive multi-select mode in drop-down components for improved user efficiency.

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199441334068" allowfullscreen></iframe>
   </div>   

6. Improved rendering of the model in the Web UI.

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199440770724" allowfullscreen></iframe>
   </div>

*Fixes & Enhancements*

1. Introduced a flexible graphics loading feature: low-resolution graphics load first while high-resolution versions stream in seamlessly; users can still set preferences in Account.
   
2. Allow project to open with progress indicator.

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199440041570" allowfullscreen></iframe>
   </div>

3. Compute the reference area automatically by selecting faces and projection direction (X, Y, Z).

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/199437275009" allowfullscreen></iframe>
   </div>

4. Improved visualization layout and interaction — surfaces, slices, isosurfaces, and streamlines now listed under *Visualization* in the left sidebar.

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.7/Improve%20the%20visualization%20layout%20and%20interaction.png
   :align: center
   :class: release-figure
   
Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.7.*"

This will automatically select the latest patch version of 25.7.

