release-25.8
=============

Released: 22 January 2026

A new version of Flow360, release-25.8, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features*

1. New Aeroacoustic Solver

   a. Performance improvements (>2x)
   b. Capability to accommodate higher number of observers
   c. Support to allow FWH solver to have different pace from flow solver
   d. Support for restart

2. Performance Improvement of mesh partitioner (3x)
3. Added support for allowing legacy user defined expressions for ``totalPressureRatio`` and ``totalTemperatureRatio`` in ``SubsonicInflow`` boundary condition
4. Exposed interpolation tolerance to the interface for multi-zone cases

*Fixes & Improvements*

1. Improved robustness of ``wallModel`` cases where some boundary nodes do not have any internal neighbors
2. Fixed a minor bug in Slater boundary condition
3. Fixed a bug which caused 10% higher memory usage in the solver

Post-processing
---------------

*New features*

1. Added more support for streamlines:

   a. Color streamlines by selected fields
   b. Streamlines of time-averaged velocity field
   c. Ribbon visualization for streamlines

2. Added support for outputting quantities and computing integrals on imported surfaces
3. Added support for time-averaged force distribution data output
4. Added support for having custom directions for force distribution plots
5. Added support for having custom force output from ``Wall``, ``BETDisk``, ``ActuatorDisk`` and ``PorousMedia`` models
6. Added support for computing moving statistics of ``ProbeOutput``, ``SurfaceProbeOutput``, ``SurfaceIntegralOutput`` and ``ForceOutput``

*Fixes & Improvements*

1. Fixed bug for time-averaged monitors when starting step for time averaging is greater than zero
2. Fixed bug for ``surfaceOutput`` with user-defined dynamics (UDD) and ``singleFile=true``

Geometry and Meshing
--------------------

*Snappy Surface Mesher*

We have introduced a new surface meshing workflow based on OpenFOAM's snappyHexMesh, which is fully integrated with our volume mesher.

- Generate meshes directly from STL geometry
- Accessible via the Python API, with mesh inspection and results visualization available in the WebUI
- Well suited for motorsport geometries
- More tolerant of dirty or imperfect CAD compared to traditional body-fitted meshing approaches
- Supports multizone configurations, including MRF and porous media (zone interfaces defined in geometry)

*Geometry AI (GAI) Surface Mesher*

This release introduces a new Geometry AI-based surface meshing workflow, built on our next-generation geometry processing technology and fully integrated with the volume mesher.

Geometry AI significantly improves meshing robustness for external aerodynamics cases, especially when working with imperfect CAD geometry. This release supports geometries with minor defects, such as small missing faces or mismatched edges, and lays the foundation for future releases that will handle progressively more challenging and dirty geometry.

The Geometry AI surface mesher introduces several new, unique features that are not available in our other surface meshing workflows:

- Import and mesh analytic (BRep/CAD) and discrete (surface meshes) geometry in the same project
- Geometry defeaturing: control the smallest feature size in the geometry that needs to be resolved accurately by setting the geometry accuracy parameter (globally or at the face level)
- Gap sealing: automatically seal holes in the geometry or gaps between solids that are smaller than a user-specified size
- Control mesh refinement near face boundaries
- Spatial transformations and mirroring of bodies before meshing
- Simple boolean operations of bodies before meshing (relevant solids must be defined in separate CAD/surface mesh files)
- Generate full-body meshes from half-body geometry, and vice-versa
- Analytical specification of a wind-tunnel geometry to be used as a farfield
- Import, view, and edit geometry resources after project creation
- Surface remeshing capabilities to support sliding interfaces (rotation zones) that intersect the geometry
- Support for curvature resolution angle specification at the face level

*(Legacy) Surface Mesher*

1. Optimized points distribution for higher quality and improved accuracy

*(Beta) Surface Mesher*

1. Added support for curvature resolution angle specification at the face level
2. Improved the preprocessing of the curvature metric to improve surface mesh quality
3. Added support for visualizing surface mesh diagnostic metrics in the web UI to help assess mesh quality. This is available in all surface meshing workflows.

*(Beta) Volume Mesher*

1. Added option for generating tetrahedra-only custom volume zones
2. Added support for rotation zones that intersect the geometry
3. Improved the performance of planar face remeshing after boundary layer projection by 40x on average
4. Added support for visualizing volume mesh slices (default or user-defined) in the web UI, together with volume mesh diagnostic metrics

*Fixes & Improvements*

1. Reduced the tessellation size of models with edge segments that are smaller than the geometry precision
2. Fixed a bug in boundary layer projection at the intersection of two planar faces

Python Client
-------------

*New features*

1. Python client now supports Python 3.13 (3.9~3.13)
2. Added functionality to change BET csv header
3. Added support to customize ``Stopping Criteria`` based on ``ProbeOutput``, ``SurfaceProbeOutput``, ``SurfaceIntegralOutput`` and ``ForceOutput``

Web UI
------

*New features*

1. Added support for defining the wind tunnel farfield

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Wind%20tunnel%20definition.png
   :align: center
   :class: release-figure

2. Support for user-defined stopping criteria for steady and unsteady simulations

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/User-defined%20stopping%20criteria.png
   :align: center
   :class: release-figure

3. Added support for generating force outputs based on wall boundary conditions, as well as the Porous Media, Actuator Disk, and Blade Element Theory (BET) models.

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Force%20output%20from%20selected%20boundaries%20and%20models.png
   :align: center
   :class: release-figure

4. Added support for creating time-averaged streamline output

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Time-averaged%20streamline%20output.png
   :align: center
   :class: release-figure

5. Added support for creating render output (animation)

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/205592606269" allowfullscreen></iframe>
   </div>

6. Added support for creating the custom volume (beta volume mesher only)

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Custom%20volume.png
   :align: center
   :class: release-figure

7. Added support for creating an arbitrary axisymmetric volume (beta volume mesher only)

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Arbitrary%20axisymmetric%20volume.png
   :align: center
   :class: release-figure

8. Added support for mirroring body group

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Mirror%20body%20group.png
   :align: center
   :class: release-figure

9. Added support for defining user-defined coordinate systems to entities

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/User-defined%20coordinate.png
   :align: center
   :class: release-figure

10. Support for adding and/or removing components (CAD or mesh files) from the geometry resource to compare the effect on aerodynamic design (Geometry AI only)

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/View%20and%20edit.png
   :align: center
   :class: release-figure

11. Added support for importing sample surfaces (for example baffle surfaces) from files to integrate and visualize results

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Imported%20sample%20surface.png
   :align: center
   :class: release-figure

12. Support for evaluating and identifying surface mesh quality issues

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Surface%20mesh%20diagnostics.png
   :align: center
   :class: release-figure

13. Support for outputting volume mesh slices by default, creating user-defined slices, and visualizing volume mesh quality diagnostics on slices

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Volume%20mesh%20diagnostics.png
   :align: center
   :class: release-figure

14. Support for selecting time frames to view results and understand the flow results for unsteady simulations

.. raw:: html

   <div class="release-video">
       <iframe src="https://hs.flexcompute.com/share/hubspotvideo/205587432547" allowfullscreen></iframe>
   </div>

*Fixes & Improvements*

1. Added progress info when loading 3D data

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Progress%20info%20when%20loading%203D%20data.png
   :align: center
   :class: release-figure

2. Added both the standard deviation and averaged forces to the Dashboard.

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Dashboard%20std%20and%20avg%20forces.png
   :align: center
   :class: release-figure

3. Added moving averages to force and heat transfer plots in Monitor

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Moving%20averages.png
   :align: center
   :class: release-figure

4. Added contributions from BET/AD/Porous media

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Added%20contributions%20from%20ADBETporous%20media.png
   :align: center
   :class: release-figure

5. Added case compare versus case name/id/tag and more group by options

.. image:: https://hs.flexcompute.com/hubfs/Flow360%20release-25.8/Case%20compare%20extra%20options.png
   :align: center
   :class: release-figure

Documentation
-------------

The documentation page has been refactored for better user experience. The key improvements involve:

- **Enhanced Navigation:** A refined structure for faster searching.
- **Expanded WebUI Library:** More examples to help you navigate the interface effectively.
- **Python API Tutorials:** Over 20 new Jupyter notebooks showcasing various features and workflows.
- **Comprehensive User Guide:** New sections explaining the core concepts and functions of the Flow360 solver.

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.8.*"

This will automatically select the latest patch version of 25.8.
