release-25.2
=============

Released: 28 February 2025

A new version of Flow360, release-25.2, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.

Solver
------

*New features / Improvements*

1. Further improvement of adaptive CFL to improve robustness of the flow solver
2. Support for transient moving reference frame (MRF) simulations
3. Wall roughness support for resolved-boundary-layer mesh
4. Permeable surface support for aeroacoustics
5. Deck and Renard shielding function (ZDES) for both SA and SST
6. Bleed zone boundary condition

*Bugfixes*

1. Fixed a bug in mass inflow BC and more logging info for massInflow/massOutflow
2. Fixed a minor bug in DDES/SA shielding function
3. Improved initialization of supersonic cases consistent with inflow Mach
4. Improved gradient calculation at wing junctures - Leading to inaccurate skin friction
5. Added porous media moment output

Mesher
------

1. Improved the quality and speed of the boundary layer generation in the beta mesher
2. Added support for spatially-varying boundary layer parameters in the beta mesher

Postprocessing 
--------------

*New features / Improvements*

1. Support for slicing boundary data (Available only through python interface)
2. Improved accuracy of cumulative quantities

Python Client
-------------

*New features / Improvements*

1. Added report generation plugin to present the postprocessed results in well-formatted PDF documentation
2. Added interface to show all of user's projects filtered by project name
3. Support for printing the project tree in the command line interface and retrieving the project asset using the short ID from the printed project tree
4. Added a new surface mesh entry point for the python API
5. Added missing interface for "rotation_correction" of kOmegaSST
6. Function that creates operating conditions from Mach and muRef (Available only through python interface)
7. Changed default temperature unit for Imperial Unit System from Rankine to Fahrenheit
8. Added function to compute Flow360 Reynolds number from operating condition
9. Additional validation locally before submitting to the cloud
10. Allowing vorticityMagnitude in output field and also as isosurface field
11. Added `WallRotation` for easy setup of rotating wall velocity
12. Added limited forward compatibility
13. Enable Python client to be used in various cloud environments

Installation
------------

To install the new package:

.. code-block:: bash

   pip install "flow360==25.2.*"

This will automatically select the latest patch version of 25.2.