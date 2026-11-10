release-22.3.3.0
================

Released: 21st October 2022

A new version of Flow360, release-22.3.3.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added monitor output for probing solution quantities at specific points in the flow domain.

2. Added isosurface output for visualizing an isosurface of a solution quantity at a fixed value.

3. Faster processing of slice, surface and volume output quantities from flow solution.

4. Volume output of solution quantities is now stored in a single file.

5. Added capability of simulating aerostructure interactions using sliding interfaces.

6. Added mesh quality metrics in mesh processing log output.

*Resolved issues*

1. Improved error messaging in solver log output and json validation.

2. Fixed issues with restarting cases for large meshes (>150 million nodes).

3. Improved accuracy of SpalartAllmaras DDES model.

4. Fixed bug in slice output visualization.

5. Fixed non-determinisitic execution of transition model.

6. Fixed VelocityRelative in volume output

Automated Meshing
-----------------

*New features*

1. Added support for 2D meshing (quasi-3D).

2. Added support for meshing narrow gaps between surfaces in close proximity.

3. Each labelled surface and edge can be inspected on WebUI in the 3D viewer.

*Changes*

1. Parameter `sources` renamed to `refinement` for volume meshing.


flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/

*New features*

1. Added validation of input files before the submission of new geometries, surface meshes and cases.
