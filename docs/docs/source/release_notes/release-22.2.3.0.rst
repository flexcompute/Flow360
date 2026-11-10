release-22.2.3.0
================

Released: 11th July 2022

A new version of Flow360, release-22.2.3.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added user defined dynamics for defined alpha controller and BET Omega controller.

2. Added support for BET inside a sliding interface for a rotating reference frame.

3. Added Quadratic Constitutive Relation to turbulence models: SA and SST.

4. Added Surface Time Solution Averaging Animation.

5. Added output of maximum residual location in solver log and as a csv file.

6. Added support for importing CGNS meshes from ANSA.

*Resolved issues*

1. Improved sliding interface implementation for kOmegaSST turbulence model.

2. Significantly reduced grid sensitivity of AFT transition model.

3. Improved numerical robustness and convergence behavior of Spalart Allmaras, kOmegaSST turbulence models and AFT transition model.

*Documentation updates*

1. Alpha controller json input has been changed to use the new user defined dynamics feature.


Automated Meshing
-----------------

*New features*

1. Added rotational interface support for automated meshing.

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

1. Improve usability of Python Client by providing a download/upload progress bar.

Web UI
------

*New features*

1. Added archive functionality for cases: see how can I archive my case.
2. Added multi-select and batch process: delete/archive/restore.
3. Added interactive 3D view for surface mesh.
4. Added interactive 3D view for surface to volume mesh generation. Shows refinement zones and actuator disks. 
5. Introduced new billing system.
