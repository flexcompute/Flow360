release-23.2.3.0
================

Released: 6th July 2023

A new version of Flow360, release-23.2.3.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added conjugate heat transfer modeling between fluid and solid interfaces.

2. Added unsteady adaptive CFL feature for automatic setting of CFL based on solution convergence.

3. Improved accuracy of gradient computation on highly anisotropic meshes.

4. Add user defined velocity direction in SubsonicInflowVelocity boundary condition.

5. Add mesh metrics computation on volume mesh upload.

6. Added yPlus check in case validation script.

7. Improved efficiency of case forking mechanism.

*Resolved issues*

1. Added missing time-averaged outputs for certain surface and volume outputs.

2. Fix monitor output when no output field is specified.

3. Update csv file outputs when case has diverged.

4. Fix residual volume and surface output to be consistent with solver.

5. Fix partitioned volume output Tecplot files.

Automated Meshing
-----------------

*New features*

1. Auto-Meshing for internal flows: generate meshes for internal flow simulations.

*Changes*

1. Updated tolerance for symmetry planes: tolerance is now calculated as 0.01 of shortest edge of the model.

2. No Group Names attribute required in CSM file: set boundary conditions after mesh generation, without the need for group names.

flow360client
-------------

Although recent versions of flow360client will still work, it is
highly recommended to upgrade to the latest version for more
convenient capabilities:

Usage:

- If downloading for the first time: :code:`pip3 install flow360client`

- If upgrading from an older version: :code:`pip3 install --upgrade flow360client`

Here is the flow360client on PyPI\: https://pypi.org/project/flow360client/
