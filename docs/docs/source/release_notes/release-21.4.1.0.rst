release-21.4.1.0
================

A new version of Flow360, release-21.4.1.0, has been deployed. Any
new submissions of mesh will use this new version by default. Any
cases based on previously submitted meshes or forked from submitted
cases using prior versions will still use their originally specified
versions of Flow360.
   
Solver
------

*New features*

1. Added support for automatic mesh generation.

2. Added support for the Darcy-Forchheimer porous media model.

3. Added support for Detached Eddy Simulation with kOmega SST turbulence model.

4. Added support for getting power usage in actuator disk method.

5. Added support for laminar-turbulent transition model.

6. Added support for both right-hand and left-hand design of rotor in BET solver.

7. Added support for time averaging capability for volume and NoSlipWall surfaces.

*Resolved issues*

1. Running a steady case won’t increase the physical time, allowing forking between unsteady and steady cases more flexible.

2. Same sectional polars at different radial locations are allowed in BET solver.

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

1. Added json input validation. Majority of the mistakes in input json will be caught immediately upon submission and an error message will be displayed.

2. Added support for checking the consistency of boundary names between the mesh file and input mesh/case json files. Currently this feature is only available for CGNS mesh. If a boundary name, specified in input json files, doesn’t exist in the mesh file, the submission will be aborted and the invalid boundary names will be displayed.

*Resolved issues*

1. Allow “meshJson” argument in NewMesh() to be either a json object or path of json file: :py:meth:`flow360client.NewMesh`.

Web UI
------

*New features*

1. Added support for downloading log files of mesh/case by clicking “Process Log” in the “Download” dropdown menu.

2. Added “Visualization 3D” tab for volume mesh, showing all boundaries in 3D view.

3. Added “Visualization 3D” tab for case, showing Q-criterion in 3D view.

4. Added json input validation. Majority of the mistakes in input json will be caught immediately upon submission and an error message will be shown in a pop-up box.

5. Added support for showing number of selected items.

6. Added support for showing billing information.

7. Added “storage size” for each mesh/case in Description tab.
