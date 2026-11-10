.. _overview:

Introduction
============

Flow360 is a platform for performing CFD simulations and can be used either through GUI or Python API.
Regardless of whether you choose to use GUI or Python API, your workflow will follow similar steps.
The workflow is centered around a project, which can be seen as a collection of resources and simulations for a given geometry.
Projects can contain geometries, meshes and simulation runs called cases. The entire process can be divided into three main parts:

- Pre-processing
- Simulation
- Post-processing

Pre-processing
--------------

Everything related to preparing input to the simulation. It involves the root resource upload, which can be one of the following:

- Geometry
- Surface mesh
- Volume mesh

Depending on the starting point, the pre-processing can entail the generation of a surface mesh and volume mesh if the root resource is a geometry.
At the same time, when the root resource is a volume mesh, there is no need for additional pre-processing.

Mesh generation is a streamlined process in which upon selecting the desired settings, mesh will be automatically generated.
These settings control both the global mesh resolution as well as local refinements. There are also options for specifying the farfield as well as rotation regions.

Simulation
----------

Simulation or rather its setup is a core step of the workflow. 
It involves specifying Flow conditions as well as Solver settings, which will include things like boundary conditions, time stepping and turbulence model definition.
Every change you make to your simulation setup will be considered a separate case. It means that in a single project you can have multiple cases with different settings, for example, different angles of attack.

Post-processing
---------------

There a few ways to post-process the results of a simulation using Flow360. You can use default outputs or create your own to better visualize the results.
These outputs can be viewed in the GUI or downloaded for further post-processing in external tools. 


.. note::
    If you are unsure where to start, :ref:`Quick Start <quickstart>` is a good place to get familiar with the basics of Flow360.


See Also: Capabilities
-----------------------

Flow360 includes a wide range of solver features and model extensions.  
You can explore them in detail in the :doc:`Capabilities section <capabilities>`.

.. toctree::
    :maxdepth: 2
    :caption: Capabilities
    :hidden:

    capabilities

