.. _formulations:

############
Formulations
############

.. admonition:: Attention
    :class: danger

    This section is under development and does not yet cover every Flow360
    formulation. Pages will be added over time.

Flow360 is a Navier-Stokes solver that offers a range of models to describe the
physical domain and to represent phenomena such as turbulence, heat transfer,
and flow resistance. This section documents the formulations behind those
models: the equations they add to the solver and, where useful, how to choose
their inputs.

The pages are organised by category. Those available so far are linked below.

Surface models (boundary conditions)
====================================

Prescribe the flow behaviour at the surfaces that bound the domain, such as
walls, freestream, inflow, outflow, periodic, symmetry, and interfaces.

*Pages coming soon.*

Volume models
=============

Act over volumetric regions of the domain, such as the fluid and turbulence
models, solid heat transfer, rotation, actuator and BET disks, and porous media.

* :doc:`Porous Media <PorousMedia>`

Other quantities
================

Supporting quantities that are not models in themselves, such as the turbulence
quantities used to set turbulence boundary conditions.

*Pages coming soon.*

.. toctree::
    :hidden:

    PorousMedia
