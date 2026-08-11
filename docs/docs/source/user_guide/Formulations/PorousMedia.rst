.. _formulation_porous_media:

############
Porous Media
############

Flow360 models flow through porous regions with the **Darcy-Forchheimer** model.
Rather than resolving the detailed geometry of the porous structure, the model
adds a momentum sink to the governing equations over the assigned volume zone,
reproducing the pressure drop the structure would cause.

.. _formulation_porous_media_governing:

Governing equation
===================

Inside a porous zone, Flow360 adds a **momentum sink per unit volume** to the
momentum equations at every point, combining a linear (viscous, Darcy) term and
a quadratic (inertial, Forchheimer) term. The sink is a force per unit volume
that opposes the local flow. Along each principal axis :math:`i` of the zone it
is:

.. math::
    :label: eq_porous_sink

    S_i = \mu\, D_i\, v_i + \frac{1}{2}\, \rho\, F_i\, |\mathbf{v}|\, v_i

where :math:`v_i` is the velocity component along axis :math:`i` and
:math:`|\mathbf{v}|` the velocity magnitude. Equivalently, the model imposes a
pressure gradient along each porous direction,
:math:`-\,\partial p / \partial x_i = S_i`.

For a uniform one-dimensional flow of velocity :math:`U` through a region of
thickness :math:`L` along a single axis, integrating the gradient recovers the
familiar pressure drop, which is the form used to calibrate the coefficients
below:

.. math::
    :label: eq_porous_darcy_forchheimer

    \Delta p = L \left( \mu\, D\, U + \frac{1}{2}\, \rho\, F\, U^2 \right)

where:

.. list-table::
    :widths: 15 65 20
    :header-rows: 1

    * - Symbol
      - Description
      - Units
    * - :math:`S_i`
      - momentum sink per unit volume along axis :math:`i`
      - :math:`\mathrm{pressure/length}`
    * - :math:`v_i,\; |\mathbf{v}|`
      - local velocity component along axis :math:`i` and velocity magnitude
      - :math:`\mathrm{length/time}`
    * - :math:`\Delta p`
      - pressure drop across the porous medium
      - :math:`\mathrm{pressure}`
    * - :math:`L`
      - porous thickness in the flow direction
      - :math:`\mathrm{length}`
    * - :math:`U`
      - through-flow velocity for the one-dimensional case
      - :math:`\mathrm{length/time}`
    * - :math:`\mu`
      - dynamic viscosity
      - :math:`\mathrm{pressure \cdot time}`
    * - :math:`\rho`
      - density
      - :math:`\mathrm{mass/length^3}`
    * - :math:`D,\; D_i`
      - Darcy (viscous loss) coefficient, per principal axis
      - :math:`\mathrm{1/length^2}`
    * - :math:`F,\; F_i`
      - Forchheimer (inertial loss) coefficient, per principal axis
      - :math:`\mathrm{1/length}`

.. note::

    The model inputs are the Darcy coefficient :math:`D` and the Forchheimer
    coefficient :math:`F` (one value per principal axis), plus an optional
    volumetric heat source. The fit coefficients :math:`a` and :math:`b`
    introduced below are intermediate values used only to derive :math:`D` and
    :math:`F` during calibration; they are not entered directly.

The Darcy term dominates at low flow rates (Reynolds numbers), where losses are
viscous; the Forchheimer term grows with the square of velocity and dominates at
higher flow rates, where losses are inertial.

Anisotropy and principal axes
==============================

:math:`D` and :math:`F` are specified as three values, one per principal axis of
the porous zone. The momentum sink is evaluated in the local reference frame
defined by the zone axes, so anisotropic media (for example a radiator that
resists cross-flow more than through-flow) are represented by assigning
different coefficients to each direction. The axes of the assigned volume must
therefore be defined to align with the physical principal directions of the
material.

Calibrating the coefficients
============================

The coefficients are most often derived from a measured pressure drop versus
velocity curve for the porous component.

1. **Measure** the pressure drop across the porous medium as a function of the
   velocity through it, giving a curve :math:`\Delta p = f(U)`.

2. **Fit** the data with a quadratic relation:

   .. math::
       :label: eq_porous_fit

       \Delta p = a\, U^2 + b\, U

   where :math:`a` is the inertial loss coefficient
   :math:`[\mathrm{pressure \cdot time^2 / length^2}]` and :math:`b` is the
   viscous loss coefficient :math:`[\mathrm{pressure \cdot time / length}]`.

3. **Convert** the fit coefficients to the Flow360 inputs by matching
   :eq:`eq_porous_fit` against :eq:`eq_porous_darcy_forchheimer` term by term:

   .. math::

       D = \frac{b}{\mu\, L}, \qquad F = \frac{2\, a}{\rho\, L}

To compute the coefficients you therefore need:

- the experimental :math:`\Delta p` versus velocity curve through the porous medium,
- the porous thickness :math:`L`,
- the fluid viscosity :math:`\mu`,
- the fluid density :math:`\rho`.

Consistency check
-----------------

With the converted coefficients, the model should reproduce the experimental
pressure drop:

.. math::

    \Delta p = L \left( \mu\, D\, U + \frac{1}{2}\, \rho\, F\, U^2 \right)

Evaluating this at a few velocities and comparing against the original curve is
a quick way to confirm the conversion.

Volumetric heat source
======================

A porous zone can optionally include a volumetric heat source, which adds a
source term to the energy equation over the zone. This is useful when the porous
region also exchanges heat with the fluid, such as a radiator core. The heat
source can be a constant or, through the Python API, a position- or
time-dependent expression.

.. note::

    Sign convention: a positive volumetric heat source adds heat to the fluid
    (heating it) and a negative value removes heat (cooling it). This is the
    opposite of the wall heat-flux convention, where a positive heat flux denotes
    heat leaving the fluid. The momentum sink itself always opposes the local
    flow, regardless of the sign of the coefficients.

Box zones and edge windowing
============================

When a porous zone is defined as a box, Flow360 automatically smooths the
resistance to zero over a thin layer at the box faces rather than letting it
jump discontinuously, which improves numerical robustness at the zone boundary.
This taper uses a fixed internal profile (it is not a user input), and the
interior coefficients are adjusted so the total resistance of the zone is
preserved.

Zones taken directly from the geometry or volume mesh are not tapered: the
coefficients are applied uniformly across the entire zone.

Setting up a porous medium
==========================

This page covers the formulation and calibration. For how to define a porous
zone and enter the coefficients in practice, see:

.. seealso::

    - :doc:`Porous medium in the simulation setup </gui_guide/02.simulation-setup/03.flow-solver/03.physics/06.porous-medium>`
    - :doc:`Volume models in the Python API reference </python_api/API_reference/volume_models/volume_model_index>`
