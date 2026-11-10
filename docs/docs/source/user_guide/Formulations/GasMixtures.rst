.. _formulation_gas_mixtures:

############
Gas Mixtures
############

By default Flow360 simulates air as a calorically perfect gas. When the flow is
another gas, a mixture of gases, or a gas whose specific heats change with
temperature, you describe it instead by the species it is made of. Flow360 then
builds the thermodynamic and transport properties of the gas from those species.
Typical cases are a helium or carbon dioxide jet injected into an air stream, a
pure gas other than air, or a hot flow where the variation of specific heat with
temperature matters.

Species do not react with each other: the composition of a fluid element
changes only by transport and mixing, not chemistry. Gas mixtures are
available through the Python API only.

.. _formulation_gas_mixtures_choice:

Choosing the material
=====================

There are three ways to describe a gas, and the choice depends on whether the
composition is the same everywhere in the domain.

- **Air preset.** Standard air as a calorically perfect gas, with a constant
  specific heat ratio and gas constant, and a Sutherland viscosity law by
  default. Use it whenever the flow is air and the specific heats can be taken
  as constant.
- **Gas with a fixed composition.** One or more species mixed in fixed
  proportions. The species are blended into a single effective gas before the
  simulation runs. Each species brings its own temperature-dependent specific
  heats, so this is also the way to model a single gas with variable specific
  heats.
- **Gas with a transported composition.** Each species is carried through the
  domain as a mass-fraction field, so the composition can vary spatially.
  Boundaries can inject a custom mixture, and the gas properties follow the
  local composition. The solver adds one transport equation per species beyond
  the first.

The material and its species are defined in the thermal state of the
operating condition. Whether the composition is fixed or transported is chosen
on the fluid model.

Where to set it: :py:attr:`~flow360.ThermalState.material`, with
:class:`~flow360.Air` or :class:`~flow360.Gas`; for a gas,
:py:attr:`~flow360.Gas.species` lists the species and
:py:attr:`~flow360.Fluid.species_transport_solver` takes a
:class:`~flow360.FrozenSpeciesSolver` (fixed composition) or a
:class:`~flow360.SpeciesTransportSolver` (transported composition).

.. _formulation_gas_mixtures_species:

Defining the species
====================

Each species is described by its thermodynamics, its molecular weight, and its
viscosity. The thermodynamics come from NASA 9-coefficient polynomials, which
give the specific heat, enthalpy, and entropy as functions of temperature over
one or more temperature ranges. The molecular weight sets the species' share of
the mixture gas constant, and a Sutherland law gives its viscosity.

Common species are available from a curated library, so these data need not be
typed by hand. The library covers N2, O2, Ar, He, H2, CO, CO2, CH4, and H2O.
Looking a species up returns an independent copy, and any of its fields can be
overridden on that copy. For a species outside the library, define it directly
with its own data.

Each species also carries a mass fraction, which defines the reference mixture
described in :ref:`formulation_gas_mixtures_reference`, and a laminar Schmidt
number, which is used only when the composition is transported.

Within one gas:

- species names must be unique;
- all species must use the same temperature-range boundaries in their
  polynomials (library species already share theirs);
- the mass fractions must sum to one. A small deviation is renormalised, a
  larger one is rejected.

.. note::

    The library data have two known limitations.

    - The polynomials assume that each species stays a stable molecule over its
      range. They do not account for dissociation at high temperature.
    - The Sutherland law for H2O is a poor physical model for water vapour,
      whose polar molecules do not follow Sutherland's form well. For flows
      dominated by water vapour, override its viscosity with an
      application-specific law.

Where to set it: :class:`~flow360.Species` and
:meth:`~flow360.Species.from_library`; polynomial data in
:class:`~flow360.NASA9Coefficients`.

.. _formulation_gas_mixtures_properties:

Mixture properties
==================

Flow360 forms the properties of the mixture from those of its species and their
mass fractions :math:`Y_s`. Three laws are involved: Sutherland's law gives the
viscosity of each species, Wilke's mixing rule combines those viscosities into
the mixture viscosity, and Fick's law gives the diffusion of each species when
the composition is transported.

**Gas constant.** The mixture gas constant follows from the universal gas
constant :math:`R_u` and the species molecular weights :math:`W_s`:

.. math::

    R_{mix} = R_u \sum_s \frac{Y_s}{W_s}

**Specific heats.** The specific heats are mass-weighted from the species
polynomials, so they vary with temperature and, when the composition is
transported, with composition.

**Species viscosity: Sutherland's law.** Each species :math:`s` has its own
viscosity :math:`\mu_s`, given as a function of temperature by Sutherland's law
with a reference viscosity :math:`\mu_{ref,s}`, a reference temperature
:math:`T_{ref,s}`, and an effective temperature :math:`S_s` (the Sutherland
constant):

.. math::

    \mu_s(T) = \mu_{ref,s} \left(\frac{T}{T_{ref,s}}\right)^{3/2}
    \frac{T_{ref,s} + S_s}{T + S_s}

**Mixture viscosity: Wilke's mixing rule.** The species viscosities
:math:`\mu_s` are combined with Wilke's mixing rule, written with the mole
fractions :math:`x_s`:

.. math::

    \mu_{mix} = \sum_s \frac{x_s\, \mu_s}{\sum_k x_k\, \Phi_{sk}},
    \qquad
    \Phi_{sk} = \frac{\left[1 + \sqrt{\mu_s/\mu_k}\,(W_k/W_s)^{1/4}\right]^2}
    {\sqrt{8\,(1 + W_s/W_k)}}

A gas's viscosity comes from its species unless a fixed-composition gas is
given its own law, as described below. How the rule is applied depends on the
choice of material.

- **Transported composition.** The solver applies Wilke's rule at every point
  with the local composition.
- **Fixed composition, one species.** The species' own Sutherland law is used.
- **Fixed composition, several species.** The solver receives a single
  Sutherland law that stands in for Wilke's rule. Its reference viscosity is
  Wilke's value at the freestream temperature, so the freestream viscosity is
  exact. Its Sutherland constant is fitted to Wilke's rule over the species'
  temperature range. The fit is within 0.02% for N2/O2 air and within about 2%
  for any pair of library species. A single Sutherland law cannot follow
  Wilke's rule exactly when the species' constants differ widely, for example
  H2O against N2. In that case, or whenever you have a better mixture law,
  give the fixed-composition mixture its own viscosity law or constant value.
  When no single Sutherland law can represent the mixture at all, the case is
  rejected with a request to set that viscosity explicitly.

**Species diffusion: Fick's law.** When the composition is transported, each
species diffuses through the mixture by Fick's law with a constant Schmidt number. The laminar diffusivity of species
:math:`s` is :math:`D_s = \mu / (\rho\, Sc_s)`, from its laminar Schmidt number
:math:`Sc_s`. In turbulent flow a turbulent diffusivity
:math:`D_t = \mu_t / (\rho\, Sc_t)` is added, from the turbulent Schmidt number
:math:`Sc_t` of the species transport solver. The turbulent Schmidt number is a
modelling constant of the turbulence closure, not a property of any species.

Where to set it: :py:attr:`~flow360.Gas.dynamic_viscosity` (fixed composition
only), :py:attr:`~flow360.Species.schmidt_number`,
:py:attr:`~flow360.SpeciesTransportSolver.turbulent_schmidt_number`.

.. _formulation_gas_mixtures_reference:

Reference composition
=====================

The species mass fractions declared on the gas define its reference mixture.
With a fixed composition, that is the mixture everywhere in the domain. With a
transported composition, it is the freestream mixture: it sets the gas
constant, the speed of sound, and the viscosity of the freestream state. It is
also the composition used wherever no other composition is given, at a
boundary or in the initial condition.

A species can have a reference mass fraction of zero. This is the usual way to
declare a gas that is present only where a boundary injects it.

.. _formulation_gas_mixtures_initial:

Initial composition
===================

A transported composition starts from the uniform reference mixture unless you
give an initial condition. The initial condition is a set of expressions in
:math:`x`, :math:`y`, :math:`z`, one per species, and it must name every
declared species. At each point the values are clipped to be non-negative and
then scaled to sum to one, so an expression that slightly overshoots is
corrected rather than rejected. The run stops if all the values at a point are
zero or negative, since no composition can be formed there.

Where to set it: :py:attr:`~flow360.FluidInitialCondition.species_mass_fractions`,
in :py:attr:`~flow360.Fluid.initial_condition`.

.. _formulation_gas_mixtures_boundaries:

Composition at boundaries
=========================

With a transported composition, an :class:`~flow360.Inflow` or
:class:`~flow360.Freestream` boundary can bring in a mixture other than the
reference one, for example a pure helium jet entering an air stream. A boundary
without its own composition uses the reference mixture.

A boundary composition is checked more strictly than the reference mixture:

- it must give a mass fraction for every declared species, and each value must
  lie between zero and one;
- the values must sum to one. They are never renormalised, so a set that does
  not sum to one is rejected;
- the gas must transport at least two species. A fixed-composition gas cannot
  take a boundary composition, because its composition is the same everywhere.

To change the composition of an inflow while the case runs, for example to
model a device that removes a species from the air it returns, use
:ref:`user defined dynamics <udd_species_composition>`.

Where to set it: :py:attr:`~flow360.Inflow.species_mass_fractions`,
:py:attr:`~flow360.Freestream.species_mass_fractions`.

.. _formulation_gas_mixtures_solver:

Species transport solver
========================

A gas needs its species solver to be chosen on the fluid model; without it
the case is rejected. The frozen species solver keeps the reference composition
fixed and solves no extra equations. The species transport solver transports
the composition. Its numerics, such as its tolerances, linear solver, and
update frequencies, and the turbulent Schmidt number are set on the solver
itself.

Where to set it: :py:attr:`~flow360.Fluid.species_transport_solver`, with
:class:`~flow360.FrozenSpeciesSolver` or
:class:`~flow360.SpeciesTransportSolver`.

.. _formulation_gas_mixtures_restrictions:

Restrictions
============

- A transported composition is not supported by the ``CompressibleIsentropic``
  Navier-Stokes solver. Use the ``Compressible`` solver.
- A fixed-composition gas with temperature-dependent specific heats is not
  supported by the ``CompressibleIsentropic`` solver either. A gas whose
  specific heats are constant (see
  :ref:`formulation_gas_mixtures_constant_gamma`) is supported.
- A gas material applies to gas operating conditions only. The liquid
  operating condition takes water as its material.

.. _formulation_gas_mixtures_constant_gamma:

A custom calorically perfect gas
================================

To model a gas other than air with constant specific heats, define a single
species whose polynomial reproduces a constant specific heat ratio. The
constant-gamma constructor of the polynomial class produces such coefficients
from the ratio alone. The species still needs its molecular weight and a
Sutherland law.

These coefficients cover a single temperature range. They therefore cannot be
mixed with library species, whose polynomials use two ranges, because all
species of a gas must share the same range boundaries.

Where to set it: :meth:`~flow360.NASA9Coefficients.constant_gamma`.

.. _formulation_gas_mixtures_outputs:

Outputs
=======

Three output fields are useful with a gas mixture:

- ``SpeciesTransport_massFractions``: the mass fraction of each species, one
  component per species, named after it;
- ``gasConstant``: the local gas constant, which varies through the domain
  when the composition is transported;
- ``mu``: the local laminar viscosity.

See :doc:`Available Output Fields </gui_guide/02.simulation-setup/04.output/output-fields>`
for their scaling.

.. _formulation_gas_mixtures_example:

Example: helium jet into air
============================

The freestream is air made of N2 and O2, and a jet boundary injects pure
helium. Helium is declared with a reference mass fraction of zero, since it is
absent from the freestream.

.. code-block:: python

    import flow360 as fl
    from flow360 import u

    n2 = fl.Species.from_library("N2")
    n2.mass_fraction = 0.767
    o2 = fl.Species.from_library("O2")
    o2.mass_fraction = 0.233
    he = fl.Species.from_library("He")
    he.mass_fraction = 0.0

    air_helium = fl.Gas(
        name="air-helium",
        species=[n2, o2, he],
    )

    operating_condition = fl.AerospaceCondition(
        velocity_magnitude=50 * u.m / u.s,
        thermal_state=fl.ThermalState(
            temperature=288.15 * u.K,
            density=1.225 * u.kg / u.m**3,
            material=air_helium,
        ),
    )

    fluid = fl.Fluid(species_transport_solver=fl.SpeciesTransportSolver())

    jet = fl.Inflow(
        entities=[geometry["jet_exit"]],
        total_temperature=300 * u.K,
        spec=fl.TotalPressure(value=1.2e5 * u.Pa),
        species_mass_fractions={"N2": 0.0, "O2": 0.0, "He": 1.0},
    )

Choosing the material at a glance
=================================

.. list-table::
    :widths: 50 50
    :header-rows: 1

    * - Situation
      - Material
    * - Air, constant specific heats
      - Air preset
    * - Another gas with constant specific heats
      - Gas with a fixed composition, one species built with the
        constant-gamma constructor
    * - Air or another gas with temperature-dependent specific heats, same
        composition everywhere
      - Gas with a fixed composition
    * - A boundary injects a gas different from the freestream
      - Gas with a transported composition, a boundary composition on that
        boundary, and the species transport solver enabled
    * - The initial field holds regions of different composition
      - Gas with a transported composition and an initial condition

.. seealso::

    - :doc:`Material in the Python API reference </python_api/API_reference/operating_condition/material>`
    - :doc:`Solver configuration in the Python API reference </python_api/API_reference/solver_configuration/solver_index>`
    - :doc:`Material in the WebUI </gui_guide/02.simulation-setup/03.flow-solver/03.physics/01.fluid/07.material>`
