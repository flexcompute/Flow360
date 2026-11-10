.. _python_api_meshing_farfield_specs:

**********************************
Wind Tunnel Floor Specifications
**********************************

.. currentmodule:: flow360

The farfield classes themselves (:class:`AutomatedFarfield`, :class:`UserDefinedFarfield` and :class:`WindTunnelFarfield`) are listed under :ref:`Zones <python_api_meshing_zones>`. This page lists the floor types of a :class:`WindTunnelFarfield`, set through :py:attr:`WindTunnelFarfield.floor_type`. The default is :class:`StaticFloor`.

.. autosummary::
   :toctree: ../_autosummary
   :template: class.rst

   StaticFloor
   FullyMovingFloor
   CentralBelt
   WheelBelts
