.. _python_api_extract_disk_forces:

.. currentmodule:: flow360

************************************
Extract BET and Actuator Disk Forces
************************************

Two snippets demonstrate how to post-process disk-model results from a completed case and report the loads as **dimensional** quantities:

- the integrated loads of a BET disk (converted to SI) plus its radial loading distribution,
- the integrated loads of an actuator disk, converted to SI units.

BET Disk Loads
--------------

.. literalinclude:: _snippets/extract_disk_forces_bet.py
   :language: python

.. figure:: Figures/bet_radial_distribution.png
   :width: 80%
   :align: center

   Example output: the BET disk thrust and torque coefficient distributions along the blade radius.

Actuator Disk Loads
--------------------

.. literalinclude:: _snippets/extract_disk_forces_actuator.py
   :language: python

Notes
=====

- Use ``Case.from_cloud(case_id="...")`` to retrieve a completed case from the cloud.
- ``bet_forces`` holds the integrated BET disk loads. Call ``to_base("SI")`` to convert them to dimensional forces and moments (e.g. ``Disk0_Force_x`` in Newtons) before reading them back with ``as_dataframe()``.
- ``bet_forces_radial_distribution`` resolves the loading per disk and per blade along the radius (columns such as ``Disk0_All_Radius`` and ``Disk0_Blade0_All_ThrustCoeff``). These are non-dimensional coefficients; the client does not provide a dimensional radial output.
- ``actuator_disks`` holds the actuator-disk power, force and moment. Call ``to_base("SI")`` to convert the non-dimensional loads to the SI unit system (``Disk0_Power`` in Watts, ``Disk0_Force`` in Newtons, ``Disk0_Moment`` in Newton-metres) before reading them back with ``as_dataframe()``.
