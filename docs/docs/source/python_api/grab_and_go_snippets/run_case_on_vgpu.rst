.. _python_api_run_case_on_vgpu:

.. currentmodule:: flow360

****************************
Run a Case on Virtual GPU
****************************

This example demonstrates how to submit a simulation case to the Virtual GPU (vGPU) queue
instead of the standard FlexCredits pay-as-you-go queue.
Use ``billing_method="VirtualGPU"`` to route the job through your daily vGPU allocation,
and ``priority`` to control its scheduling order relative to other queued jobs.

.. literalinclude:: _snippets/run_case_on_vgpu.py
   :language: python

Notes
-----

- ``billing_method`` accepts ``"VirtualGPU"`` or ``"FlexCredit"``. When omitted, the account
  default is used.
- ``priority`` is an integer from ``1`` (lowest) to ``10`` (highest). It controls the order
  in which queued jobs are dispatched when vGPU slots become available. Only applies when
  ``billing_method="VirtualGPU"``; it is ignored otherwise.
- The Virtual GPU option requires an active vGPU license on your account. If no vGPU
  allocation is available, use ``billing_method="FlexCredit"`` instead.
- You can monitor the queue, change job priorities, or switch a queued job to FlexCredits
  from the **Virtual GPU Scheduler** tab in **Account Settings**.

.. seealso::

   :doc:`Fork a Case <fork>`: submit a follow-on run branching from an existing case.
