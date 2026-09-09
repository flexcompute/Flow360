Error when calculating patch interpolation coefficients
=======================================================

When running a case with a rotational volume zone, if you get a solver error that looks like this:

.. code-block::

   Error when calculating patch interpolation coefficients. Patch has a distance mismatch of 0.7883948e-01
   with matching patch at location [1.7592032e-1, 1.0184957e-03, -4.7304816e-1].

This means that at a given time step, once the mesh has been rotated according to the prescribed motion, the maximum distance between the two interfaces is 0.7883948e-01. If this maximum distance between the two patches is greater than 20% of the maximum edge length contained in either patch, this error will be raised. It indicates that either the center of rotation/axis of rotation is likely incorrect or that the mesh is too coarse in this location.