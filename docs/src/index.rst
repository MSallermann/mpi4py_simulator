mpi4py-simulator
================

``mpi4py-simulator`` runs the control flow of an existing blocking mpi4py program at
large virtual rank counts in one Python process. It is for testing message protocols,
scheduling decisions, and simplified timing assumptions without starting ``mpiexec``
or allocating a cluster.

The normal workflow is:

#. put the existing rank body in a function that accepts ``comm``;
#. pass the real ``MPI.COMM_WORLD`` in production;
#. pass the communicator supplied by ``Simulator`` in a virtual test; and
#. replace expensive rank work with ``comm.compute(seconds)`` when testing timing at
   scale.

Start with :doc:`overview`, then follow :doc:`blocking` for a working conversion.
The simulator is deliberately not a complete MPI implementation or an automatic
cluster-performance predictor.

.. toctree::
   :maxdepth: 2
   :caption: User guide

   overview
   blocking
   results
   calibration
   chemfit
   limitations
   api
