API reference
=============

The package exposes one synchronous simulation model.

.. autoclass:: mpi4py_simulator.Simulator
   :members:

.. autoclass:: mpi4py_simulator.Comm
   :members:

.. autoclass:: mpi4py_simulator.Request
   :members:

.. autoclass:: mpi4py_simulator.SimulationResult
   :members:

.. autoclass:: mpi4py_simulator.SimulationStats
   :members:

.. autoclass:: mpi4py_simulator.Status
   :members:

.. autoclass:: mpi4py_simulator.Reduction
   :members:

.. autoexception:: mpi4py_simulator.DeadlockError

``ANY_SOURCE`` and ``ANY_TAG`` select wildcard receive filters. Built-in reductions
are ``SUM``, ``PROD``, ``MIN``, and ``MAX``. ``DeadlockError`` reports an unfinished
set of ranks with no future event that can make progress. ``MPI`` groups the status,
request, wildcard, and reduction helpers for dependencies that expect an MPI-like
module namespace.
