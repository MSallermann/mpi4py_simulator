ChemFit example
===============

``example.py`` runs ChemFit's public MPI scheduling path without defining a simulator-
specific scheduler. Every virtual rank constructs the normal
``MPIGraphScheduler(comm=comm)``. Rank zero calls ``evaluate_many`` and the remaining
ranks enter ChemFit's normal ``worker_loop``.

The only ChemFit-specific adapter is:

.. code-block:: python

   mpi_graph_schedule.MPI = MPI

ChemFit already accepts the communicator, but its shutdown implementation obtains
``Status``, wildcard constants, and ``Request.Waitall`` from its module-level MPI
namespace. The simulator namespace supplies those helpers. It does not replace any
ChemFit scheduler behavior.

Complete example
----------------

.. literalinclude:: ../../example.py
   :language: python
   :linenos:

Run it with:

.. code-block:: console

   python example.py --workers 10000 --tasks 32768 --task-ms 1

``--workers`` excludes rank zero. The simulator therefore creates ``workers + 1``
virtual ranks. ``--task-ms`` is passed to ``comm.compute`` by the objective and does
not cause real sleeping.

The output reports completed evaluations, simulated runtime, and modeled worker
utilization. Simulated runtime is the duration on the configured virtual clock;
wall-clock runtime only describes the cost of executing the simulator.

The example requires ChemFit and its MPI extras for imports, although it does not
launch MPI processes. Initializing a locally installed MPI library can print
environment-specific network warnings; those do not mean that workers were started.
