Existing blocking MPI code
==========================

``Simulator`` instruments a working mpi4py
application. It calls one ordinary function per virtual rank and supplies a
synchronous communicator with mpi4py-style object operations.

Quick conversion
----------------

Keep the application protocol in one function and inject its external dependencies:

.. code-block:: python

   def rank_main(comm, work):
       rank = comm.Get_rank()
       if rank == 0:
           comm.send("job", dest=1, tag=10)
           return comm.recv(source=1, tag=11)

       job = comm.recv(source=0, tag=10)
       answer = work(job)
       comm.send(answer, dest=0, tag=11)
       return None

Production supplies real dependencies:

.. code-block:: python

   from mpi4py import MPI

   rank_main(MPI.COMM_WORLD, real_work)

The virtual harness supplies modeled ones:

.. code-block:: python

   from mpi4py_simulator import Comm, Simulator


   def virtual_rank(comm: Comm):
       def modeled_work(job):
           comm.compute(0.005)
           return "modeled answer"

       return rank_main(comm, modeled_work)


   result = Simulator(size=10_001).run(virtual_rank)

The body of ``rank_main`` is shared. There is no ``async`` conversion and no MPI
launcher. ``size`` is the total rank count, so rank 0 plus 10,000 workers is 10,001.

Runnable example
----------------

The repository example contains a complete coordinator/worker program, injected work
model, command-line wrapper, and result reporting:

.. literalinclude:: ../../examples/existing_mpi.py
   :language: python
   :linenos:

Run it from the repository root:

.. code-block:: console

   python examples/existing_mpi.py --ranks 10001

Instrumenting computation
-------------------------

Ordinary application Python runs for real, once per rank, and consumes wall time. It
does not advance virtual time by default. Use this behavior intentionally:

Functional test
   Run a small real kernel and assert rank return values. This checks more application
   behavior, but it scales no better than sequential execution.

Timing experiment
   Replace the expensive kernel with a model. Call ``comm.compute(seconds)`` to
   suspend only the current virtual rank, then cheaply produce whatever value the
   protocol needs. Use measured durations or a reproducible duration distribution.

It is usually cleanest to inject ``work`` as shown above. Another option is a small
application-owned abstraction whose real implementation computes and whose simulated
implementation calls ``comm.compute``.

Do not call ``time.sleep`` to model work. It blocks the only real interpreter and
changes wall time without changing virtual time.

Accounting for coordinator logic
--------------------------------

Some work must execute because its result controls the protocol. Rank zero may, for
example, update a priority queue and use the selected worker in its next send. Enable
automatic Python timing once on the simulator:

.. code-block:: python

   result = Simulator(10_001, measure_python=True).run(rank_main)

No rank-function changes are needed. The simulator measures each uninterrupted slice
of rank Python between communicator calls. A rank's event-loop waiting and execution
of other virtual ranks are not included. Rank-zero scheduling therefore contributes
to its finish time while preserving the existing MPI control flow.

The measurement applies to all ranks. Expensive worker kernels still execute for real
and should be replaced with cheap models plus ``comm.compute`` at large rank counts.
Because measurements depend on the host and current load, leave the option disabled
and use calibrated explicit durations for reproducible comparisons.

Supported calls
---------------

``Comm`` supports object ``send``, ``recv``, ``isend``, ``irecv``, ``probe``,
and ``iprobe``; sources, tags, wildcards, and statuses; barriers and the common object
collectives; and reductions using the built-in simulator operations, two-argument
callables, or mpi4py's predefined ``SUM``, ``PROD``, ``MIN``, and ``MAX`` operations.

Requests provide ``wait``/``Wait`` and ``test``/``Test``. Use ``Request.waitall`` or
``Comm.waitall`` for a group of requests. The package's small ``MPI`` namespace also
provides ``MPI.Request.Waitall`` for dependencies that obtain request helpers through
an MPI module.

Only Python-object operations are modeled. Capitalized buffer operations such as
``Send`` and ``Recv`` are deliberately absent because the simulator has no MPI
datatype or buffer model.

Hard-coded MPI state
--------------------

The simulator does not replace the process-wide ``mpi4py.MPI`` module. This avoids
surprising imports and global monkey-patching, but means these patterns need an
application boundary:

``MPI.COMM_WORLD`` inside the rank body
   Pass ``comm`` into the body instead.

``MPI.Request.Waitall``
   Inject ``mpi4py_simulator.MPI`` into the dependency for the virtual run, or call
   ``Request.waitall`` directly.

``MPI.SUM`` passed to ``comm.allreduce``
   This is supported when mpi4py is installed. ``mpi4py_simulator.SUM`` works without
   mpi4py.

``MPI.Status()``
   Real mpi4py statuses and ``mpi4py_simulator.Status`` both support the metadata the
   simulated communicator sets.

Global and closure state
------------------------

Real MPI ranks have separate interpreters. Virtual ranks share one interpreter, so a
module-level list or mutable closure is shared between them. Put process-local state
inside ``rank_main`` or construct it per invocation. Intentional shared fixtures
should be read-only.

Result and failure behavior
---------------------------

``run`` returns each rank function's return value and timing/message statistics. See
:doc:`results` for interpretation. If all unfinished ranks are blocked and no
scheduled event can wake them, the simulator raises ``DeadlockError`` with each
blocked rank's operation. Application exceptions are propagated after suspended rank
stacks are closed.

The blocking interface uses the same timing and communication engine described in
:doc:`calibration` and has the restrictions collected in :doc:`limitations`.

Documentation test
------------------

This small blocking program is executed by the documentation build:

.. doctest::

   >>> from mpi4py_simulator import Simulator
   >>> def return_rank(comm):
   ...     return comm.Get_rank()
   >>> result = Simulator(4).run(return_rank)
   >>> result.rank_results
   (0, 1, 2, 3)
