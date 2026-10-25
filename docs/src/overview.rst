Purpose and mental model
========================

Suppose a working MPI application has a coordinator, worker branches, blocking
receives, and collectives. It behaves correctly in a small allocation, but you want
to exercise it with 10,000 workers in CI or on a laptop. ``mpi4py-simulator`` supplies
virtual communicators to that same rank function and schedules its communication on
a deterministic event queue.

.. code-block:: text

   existing rank function -> simulated communicator -> virtual event queue
          ^                                                |
          +----------- result / resumed call --------------+

Every virtual rank has an independent suspended call stack, but only one is executing
at a time. The stacks are greenlets inside one interpreter, not operating-system
threads or processes. A blocking receive suspends its rank; it does not block the
simulator process.

What this is for
----------------

The simulator helps answer questions about application control flow:

* Does the real source/tag protocol complete or deadlock at a larger rank count?
* Are messages left unmatched when rank functions return?
* How do batching, refill, and load-balancing choices affect a simplified virtual
  makespan?
* Which ranks finish last under supplied computation and communication costs?
* Does a scheduling regression reproduce deterministically in a small CI job?

It does not execute MPI, reproduce a network, or make computation parallel. It cannot
validate MPI datatypes or buffer layouts. Absolute time is meaningful only after the
model inputs have been calibrated and checked against a real run.

The smallest useful adaptation
------------------------------

Production code often starts like this:

.. code-block:: python

   from mpi4py import MPI


   def main():
       comm = MPI.COMM_WORLD
       # rank-specific application code follows


   main()

Move the communicator across the function boundary:

.. code-block:: python

   def rank_main(comm):
       # the same rank-specific application code
       ...


   if __name__ == "__main__":
       from mpi4py import MPI

       rank_main(MPI.COMM_WORLD)

The virtual harness is then:

.. code-block:: python

   from mpi4py_simulator import Simulator

   result = Simulator(size=10_001).run(rank_main)

``10_001`` means rank 0 and 10,000 workers. See :doc:`blocking` for computation
instrumentation, a runnable program, supported methods, and common migration issues.

Real Python work versus virtual work
------------------------------------

The simulator actually calls ``rank_main`` for every rank. Normal Python statements
therefore run sequentially in wall-clock time. They add no virtual duration because
the simulator cannot know what the same work would cost on 10,000 real CPUs.

For a small functional test, running the real Python kernel may be useful. For a large
timing test, inject a cheap work model that returns the value needed by later control
flow and calls ``comm.compute(measured_seconds)``. This skips real waiting while
occupying only that rank on the virtual timeline.

One programming model
---------------------

The package has one execution model: ``Simulator`` runs ordinary synchronous rank
functions against ``Comm``. Communication calls block the virtual rank and let the
private event engine advance other ranks.

Installation
------------

Install the package from the repository root:

.. code-block:: console

   python -m pip install -e .

Python 3.10 or newer is required. ``greenlet`` is installed as the single runtime
dependency. mpi4py is optional: a production application may use it, while a virtual
test does not need an MPI runtime.

Build these docs with:

.. code-block:: console

   python -m pip install -e '.[docs]'
   ./build_docs.sh --build-only

Run ``./build_docs.sh`` without ``--build-only`` for documentation tests followed by
a live-reloading development server.
