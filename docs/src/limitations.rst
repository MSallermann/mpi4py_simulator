Limitations
===========

The simulator deliberately omits much of MPI. Its value comes from making scheduling
and protocol assumptions explicit without recreating a complete MPI runtime.

Execution model
---------------

* There is no parallel CPU execution, CPU contention, or operating-system scheduling.
* All ranks share one Python interpreter, including module globals and mutable closure
  state. Real MPI processes normally do not share these objects.
* Rank functions use one greenlet stack per virtual rank.
* Code that hard-codes ``MPI.COMM_WORLD`` must accept the simulated communicator at an
  application boundary. The simulator does not monkey-patch mpi4py globally.
* Exceptions are regular Python exceptions, not exact MPI error classes.

Communication model
-------------------

* Messages are Python objects, not MPI buffers.
* Datatypes, element counts, ``Get_count``, and memory layouts are not checked.
* Point-to-point byte rates use the size of Python's pickled representation. They do
  not model MPI buffer datatypes or zero-copy transfers.
* Payload-dependent point-to-point time can be configured, but congestion, topology,
  shared links, and rank locality are not modeled.
* Sends are buffered. Eager limits, rendezvous handshakes, finite buffers, and
  blocking-send backpressure are absent.
* Process failure, message corruption, cancellation, and nondeterministic runtime
  progress are not modeled.

Communicators and operations
----------------------------

* The simulator has one world communicator. Duplication, splitting, groups,
  intercommunicators, and Cartesian or graph topologies are absent.
* Only the documented object collectives are supported. Nonblocking collectives and
  neighborhood collectives are absent.
* Wait-any/test-any families, persistent requests, and cancellation are absent.
* One-sided communication, windows, atomics, MPI I/O, dynamic process management,
  sessions, and fault tolerance are absent.
* The API is mpi4py-like where useful but is not a drop-in implementation of every
  capitalized buffer-oriented method.

Timing accuracy
---------------

* Point-to-point costs can depend on payload size, but configured byte rates are
  fixed and independent of topology or concurrent traffic.
* Collectives use one cost after the last arrival rather than simulating an algorithm.
* Python rank-function execution advances simulated time only when
  ``measure_python=True``. Measurements include simulator-call overhead and are host-
  and load-dependent, so they are not deterministic.
* The simulator's actual pickle and copy operations do not advance simulated time.
* Uncalibrated virtual time is suitable for relative experiments, not prediction.

Scaling consequences
--------------------

Memory grows linearly with greenlet stacks and rank state, plus queued messages and
returned values. ``allgather`` and ``alltoall`` inherently materialize quadratic
result data. Wildcard matching can scan queued unmatched messages, so protocols with
large mailbox backlogs cost more wall time.
