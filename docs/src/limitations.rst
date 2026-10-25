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
* ``deepcopy`` approximates object serialization but is not identical to pickle.
* Payload-dependent time, bandwidth, congestion, topology, and rank locality are not
  modeled.
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

* Timing costs are fixed constants rather than payload- or topology-dependent models.
* Collectives use one cost after the last arrival rather than simulating an algorithm.
* Python rank-function execution and object copying do not advance virtual time.
* Uncalibrated virtual time is suitable for relative experiments, not prediction.

Scaling consequences
--------------------

Memory grows linearly with greenlet stacks and rank state, plus queued messages and
returned values. ``allgather`` and ``alltoall`` inherently materialize quadratic
result data. Wildcard matching can scan queued unmatched messages, so protocols with
large mailbox backlogs cost more wall time.
