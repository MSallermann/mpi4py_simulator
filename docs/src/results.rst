Interpreting results
====================

Always distinguish virtual time from wall-clock time:

* **Virtual time** is produced by the model and is used for scheduling comparisons.
* **Wall-clock time** is how long Python took to execute the simulation. It is an
  operational cost, not the simulated application runtime. When automatic Python
  timing is enabled, selected slices of this execution feed the virtual model.

Rank results
------------

``Simulator.run`` returns ``SimulationResult``.

``rank_results``
   Tuple indexed by rank containing each rank function's return value.

``time`` / ``stats.time``
   Simulated runtime, defined as the time when the final rank finishes. This is also
   called the *makespan* in scheduling literature.

``stats.rank_finish_seconds``
   Virtual time at which each rank function returned.

``stats.rank_compute_seconds``
   Explicit modeled computation charged through ``compute`` by each rank.

``stats.rank_python_seconds``
   Ordinary rank-function execution measured automatically between communicator
   calls. It is zero when ``measure_python=False``.

``stats.rank_communication_seconds``
   Sender and receiver CPU time charged to each rank. It includes fixed point-to-
   point overhead plus configured serialization and deserialization time, but not
   network transit or collective delay.

``stats.rank_compute_idle_seconds``
   Gaps between a rank's modeled compute calls, excluding its initial wait.

``stats.rank_last_compute_finish_seconds``
   End of each rank's final modeled computation, before later communication or
   shutdown handling.

``stats.messages_sent`` / ``messages_received``
   Logical point-to-point messages posted and receives completed.

``stats.pending_messages``
   Sent point-to-point messages not received when all programs returned.

Collectives do not increment point-to-point counters. Their costs appear in rank
finish times and the simulated runtime.

Find the last finishing rank with:

.. code-block:: python

   finish_times = result.stats.rank_finish_seconds
   last_rank = max(range(len(finish_times)), key=finish_times.__getitem__)

The last rank is not automatically the root cause of a bottleneck. It may be waiting
on an earlier slow rank. Change one timing parameter at a time or inspect intermediate
protocol phases before attributing causality.

Rank zero is included in every per-rank tuple and in the overall runtime. Its
communicator costs and explicit ``compute`` calls are therefore modeled. Set
``measure_python=True`` to also charge its ordinary Python coordinator logic
automatically; otherwise that code has zero simulated duration unless represented by
``comm.compute(seconds)``.

A nonzero pending message count can be intentional for fire-and-forget traffic, but
usually deserves investigation for a missing receive or premature return.

For a worker subset, useful derived metrics are:

.. code-block:: python

   worker_ranks = range(1, size)
   runtime = result.time
   busy = sum(result.stats.rank_compute_seconds[rank] for rank in worker_ranks)
   utilization = busy / ((size - 1) * runtime)
   tail_idle = sum(
       runtime - result.stats.rank_last_compute_finish_seconds[rank]
       for rank in worker_ranks
   )
   throughput = completed_tasks / runtime

Utilization is modeled worker occupancy, not real CPU utilization. An unused worker
has a last-compute finish of zero and contributes the full runtime to tail idle.

Comparing simulations
---------------------

For a fair policy comparison:

* use identical task durations and random seeds;
* use the same latency and overhead assumptions;
* compare virtual metrics, never simulator wall time;
* inspect both simulated runtime and idle-time decomposition; and
* sweep uncertain inputs rather than reporting one overly precise value.

Small changes below the uncertainty of the timing inputs are not meaningful.
