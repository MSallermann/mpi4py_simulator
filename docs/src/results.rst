Interpreting results
====================

Always distinguish virtual time from wall-clock time:

* **Virtual time** is produced by the model and is used for scheduling comparisons.
* **Wall-clock time** is how long Python took to execute the simulation and measures
  only simulator efficiency.

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
   Duration passed to ``compute`` by each rank.

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
