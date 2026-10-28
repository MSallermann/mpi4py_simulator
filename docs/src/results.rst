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

If the rank program calls :meth:`~mpi4py_simulator.Comm.reset_measurement`, all times
and counters are relative to the final reset boundary rather than simulation startup.
See :doc:`blocking` for its collective semantics and setup-message behavior.

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

Saving statistics
-----------------

``SimulationStats`` has a versioned JSON format for saving results or passing them to
analysis tools. It includes all aggregate and per-rank statistics, but deliberately
excludes ``rank_results`` because those application-defined values may not be JSON
serializable.

.. code-block:: python

   from pathlib import Path

   from mpi4py_simulator import SimulationStats


   Path("simulation-stats.json").write_text(
       result.stats.to_json(indent=2),
       encoding="utf-8",
   )
   restored = SimulationStats.from_json(
       Path("simulation-stats.json").read_text(encoding="utf-8"),
   )

The top-level JSON object contains ``format: "mpi4py-simulator.stats"``, an integer
``version``, and a ``stats`` object. Version 1 uses the public statistics attribute
names as keys and represents each per-rank tuple as a JSON array in rank order.
``to_dict`` and ``from_dict`` expose the same structure without JSON encoding.
Loaders reject unknown versions, missing or extra fields, invalid numeric values, and
per-rank arrays with inconsistent lengths rather than silently misinterpreting data.

Visualizing statistics
----------------------

The package includes a dependency-free SVG renderer. Give it a saved statistics
document and an output path:

.. code-block:: console

   mpi4py-simulator-visualize simulation-stats.json --output simulation-stats.svg

The equivalent module command works without installing the console entry point:

.. code-block:: console

   python -m mpi4py_simulator.visualization simulation-stats.json \
       --output simulation-stats.svg --max-ranks 64

Each horizontal rank bar decomposes its finish time into explicit modeled compute,
automatically measured Python, communication CPU, and ``Waiting / other``. The last
category is derived rather than traced: it can include blocking waits, transport,
collectives, and idle time. For simulations larger than ``--max-ranks``, the chart
keeps rank zero and the slowest-finishing ranks.

Use :func:`mpi4py_simulator.visualization.render_stats_svg` to generate the same
standalone SVG in Python.

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
