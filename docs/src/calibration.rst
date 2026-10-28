Timing and calibration
======================

All timing inputs are finite nonnegative seconds. The simulator uses fixed costs, so
calibration determines what its output can legitimately mean.

Recommended workflow
--------------------

#. Measure representative work durations on the target system. Retain a distribution
   rather than only a mean when load balance matters.
#. Estimate send and receive overhead from application traces or a controlled
   communication benchmark.
#. Use one-way latency for ``latency``, not round-trip latency.
#. When payload cost matters, measure network, serialization, and deserialization
   throughput in bytes per second for representative Python objects.
#. Measure a matching collective and payload before assigning predictive meaning to
   ``collective_latency``.
#. Validate a small simulated configuration against a real MPI run.
#. Sweep low, typical, and high assumptions before extrapolating to extreme ranks.

Conclusions that hold across a plausible parameter sweep are more trustworthy than
one precise-looking result.

Calibration command
-------------------

Install the optional mpi4py dependency and run the calibration module through a real
MPI launcher on the target system:

.. code-block:: console

   python -m pip install -e '.[calibration]'
   mpiexec -n 2 python -m mpi4py_simulator.calibration

Use the process placement that the real application will use. At least two MPI ranks
are required. Ranks zero and one perform the point-to-point measurements, while all
launched ranks participate in the barrier measurement.

By default, the command measures byte payloads ranging from zero bytes to one MiB and
prints a copyable ``Simulator`` configuration followed by measured and predicted
one-way object-message times. Its options are:

``--sizes 0,64,1024,...``
   Comma-separated nonnegative byte-payload sizes. Supply at least two sizes and use
   a range wide enough to distinguish fixed cost from per-byte cost.

``--iterations N``
   Timed repetitions per payload and operation. The default is 100.

``--warmups N``
   Untimed repetitions before each measurement. The default is 10.

``--json``
   Emit the rank count, fitted settings, and per-payload samples as JSON.

For example:

.. code-block:: console

   mpiexec -n 2 python -m mpi4py_simulator.calibration \
       --sizes 0,1024,65536,1048576 --iterations 200 --json

Calibration fails rather than returning a negative byte rate when noisy measurements
do not produce a positive fitted slope. Increase the iteration count and widen the
payload-size range before retrying.

Representative Python objects
-----------------------------

The command-line interface uses ``bytes`` objects of the requested sizes. To measure
the serialization behavior of real application messages, call :func:`calibrate`
from an MPI program. Rank zero supplies the payloads; other ranks may pass ``None``:

.. code-block:: python

   from mpi4py import MPI

   from mpi4py_simulator import Simulator
   from mpi4py_simulator.calibration import calibrate


   payloads = (
       [
           {"kind": "small", "values": [1, 2, 3]},
           {"kind": "large", "values": list(range(10_000))},
       ]
       if MPI.COMM_WORLD.rank == 0
       else None
   )
   calibration = calibrate(
       MPI.COMM_WORLD,
       payloads,
       iterations=200,
       warmups=20,
   )

   if MPI.COMM_WORLD.rank == 0:
       simulation = Simulator(
           size=10_001,
           **calibration.simulator_kwargs,
       )

Every rank receives the same :class:`CalibrationResult`. The result records the
fitted settings and a :class:`CalibrationSample` for each input payload.
``simulator_kwargs`` excludes ``size`` and ``measure_python`` because those describe
the virtual experiment rather than the measured MPI run. ``format_simulator``
produces the same readable configuration style as the command-line output.

What the calibrator measures
----------------------------

For each representative payload, the calibrator measures:

* serialization and deserialization with ``MPI.pickle`` on rank zero;
* raw-buffer ping-pong between ranks zero and one, divided into a one-way time;
* lowercase object ping-pong between ranks zero and one, used to validate the fitted
  model; and
* average barrier duration across every participating rank.

Linear fits assign the raw-buffer intercept to ``latency``, its byte slope to
``bandwidth``, the pickle intercepts to ``send_overhead`` and
``receive_overhead``, and the pickle slopes to the serialization byte rates. A
ping-pong cannot uniquely separate network latency from all MPI endpoint overhead,
so this assignment is a practical model input rather than a physical decomposition.
Compare each sample's ``object_one_way_seconds`` with
``predicted_one_way_seconds`` before using the settings beyond the measured range.

The barrier result becomes one fixed ``collective_latency``. It does not calibrate
the other collective algorithms or payload sizes. Likewise, two-rank ping-pong does
not measure congestion, shared-link effects, eager/rendezvous transitions, or
placement differences. Validate a small simulated configuration against a real run
and sweep plausible alternatives before extrapolating.

Event timeline
--------------

A send first occupies its sender for fixed overhead plus serialization time. The
message then incurs latency plus payload transfer time. After matching, the receiver
is occupied for fixed overhead plus deserialization time. Computation advances only
the calling rank. A collective finishes a fixed collective latency after its final
participant arrives.

For ``n`` pickled bytes, the three components are:

.. code-block:: text

   sender CPU  = send_overhead + n / serialization_bandwidth
   transport   = latency + n / bandwidth
   receiver CPU = receive_overhead + n / deserialization_bandwidth

Unset bandwidths contribute zero. Sender and receiver CPU costs are serialized with
``compute`` and other communication work on their respective ranks. Network transfer
can overlap and currently has no contention model.

For ``irecv``, transfer may complete while the rank computes. Receiver CPU time is
charged later when ``wait`` or ``test`` materializes the Python object, matching the
fact that Python deserialization is not background network progress.

For example:

.. code-block:: python

   simulator = Simulator(
       size=10_001,
       latency=2e-6,
       send_overhead=1e-6,
       receive_overhead=2e-6,
       bandwidth=1e9,
       serialization_bandwidth=500e6,
       deserialization_bandwidth=700e6,
   )

These rates must be calibrated for the objects, Python version, MPI stack, and target
machine being modeled. Do not treat the example numbers as defaults or measurements.

What fixed costs omit
---------------------

The model does not automatically vary cost with:

* intra-node versus inter-node placement;
* bandwidth saturation or simultaneous traffic;
* collective implementation and message segmentation;
* eager versus rendezvous protocols; or
* operating-system and runtime noise.

Represent these effects with separate scenarios, richer duration inputs, or a
targeted model extension. Do not infer them from existing fields.

Reproducibility
---------------

Event ties are resolved in deterministic insertion order. Simulator behavior is
therefore repeatable when rank programs and inputs are deterministic and automatic
Python timing is disabled.
Seed any random duration generator explicitly and reuse the same generated durations
across policies.

Pickling and object copying consume wall time but do not advance simulated time.
Payload size influences simulated time only through configured byte rates.

With ``measure_python=True``, execution inside each rank function is timed between
communicator calls and charged to that rank. Event-loop work and execution of other
ranks are outside each measurement. This is useful for coordinator bookkeeping but
depends on the simulator host and current system load. Leave it disabled and use
fixed calibrated ``compute`` durations when runs must be reproducible.
