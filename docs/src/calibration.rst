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
