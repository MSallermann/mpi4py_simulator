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
#. Measure a matching collective and payload before assigning predictive meaning to
   ``collective_latency``.
#. Validate a small simulated configuration against a real MPI run.
#. Sweep low, typical, and high assumptions before extrapolating to extreme ranks.

Conclusions that hold across a plausible parameter sweep are more trustworthy than
one precise-looking result.

Event timeline
--------------

A blocking send resumes after send overhead. Delivery occurs one latency later. A
posted matching receive completes after receive overhead. Computation advances only
the calling rank. A collective finishes a fixed collective latency after its final
participant arrives.

What fixed costs omit
---------------------

The model does not automatically vary cost with:

* payload size or serialization size;
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
therefore repeatable when rank programs and inputs are deterministic.
Seed any random duration generator explicitly and reuse the same generated durations
across policies.

Message ``deepcopy`` and user Python code consume wall time but no virtual time.
Never mix measured simulator runtime into the virtual timing parameters.
