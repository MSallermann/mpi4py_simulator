Real MPI versus calibrated simulation
=====================================

The repository includes a worked coordinator/worker example that runs the same rank
function through real mpi4py and through ``Simulator``. It calibrates communication
on the target machine, records the real runtime and return values, runs the virtual
comparison, and saves versioned statistics for visualization.

The shared rank function
------------------------

The example dynamically refills workers as they return results. Its communicator,
work implementation, wildcard constant, and status constructor are injected, but the
message protocol itself is shared:

.. literalinclude:: ../../examples/compare_real_and_simulated.py
   :language: python
   :start-after: # rank-main-start
   :end-before: # rank-main-end

The real worker calls ``time.sleep(task.seconds)`` before computing the square. The
virtual worker calls ``comm.compute(task.seconds)`` and computes the same square
without sleeping. Both runs execute the same sends, receives, dynamic refill policy,
and final barrier.

Run the comparison
------------------

Install the package and optional real-MPI dependency on the machine being measured:

.. code-block:: console

   python -m pip install -e '.[calibration]'

First calibrate with the same rank count and process placement as the real run. This
lets the barrier measurement reflect the comparison's communicator size:

.. code-block:: console

   mpiexec -n 5 python examples/compare_real_and_simulated.py calibrate \
       --iterations 200 --output calibration.json

Next run the actual workload. The initial barrier is outside the timed region; the
final barrier ensures the reported runtime includes completion by every worker:

.. code-block:: console

   mpiexec -n 5 python examples/compare_real_and_simulated.py actual \
       --tasks 64 --task-ms 5 --output actual-run.json

Finally run the same protocol with virtual ranks. The command takes rank count,
workload settings, and expected rank-zero values from ``actual-run.json``. It refuses
to compare calibration and actual-run documents with different rank counts:

.. code-block:: console

   python examples/compare_real_and_simulated.py simulate \
       --calibration calibration.json \
       --actual actual-run.json \
       --stats simulation-stats.json

The last command verifies identical rank-zero results and prints output in this form:

.. code-block:: text

   actual MPI runtime: 0.083421 s
   simulated runtime:  0.079816 s
   difference:         -0.003605 s (-4.3%)
   rank-zero results:  identical
   wrote simulated statistics to simulation-stats.json

These numbers are illustrative, not expected output. Results depend on hardware,
placement, MPI, Python, and system load.

Visualize the simulated run
---------------------------

Render the saved statistics as a self-contained interactive HTML report:

.. code-block:: console

   mpi4py-simulator-visualize simulation-stats.json \
       --output simulation-stats.html

The report shows each rank's finish-time composition and provides sortable rank rows
with exact timing details. See :doc:`results` for the JSON schema and an explanation
of the derived ``Waiting / other`` segment.

Interpreting the difference
---------------------------

The comparison is a validation exercise, not proof that the fixed-cost model will
predict every workload. The calibrator fits raw-buffer transport and rank-zero pickle
costs over byte payloads, while the workload exchanges small ``Task`` and result
objects. It also assigns one fixed cost to the final barrier. The simulation enables
``measure_python`` so coordinator bookkeeping contributes to virtual time, which
makes the result dependent on the host and current load.

After this first run:

* inspect measured versus predicted calibration samples;
* repeat both real and simulated runs to quantify noise;
* try larger task payloads and work durations;
* use the same process placement for calibration and the actual run; and
* sweep timing inputs rather than tuning one configuration until a single run agrees.

Agreement on this small workload supports the chosen inputs only near the measured
regime. The limitations in :doc:`calibration` and :doc:`limitations` still apply.
