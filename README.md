# mpi4py-simulator

Test the control flow of an existing `mpi4py` program at large virtual rank counts,
on one machine and without launching MPI processes.

The usual use case is: *“My MPI program works at 8 ranks. Will its scheduling and
message protocol still behave sensibly with 10,000 workers?”* The simulator runs the
same blocking rank function once for every virtual rank, intercepts its communicator
calls, and advances a deterministic virtual clock.

```text
your rank function ── send / recv / collective ──> simulated communicator
        ▲                                           │
        └──────── result or virtual wake-up ────────┘
```

It uses lightweight greenlets to preserve each rank's blocking call stack. Everything
runs in one Python process: no `mpiexec`, subprocesses, OS threads, MPI installation,
or real waiting for modeled computation.

> [!IMPORTANT]
> This tests Python-object communication, protocol correctness, and a timing model.
> It is not an MPI implementation, cluster emulator, or network benchmark. A result
> predicts real performance only to the extent that its timing inputs model the real
> machine.

## From a working MPI program to a virtual run

The simulator needs your rank body as a function that accepts a communicator. Many
MPI programs already have this shape. If yours reads `MPI.COMM_WORLD` directly, this
is normally a one-line refactor.

```python
# application.py
def rank_main(comm, work):
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        for worker in range(1, size):
            comm.send(worker, dest=worker, tag=10)
        return [comm.recv(source=-1, tag=11) for _ in range(size - 1)]

    task = comm.recv(source=0, tag=10)
    answer = work(task)
    comm.send((rank, answer), dest=0, tag=11)
    return None
```

The real MPI entry point remains conventional:

```python
from mpi4py import MPI

rank_main(MPI.COMM_WORLD, real_work)
```

For a virtual run, let `Simulator` supply the communicator instead:

```python
from application import rank_main
from mpi4py_simulator import Comm, Simulator


def virtual_rank(comm: Comm):
    def modeled_work(task: int) -> int:
        comm.compute(0.001)  # 1 ms of virtual work; returns immediately in real time
        return task * task   # value needed by the rest of the protocol

    return rank_main(comm, modeled_work)


result = Simulator(
    size=10_001,             # rank 0 plus 10,000 workers
    latency=2e-6,
    send_overhead=1e-6,
    receive_overhead=3e-6,
    bandwidth=1e9,                    # bytes transferred per second
    serialization_bandwidth=500e6,   # sender-side pickle bytes per second
    deserialization_bandwidth=700e6, # receiver-side pickle bytes per second
    collective_latency=10e-6,
    measure_python=True,              # include coordinator/bookkeeping execution
    trace=True,                       # retain intervals for the HTML timeline
).run(virtual_rank)

print(result.time)
print(result.stats.rank_finish_seconds[0])
print(result.stats.pending_messages)
```

That is the core workflow. [`examples/existing_mpi.py`](examples/existing_mpi.py) is
a complete runnable version:

```bash
python examples/existing_mpi.py --ranks 10001
```

`size` means total ranks, just as `COMM_WORLD.Get_size()` does. Therefore 10,000
workers plus a rank-0 coordinator means `size=10_001`.

For an end-to-end validation workflow, the
[`compare_real_and_simulated.py`](examples/compare_real_and_simulated.py) example
calibrates a real MPI communicator, times the actual coordinator/worker program,
runs the same rank function in the simulator, verifies its results, saves statistics,
and reports the timing difference. The
[worked comparison guide](docs/src/comparison.rst) gives the exact commands.

[`examples/example_ring.py`](examples/example_ring.py) demonstrates a ring exchange
using nonblocking sends and predecessor receives. Every rank's token completes one or
more full laps, and the example verifies the recorded path:

```bash
python examples/example_ring.py --ranks 8 --laps 2 \
  --hop-us 5 --stats ring-stats.json
mpi4py-simulator-visualize ring-stats.json -o ring-stats.html
```

## What is reused and what must be adapted?

The application keeps its rank branches, loops, message values, sources, tags,
blocking calls, nonblocking requests, and collectives. The supported calls use
mpi4py's object API and naming, so ordinary `comm.send(...)`, `comm.recv(...)`, and
`comm.allreduce(...)` code remains synchronous.

Three boundaries still matter:

1. **Supply the communicator.** Code that hard-codes `MPI.COMM_WORLD` inside the rank
   body must accept `comm` as an argument. The simulator does not monkey-patch
   `mpi4py` globally.
2. **Choose what happens to computation.** With `measure_python=True`, ordinary rank
   code—including coordinator logic on rank 0—is measured automatically between MPI
   calls. Expensive worker kernels should still be replaced by cheap result models
   plus `comm.compute(seconds)` so a 10,000-rank simulation does not execute the real
   kernel 10,000 times.
3. **Stay within the supported MPI subset.** The simulator handles Python objects,
   not MPI buffers, datatypes, communicators, files, windows, or processes.

This split is deliberate: the control flow under test remains real application code,
while workload cost and communication cost are explicit model inputs.

## Supported blocking API

`Comm` currently provides:

- `Get_rank`, `Get_size`, and the `rank` and `size` properties;
- `send`, `recv`, `isend`, `irecv`, `probe`, and `iprobe`;
- message tags, `ANY_SOURCE`, `ANY_TAG`, and `Status`;
- request `wait`/`test` and mpi4py-style `Wait`/`Test`;
- `barrier`, `bcast`, `scatter`, `gather`, `allgather`, and `alltoall`;
- `reduce` and `allreduce` with `SUM`, `PROD`, `MIN`, `MAX`, a callable, or the
  corresponding `mpi4py.MPI` predefined operation; and
- `compute(seconds)` (also named `sleep`) for instrumented virtual work; and
- `reset_measurement()` for a collective post-setup measurement boundary.

Lowercase and relevant mpi4py-capitalized aliases are accepted. Messages are copied
by default so mutation after a send cannot change the received value. When a payload
rate is configured, point-to-point objects are pickled to determine their byte size.

`Request.waitall(requests)` is also available as ``MPI.Request.Waitall(requests)`` on
the small namespace exported by this package. Applications that keep an MPI module in
a dependency can inject that namespace for virtual runs.

## What the simulator actually does

Calling `Simulator.run(rank_main)` creates one suspended greenlet per rank.
Only one rank executes at a time. When it reaches a simulated operation, control
returns to a deterministic event loop. The event loop advances to the next virtual
event and resumes the affected rank.

- `send` consumes sender CPU time; its message then travels independently.
- A matched `recv` consumes receiver CPU time before the value becomes available.
- `compute(duration)` suspends that rank for the given virtual duration.
- A collective completes `collective_latency` after its last rank arrives.
- A blocked set of ranks with no event that can wake them raises `DeadlockError`.

Buffered-send semantics are assumed: a send need not wait for its matching receive.
The event ordering is deterministic for deterministic application code and inputs
when automatic Python timing is disabled.

### Point-to-point cost model

For a message whose pickled representation has `n` bytes, the model uses:

```text
sender CPU = send_overhead + n / serialization_bandwidth
transport  = latency       + n / bandwidth
receiver CPU = receive_overhead + n / deserialization_bandwidth
```

An omitted bandwidth removes that term. Sender and receiver CPU time cannot overlap
other modeled CPU work on the same rank; transport can overlap computation and other
messages. This means dispatch and result processing on rank 0 are included when they
occur through communicator calls. With `measure_python=True`, ordinary scheduling
code between those calls is measured and charged automatically as well.

For `irecv()`, transport can finish in the background, but receiver CPU time is paid
when `wait()` or `test()` materializes the Python object. This lets computation overlap
the network without also making Python deserialization happen in the background.

Byte rates are calibration inputs, not measurements taken while the simulation runs.
The real time spent pickling or copying objects affects only how long the simulator
command takes. It is not added to the simulated runtime.

### Calibrating the timing inputs

The optional calibration command measures a starting configuration with a real
mpi4py communicator. Run it under MPI on the target system, with the process
placement you intend to model:

```bash
python -m pip install -e '.[calibration]'
mpiexec -n 2 python -m mpi4py_simulator.calibration
```

The command prints a `Simulator(size=..., ...)` expression and compares measured
object-message times with the fitted model. Use `--sizes` to choose comma-separated
byte-payload sizes, `--iterations` and `--warmups` to control sampling, or `--json`
for machine-readable output. For example:

```bash
mpiexec -n 2 python -m mpi4py_simulator.calibration \
  --sizes 0,1024,65536,1048576 --iterations 200 --json
```

For application-specific Python objects, use the programmatic
`mpi4py_simulator.calibration.calibrate` API. Rank zero supplies at least two
representative payloads with different serialized sizes; other ranks may pass
`None`. The returned `result.simulator_kwargs` can be expanded directly into
`Simulator(size=..., **result.simulator_kwargs)`.

Calibration measures raw-buffer ping-pong, lowercase object ping-pong, pickle costs
on rank zero, and a barrier. It cannot uniquely separate all MPI endpoint and network
costs, and it does not add congestion, topology, or eager/rendezvous behavior to the
model. Check the measured-versus-predicted sample times and validate a small real run
before extrapolating. See the
[timing and calibration guide](docs/src/calibration.rst) for the full workflow.

### Accounting for coordinator work

Set `measure_python=True` once on `Simulator`; no source annotations are required.
Each time a rank runs between communicator calls, only that rank's Python execution
slice is timed and added to its simulated clock. Time in the event loop, greenlet
context switches, and other virtual ranks is outside the measurement.

This naturally accounts for rank-zero queue management, worker selection, task
construction, and result processing. It also measures lightweight worker-side Python
bookkeeping. The measured values are available in
`result.stats.rank_python_seconds`.

Automatic measurement depends on the current machine, Python interpreter, system
load, and simulator-call overhead. Leave it disabled and use calibrated
`comm.compute(seconds)` durations when exact reproducibility matters.

To exclude initialization, every rank can call `comm.reset_measurement()` after
setup. It acts as a collective boundary and restarts reported time, per-rank timing,
and message counters without discarding communicator state. Its own synchronization
cost is excluded; the final result describes only the interval after the last reset.
Complete setup communication first when possible, because outstanding setup work can
still delay code after the boundary.

## Interpreting the result

`run()` returns `SimulationResult`. The most useful fields are:

| Field | Meaning |
| --- | --- |
| `rank_results[rank]` | Return value of that rank's function. |
| `time` / `stats.time` | Simulated runtime: the time when the final rank finishes. |
| `stats.rank_finish_seconds` | Virtual finish time of each rank. |
| `stats.rank_compute_seconds` | Explicit modeled computation charged with `compute()`. |
| `stats.rank_python_seconds` | Automatically measured rank-code execution. |
| `stats.rank_communication_seconds` | Sender/receiver CPU cost charged to each rank. |
| `stats.rank_compute_idle_seconds` | Gaps between a rank's modeled compute calls. |
| `stats.rank_last_compute_finish_seconds` | End of each rank's final modeled computation. |
| `stats.messages_sent` | Point-to-point messages posted. |
| `stats.messages_received` | Point-to-point receives completed. |
| `stats.pending_messages` | Messages left unmatched when all rank functions returned. |
| `stats.trace_events` | Optional chronological intervals retained with `trace=True`. |

Statistics have a versioned JSON representation for storage and analysis:

```python
payload = result.stats.to_json(indent=2)
restored = SimulationStats.from_json(payload)
```

Import `SimulationStats` from `mpi4py_simulator`. The format includes all statistics
but not application-defined `rank_results`; see the
[results guide](docs/src/results.rst) for its schema and validation behavior.

Turn a saved document into a self-contained interactive HTML report:

```bash
mpi4py-simulator-visualize simulation-stats.json -o simulation-stats.html
```

The command opens the report in the system default browser. Pass `--no-open` when
generating it in CI, over SSH, or as part of another script. The file needs no server
or external assets. Rank rows can be filtered and sorted; selecting one shows exact
category totals, compute occupancy, delay to its first compute call, gaps between
compute calls, and time after its last compute call. These intervals can contain
measured Python or communication work; the saved statistics are aggregates, not a
chronological trace. When the simulation used `trace=True`, switch to **Event
timeline** for rank-local compute, Python, communication, wait, and collective
intervals. The timeline's buttons or mouse wheel zoom its shared time axis; drag any
rank track to pan all ranks together. Tracing is disabled by default because memory
and JSON size grow with the number of simulated operations.

The Python command's total wall-clock duration is not the modeled runtime. With
`measure_python=True`, measured per-rank execution slices contribute to `result.time`,
but event-loop overhead, message copying, and other simulator work do not.

For comparisons, keep the workload and calibration fixed and vary one scheduling
choice at a time. Useful signals include:

- a lower simulated runtime means the modeled critical path became shorter;
- a wide spread in rank finish times suggests imbalance or a long queue-draining
  tail;
- unexpected pending messages usually mean a missing receive or a fire-and-forget
  protocol that should be documented;
- a `DeadlockError` means every unfinished virtual rank was waiting with no future
  event able to make progress; and
- correct `rank_results` show functional behavior only within the modeled MPI subset.

Collectives are reflected in simulated runtime and rank finish times, not in the
point-to-point message counters. Compare multiple calibrated scenarios; do not read
more precision from an absolute time than the timing model contains.

## Installation

Python 3.10 or newer is required. `mpi4py` itself is optional for simulation; it is
required only by production MPI code and the calibration command.

```bash
python -m pip install -e .
```

`greenlet` is the only runtime dependency. It provides suspended blocking call stacks
inside the one simulator process; it does not create threads or child processes.

## Limitations

- No real parallel execution, CPU contention, OS scheduling, or shared-memory races.
- Optional payload-dependent transfer and serialization costs, but no congestion,
  network topology, link sharing, or rank locality.
- No eager/rendezvous transition, finite MPI buffering, or send backpressure.
- Python object messages only; no buffer API, counts, or MPI datatypes.
- One world communicator; no communicator splitting, groups, or virtual topologies.
- No one-sided operations, MPI I/O, dynamic processes, or fault tolerance.
- No nonblocking collectives, persistent requests, cancellation, or wait-any family.
- Virtual ranks share module globals because they live in one interpreter. MPI code
  that relies on process-isolated global state must move that state into the rank
  function or another rank-local object.
- The greenlet stack for every virtual rank uses memory, so memory use grows with the
  configured rank count.
- Python execution can be timed automatically, but still runs sequentially in the
  simulator process. Replace expensive kernels with `compute()` for large rank
  counts.

Collectives use one fixed latency rather than a payload-dependent tree or ring
algorithm. See the
[limitations guide](docs/src/limitations.rst) and
[timing/calibration guide](docs/src/calibration.rst) before treating the output as a
performance prediction.

## Documentation

The full documentation includes the existing-code walkthrough, API-specific guides,
result interpretation, calibration, and limitations.

```bash
python -m pip install -e '.[docs]'
./build_docs.sh --build-only
```

The script always runs the documentation tests first. With `--build-only` it then
builds static HTML; without an argument it starts a live-reloading documentation
server.

## Development

```bash
python -m pip install -e '.[test,docs]'
pytest
ruff check .
ruff format --check .
pyright src
./build_docs.sh --build-only
```
