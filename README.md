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
    collective_latency=10e-6,
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

## What is reused and what must be adapted?

The application keeps its rank branches, loops, message values, sources, tags,
blocking calls, nonblocking requests, and collectives. The supported calls use
mpi4py's object API and naming, so ordinary `comm.send(...)`, `comm.recv(...)`, and
`comm.allreduce(...)` code remains synchronous.

Three boundaries still matter:

1. **Supply the communicator.** Code that hard-codes `MPI.COMM_WORLD` inside the rank
   body must accept `comm` as an argument. The simulator does not monkey-patch
   `mpi4py` globally.
2. **Choose what happens to computation.** Ordinary Python executes for real and
   contributes no virtual time. That is useful for small correctness tests. At
   10,000 ranks, inject a cheap result model and call `comm.compute(seconds)` instead
   of running the expensive kernel 10,000 times.
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
- `compute(seconds)` (also named `sleep`) for instrumented virtual work.

Lowercase and relevant mpi4py-capitalized aliases are accepted. Messages are deep
copied by default to approximate mpi4py's pickle-based object communication.

`Request.waitall(requests)` is also available as ``MPI.Request.Waitall(requests)`` on
the small namespace exported by this package. Applications that keep an MPI module in
a dependency can inject that namespace for virtual runs; the
[ChemFit example](example.py) shows the one-line adaptation.

## What the simulator actually does

Calling `Simulator.run(rank_main)` creates one suspended greenlet per rank.
Only one rank executes at a time. When it reaches a simulated operation, control
returns to a deterministic event loop. The event loop advances to the next virtual
event and resumes the affected rank.

- `send` consumes `send_overhead`; its message arrives after `latency`.
- A matched `recv` consumes `receive_overhead`.
- `compute(duration)` suspends that rank for the given virtual duration.
- A collective completes `collective_latency` after its last rank arrives.
- A blocked set of ranks with no event that can wake them raises `DeadlockError`.

Buffered-send semantics are assumed: a send need not wait for its matching receive.
The event ordering is deterministic for deterministic application code and inputs.

## Interpreting the result

`run()` returns `SimulationResult`. The most useful fields are:

| Field | Meaning |
| --- | --- |
| `rank_results[rank]` | Return value of that rank's function. |
| `time` / `stats.time` | Virtual makespan: the latest rank finish. |
| `stats.rank_finish_seconds` | Virtual finish time of each rank. |
| `stats.rank_compute_seconds` | Total duration passed to `compute()` by each rank. |
| `stats.rank_compute_idle_seconds` | Gaps between a rank's modeled compute calls. |
| `stats.rank_last_compute_finish_seconds` | End of each rank's final modeled computation. |
| `stats.messages_sent` | Point-to-point messages posted. |
| `stats.messages_received` | Point-to-point receives completed. |
| `stats.pending_messages` | Messages left unmatched when all rank functions returned. |

The Python command's wall-clock duration measures simulator overhead. It says nothing
about the modeled MPI program. `result.time` is the modeled duration.

For comparisons, keep the workload and calibration fixed and vary one scheduling
choice at a time. Useful signals include:

- a lower makespan means the modeled critical path became shorter;
- a wide spread in rank finish times suggests imbalance or a long queue-draining
  tail;
- unexpected pending messages usually mean a missing receive or a fire-and-forget
  protocol that should be documented;
- a `DeadlockError` means every unfinished virtual rank was waiting with no future
  event able to make progress; and
- correct `rank_results` show functional behavior only within the modeled MPI subset.

Collectives are reflected in makespan and rank finish times, not in the point-to-point
message counters. Compare multiple calibrated scenarios; do not read more precision
from an absolute time than the timing model contains.

## Installation

Python 3.10 or newer is required. `mpi4py` itself is optional.

```bash
python -m pip install -e .
```

`greenlet` is the only runtime dependency. It provides suspended blocking call stacks
inside the one simulator process; it does not create threads or child processes.

## Limitations

- No real parallel execution, CPU contention, OS scheduling, or shared-memory races.
- No payload-dependent bandwidth, congestion, network topology, or rank locality.
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
- Real computation is neither timed nor parallelized automatically. Instrument it
  explicitly with `compute()` when virtual timing matters.

Deep copy resembles object serialization but is not identical to pickle. Collectives
use one fixed latency rather than a tree or ring algorithm. See the
[limitations guide](docs/src/limitations.rst) and
[timing/calibration guide](docs/src/calibration.rst) before treating the output as a
performance prediction.

## Documentation

The full documentation includes the existing-code walkthrough, API-specific guides,
result interpretation, calibration, the ChemFit scheduling example, and limitations.

```bash
python -m pip install -e '.[docs]'
./build_docs.sh --build-only
```

Run `./build_docs.sh` without an argument to execute documentation tests and start a
live-reloading documentation server, following ChemFit's documentation workflow.

## Development

```bash
python -m pip install -e '.[test,docs]'
pytest
ruff check .
ruff format --check .
pyright src
./build_docs.sh --build-only
```
