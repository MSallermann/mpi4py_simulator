"""Private deterministic event engine for synchronous virtual ranks."""

from __future__ import annotations

import heapq
import math
import operator
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ._status import ANY_SOURCE, ANY_TAG, Status


class _Task(Protocol):
    """A suspended rank program driven by the simulation event loop."""

    def send(self, value: Any, /) -> object:
        """Resume the program until its next simulated operation."""

    def close(self) -> None:
        """Release a suspended program after an error."""


class DeadlockError(RuntimeError):
    """Report that unfinished virtual ranks can no longer make progress.

    :meth:`Simulator.run` raises this exception when its event queue is empty while
    at least one rank is still blocked. The message lists blocked ranks and their
    current operations, such as a receive filter or collective sequence. This is a
    deterministic model-level deadlock report, not an MPI runtime error.
    """


@dataclass(frozen=True, slots=True)
class Reduction:
    """Describe a binary operation for ``reduce`` and ``allreduce``.

    Args:
        name: Human-readable operation name used in diagnostics and collective
            consistency checks.
        function: Two-argument callable that combines an accumulated value with the
            next rank's value.

    Notes:
        Values are reduced in ascending rank order. The simulator does not verify
        associativity or commutativity. Most applications should use the predefined
        ``SUM``, ``PROD``, ``MIN``, or ``MAX`` instances instead of constructing this
        class directly.

    """

    name: str
    function: Callable[[Any, Any], Any] = field(repr=False)

    def __call__(self, left: Any, right: Any) -> Any:
        """Combine an accumulated value with the next rank's value.

        Args:
            left: Value accumulated from lower-numbered ranks.
            right: Contribution from the next rank.

        Returns:
            The result returned by the configured binary function.

        """
        return self.function(left, right)


SUM = Reduction("SUM", operator.add)
PROD = Reduction("PROD", operator.mul)
MIN = Reduction("MIN", min)
MAX = Reduction("MAX", max)


@dataclass(frozen=True, slots=True)
class SimulationStats:
    """Immutable timing and message statistics from a completed simulation.

    Args:
        time: Simulated runtime in seconds, defined by the latest rank finish.
        messages_sent: Number of point-to-point messages posted.
        messages_received: Number of point-to-point receives completed.
        pending_messages: Sent point-to-point messages that remained unmatched when
            all rank functions returned.
        rank_finish_seconds: Simulated return time for each rank, indexed by rank.
        rank_compute_seconds: Total duration passed to :meth:`Comm.compute` by each
            rank.
        rank_compute_idle_seconds: Gaps between consecutive ``compute`` calls on each
            rank. Initial waiting before a rank's first compute call is excluded.
        rank_last_compute_finish_seconds: End time of each rank's final ``compute``
            call. It is zero for ranks that never modeled computation.

    Notes:
        Collective operations affect finish times but are not included in the
        point-to-point message counters. These values describe the configured model,
        not wall-clock execution or operating-system CPU utilization.

    """

    time: float
    messages_sent: int
    messages_received: int
    pending_messages: int
    rank_finish_seconds: tuple[float, ...]
    rank_compute_seconds: tuple[float, ...]
    rank_compute_idle_seconds: tuple[float, ...]
    rank_last_compute_finish_seconds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Return values and statistics produced by :meth:`Simulator.run`.

    Args:
        rank_results: Tuple indexed by virtual rank. Each item is the value returned
            by that rank's ``program(comm)`` invocation.
        stats: Immutable timing and message statistics for the run.

    Attributes:
        rank_results: Per-rank application return values in rank order.
        stats: Detailed simulated timing and message counts.

    """

    rank_results: tuple[Any, ...]
    stats: SimulationStats

    @property
    def time(self) -> float:
        """Return the simulated runtime in seconds.

        This convenience property is identical to ``result.stats.time``. It is the
        latest virtual rank finish and is unrelated to the wall-clock time needed to
        execute the simulator.
        """
        return self.stats.time


@dataclass(frozen=True, slots=True)
class _Send:
    value: Any
    destination: int
    tag: int
    nonblocking: bool


@dataclass(frozen=True, slots=True)
class _Receive:
    source: int
    tag: int
    status: Status | None
    nonblocking: bool


@dataclass(frozen=True, slots=True)
class _WaitRequest:
    request: _RequestState
    status: Status | None


@dataclass(frozen=True, slots=True)
class _TestRequest:
    request: _RequestState
    status: Status | None


@dataclass(frozen=True, slots=True)
class _Compute:
    duration: float


@dataclass(frozen=True, slots=True)
class _Probe:
    source: int
    tag: int
    status: Status | None
    blocking: bool


@dataclass(frozen=True, slots=True)
class _Collective:
    sequence: int
    kind: str
    value: Any = None
    root: int | None = None
    reduction: Reduction | Callable[[Any, Any], Any] | None = None


class _RequestState:
    """Event-engine state for one nonblocking operation."""

    def __init__(self, owner: int, *, receive: bool) -> None:
        self._owner = owner
        self._receive = receive
        self._complete = False
        self._value: Any = None
        self._source = ANY_SOURCE
        self._tag = ANY_TAG
        self._waiters: list[tuple[int, Status | None]] = []
        self._completion_statuses: list[Status] = []

    @property
    def complete(self) -> bool:
        """Return whether the operation has completed in virtual time."""
        return self._complete


@dataclass(order=True, slots=True)
class _ScheduledAction:
    time: float
    sequence: int
    callback: Callable[[], None] = field(compare=False, repr=False)


@dataclass(frozen=True, slots=True)
class _Message:
    source: int
    destination: int
    tag: int
    value: Any
    sequence: int


@dataclass(slots=True)
class _CollectiveState:
    operation: _Collective
    participants: dict[int, Any] = field(default_factory=dict)


class _EventEngine:
    """Schedule synchronous virtual ranks on a deterministic clock."""

    def __init__(
        self,
        size: int,
        *,
        latency: float = 0.0,
        send_overhead: float = 0.0,
        receive_overhead: float = 0.0,
        collective_latency: float | None = None,
        copy_messages: bool = True,
    ) -> None:
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            msg = "size must be a positive integer"
            raise ValueError(msg)
        _validate_time("latency", latency)
        _validate_time("send_overhead", send_overhead)
        _validate_time("receive_overhead", receive_overhead)
        if collective_latency is None:
            collective_latency = latency
        _validate_time("collective_latency", collective_latency)
        if not isinstance(copy_messages, bool):
            msg = "copy_messages must be a boolean"
            raise TypeError(msg)

        self.size = size
        self.latency = float(latency)
        self.send_overhead = float(send_overhead)
        self.receive_overhead = float(receive_overhead)
        self.collective_latency = float(collective_latency)
        self.copy_messages = copy_messages

        self._time = 0.0
        self._sequence = 0
        self._actions: list[_ScheduledAction] = []
        self._tasks: list[_Task | None] = []
        self._results: list[Any] = []
        self._rank_times: list[float] = []
        self._rank_compute: list[float] = []
        self._rank_compute_idle: list[float] = []
        self._rank_last_compute_finish: list[float] = []
        self._rank_has_computed: list[bool] = []
        self._waiting: list[str | None] = []
        self._mailboxes: list[list[_Message]] = []
        self._posted_receives: list[list[_RequestState]] = []
        self._pending_probes: list[list[tuple[int, int, Status | None]]] = []
        self._collectives: dict[int, _CollectiveState] = {}
        self._messages_sent = 0
        self._messages_received = 0
        self._completed = 0

    def run(self, create_task: Callable[[int], _Task]) -> SimulationResult:
        """Drive rank tasks that yield the simulator's internal operations."""
        self._reset()
        try:
            for rank in range(self.size):
                self._tasks.append(create_task(rank))
                self._schedule(0.0, lambda rank=rank: self._resume(rank))

            while self._completed < self.size:
                if not self._actions:
                    blocked = ", ".join(
                        f"rank {rank}: {reason or 'not scheduled'}"
                        for rank, reason in enumerate(self._waiting)
                        if self._tasks[rank] is not None
                    )
                    msg = f"virtual MPI deadlock ({blocked})"
                    raise DeadlockError(msg)
                action = heapq.heappop(self._actions)
                self._time = action.time
                action.callback()
        except BaseException:
            self._close_tasks()
            raise

        stats = SimulationStats(
            time=max(self._rank_times, default=0.0),
            messages_sent=self._messages_sent,
            messages_received=self._messages_received,
            pending_messages=self._messages_sent - self._messages_received,
            rank_finish_seconds=tuple(self._rank_times),
            rank_compute_seconds=tuple(self._rank_compute),
            rank_compute_idle_seconds=tuple(self._rank_compute_idle),
            rank_last_compute_finish_seconds=tuple(
                self._rank_last_compute_finish,
            ),
        )
        return SimulationResult(tuple(self._results), stats)

    def _reset(self) -> None:
        self._time = 0.0
        self._sequence = 0
        self._actions = []
        self._tasks = []
        self._results = [None] * self.size
        self._rank_times = [0.0] * self.size
        self._rank_compute = [0.0] * self.size
        self._rank_compute_idle = [0.0] * self.size
        self._rank_last_compute_finish = [0.0] * self.size
        self._rank_has_computed = [False] * self.size
        self._waiting = [None] * self.size
        self._mailboxes = [[] for _ in range(self.size)]
        self._posted_receives = [[] for _ in range(self.size)]
        self._pending_probes = [[] for _ in range(self.size)]
        self._collectives = {}
        self._messages_sent = 0
        self._messages_received = 0
        self._completed = 0

    def _schedule(self, time: float, callback: Callable[[], None]) -> None:
        action = _ScheduledAction(time, self._sequence, callback)
        self._sequence += 1
        heapq.heappush(self._actions, action)

    def _resume(self, rank: int, value: Any = None) -> None:
        task = self._tasks[rank]
        if task is None:
            return
        self._rank_times[rank] = self._time
        self._waiting[rank] = None
        try:
            operation = task.send(value)
        except StopIteration as stop:
            self._results[rank] = stop.value
            self._tasks[rank] = None
            self._completed += 1
            return
        self._handle(rank, operation)

    def _handle(self, rank: int, operation: object) -> None:
        if isinstance(operation, _Send):
            self._handle_send(rank, operation)
        elif isinstance(operation, _Receive):
            self._handle_receive(rank, operation)
        elif isinstance(operation, _WaitRequest):
            self._handle_wait(rank, operation)
        elif isinstance(operation, _TestRequest):
            self._handle_test(rank, operation)
        elif isinstance(operation, _Compute):
            _validate_time("duration", operation.duration)
            if self._rank_has_computed[rank]:
                self._rank_compute_idle[rank] += max(
                    0.0,
                    self._time - self._rank_last_compute_finish[rank],
                )
            finish = self._time + operation.duration
            self._rank_compute[rank] += operation.duration
            self._rank_last_compute_finish[rank] = finish
            self._rank_has_computed[rank] = True
            self._waiting[rank] = "computing"
            self._schedule(finish, lambda: self._resume(rank))
        elif isinstance(operation, _Probe):
            self._handle_probe(rank, operation)
        elif isinstance(operation, _Collective):
            self._handle_collective(rank, operation)
        else:
            msg = f"rank {rank} yielded an unsupported operation: {operation!r}"
            raise TypeError(msg)

    def _handle_send(self, rank: int, operation: _Send) -> None:
        self._validate_rank(operation.destination, "destination")
        _validate_tag(operation.tag)
        message = _Message(
            source=rank,
            destination=operation.destination,
            tag=operation.tag,
            value=self._copy(operation.value),
            sequence=self._sequence,
        )
        send_done = self._time + self.send_overhead
        arrival = send_done + self.latency
        self._messages_sent += 1
        self._schedule(arrival, lambda: self._deliver(message))

        if operation.nonblocking:
            request = _RequestState(rank, receive=False)
            self._schedule(send_done, lambda: self._complete_request(request, None))
            self._schedule(self._time, lambda: self._resume(rank, request))
        else:
            self._waiting[rank] = f"sending to rank {operation.destination}"
            self._schedule(send_done, lambda: self._resume(rank))

    def _handle_receive(self, rank: int, operation: _Receive) -> None:
        self._validate_source(operation.source)
        self._validate_tag_filter(operation.tag)
        request = _RequestState(rank, receive=True)
        if operation.status is not None:
            request._completion_statuses.append(operation.status)
        self._post_receive(request, operation.source, operation.tag)
        if operation.nonblocking:
            self._schedule(self._time, lambda: self._resume(rank, request))
        else:
            request._waiters.append((rank, None))
            self._waiting[rank] = (
                f"receiving source={operation.source}, tag={operation.tag}"
            )

    def _post_receive(
        self,
        request: _RequestState,
        source: int,
        tag: int,
    ) -> None:
        rank = request._owner
        message_index = self._find_message(rank, source, tag)
        request._source = source
        request._tag = tag
        if message_index is None:
            self._posted_receives[rank].append(request)
            return
        message = self._mailboxes[rank].pop(message_index)
        self._schedule(
            self._time + self.receive_overhead,
            lambda: self._complete_receive(request, message),
        )

    def _deliver(self, message: _Message) -> None:
        requests = self._posted_receives[message.destination]
        for index, request in enumerate(requests):
            if self._request_matches(request, message):
                requests.pop(index)
                self._schedule(
                    self._time + self.receive_overhead,
                    lambda request=request: self._complete_receive(request, message),
                )
                return
        self._mailboxes[message.destination].append(message)
        self._wake_matching_probes(message.destination)

    def _complete_receive(
        self,
        request: _RequestState,
        message: _Message,
    ) -> None:
        request._source = message.source
        request._tag = message.tag
        self._messages_received += 1
        self._complete_request(request, message.value)

    def _complete_request(self, request: _RequestState, value: Any) -> None:
        request._complete = True
        request._value = value
        waiters = request._waiters
        request._waiters = []
        for status in request._completion_statuses:
            self._update_status(request, status)
        request._completion_statuses = []
        for rank, status in waiters:
            self._update_status(request, status)
            self._schedule(self._time, lambda rank=rank: self._resume(rank, value))

    def _handle_wait(self, rank: int, operation: _WaitRequest) -> None:
        request = operation.request
        self._validate_request_owner(rank, request)
        if request._complete:
            self._update_status(request, operation.status)
            self._schedule(self._time, lambda: self._resume(rank, request._value))
            return
        request._waiters.append((rank, operation.status))
        self._waiting[rank] = "waiting for a request"

    def _handle_test(self, rank: int, operation: _TestRequest) -> None:
        request = operation.request
        self._validate_request_owner(rank, request)
        if request._complete:
            self._update_status(request, operation.status)
        result = request._complete, request._value
        self._schedule(self._time, lambda: self._resume(rank, result))

    def _handle_probe(self, rank: int, operation: _Probe) -> None:
        self._validate_source(operation.source)
        self._validate_tag_filter(operation.tag)
        index = self._find_message(rank, operation.source, operation.tag)
        if index is not None:
            message = self._mailboxes[rank][index]
            _set_status(operation.status, message.source, message.tag)
            self._schedule(self._time, lambda: self._resume(rank, True))
        elif operation.blocking:
            self._pending_probes[rank].append(
                (operation.source, operation.tag, operation.status),
            )
            self._waiting[rank] = (
                f"probing source={operation.source}, tag={operation.tag}"
            )
        else:
            self._schedule(self._time, lambda: self._resume(rank, False))

    def _wake_matching_probes(self, rank: int) -> None:
        probes = self._pending_probes[rank]
        remaining = []
        for source, tag, status in probes:
            index = self._find_message(rank, source, tag)
            if index is None:
                remaining.append((source, tag, status))
                continue
            message = self._mailboxes[rank][index]
            _set_status(status, message.source, message.tag)
            self._schedule(self._time, lambda rank=rank: self._resume(rank, True))
        self._pending_probes[rank] = remaining

    def _handle_collective(self, rank: int, operation: _Collective) -> None:
        if operation.root is not None:
            self._validate_rank(operation.root, "root")
        state = self._collectives.get(operation.sequence)
        if state is None:
            state = _CollectiveState(operation)
            self._collectives[operation.sequence] = state
        else:
            self._validate_collective(state.operation, operation)
        state.participants[rank] = self._copy(operation.value)
        self._waiting[rank] = f"collective {operation.kind} #{operation.sequence}"
        if len(state.participants) != self.size:
            return

        results = self._collective_results(state)
        completion = self._time + self.collective_latency
        for result_rank, result in enumerate(results):
            self._schedule(
                completion,
                lambda rank=result_rank, value=result: self._resume(rank, value),
            )
        del self._collectives[operation.sequence]

    def _collective_results(self, state: _CollectiveState) -> list[Any]:
        operation = state.operation
        values = [state.participants[rank] for rank in range(self.size)]
        root = operation.root
        if operation.kind == "barrier":
            return [None] * self.size
        if operation.kind == "bcast":
            value = values[cast("int", root)]
            return [self._copy(value) for _ in range(self.size)]
        if operation.kind == "scatter":
            scattered = values[cast("int", root)]
            if not isinstance(scattered, Sequence) or len(scattered) != self.size:
                msg = "scatter root must provide one value per rank"
                raise ValueError(msg)
            return [self._copy(value) for value in scattered]
        if operation.kind == "gather":
            results: list[Any] = [None] * self.size
            results[cast("int", root)] = values
            return results
        if operation.kind == "allgather":
            return [self._copy(values) for _ in range(self.size)]
        if operation.kind == "alltoall":
            if any(not isinstance(items, Sequence) for items in values) or any(
                len(items) != self.size for items in values
            ):
                msg = "alltoall requires one value per destination from every rank"
                raise ValueError(msg)
            return [
                [self._copy(values[source][destination]) for source in range(self.size)]
                for destination in range(self.size)
            ]
        if operation.kind in {"reduce", "allreduce"}:
            reduction = cast("Callable[[Any, Any], Any]", operation.reduction)
            reduced = values[0]
            for value in values[1:]:
                reduced = reduction(reduced, value)
            if operation.kind == "allreduce":
                return [self._copy(reduced) for _ in range(self.size)]
            results = [None] * self.size
            results[cast("int", root)] = reduced
            return results
        msg = f"unsupported collective: {operation.kind}"
        raise ValueError(msg)

    def _validate_collective(self, first: _Collective, current: _Collective) -> None:
        first_reduction = _reduction_identity(first.reduction)
        current_reduction = _reduction_identity(current.reduction)
        if (
            first.kind != current.kind
            or first.root != current.root
            or first_reduction != current_reduction
        ):
            msg = (
                f"collective #{current.sequence} differs between ranks: "
                f"{first.kind} and {current.kind}"
            )
            raise ValueError(msg)

    def _find_message(self, rank: int, source: int, tag: int) -> int | None:
        for index, message in enumerate(self._mailboxes[rank]):
            if source in (ANY_SOURCE, message.source) and tag in (ANY_TAG, message.tag):
                return index
        return None

    @staticmethod
    def _request_matches(request: _RequestState, message: _Message) -> bool:
        source = request._source
        tag = request._tag
        return source in (ANY_SOURCE, message.source) and tag in (ANY_TAG, message.tag)

    @staticmethod
    def _update_status(request: _RequestState, status: Status | None) -> None:
        if request._receive:
            _set_status(status, request._source, request._tag)

    @staticmethod
    def _validate_request_owner(rank: int, request: _RequestState) -> None:
        if request._owner != rank:
            msg = "a request may only be used by the rank that created it"
            raise ValueError(msg)

    def _validate_rank(self, rank: int, name: str) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or not 0 <= rank < self.size
        ):
            msg = f"{name} must be a rank from 0 to {self.size - 1}"
            raise ValueError(msg)

    def _validate_source(self, source: int) -> None:
        if source != ANY_SOURCE:
            self._validate_rank(source, "source")

    def _copy(self, value: Any) -> Any:
        return deepcopy(value) if self.copy_messages else value

    @staticmethod
    def _validate_tag_filter(tag: int) -> None:
        if tag != ANY_TAG:
            _validate_tag(tag)

    def _close_tasks(self) -> None:
        for task in self._tasks:
            if task is not None:
                task.close()


def _set_status(status: Status | None, source: int, tag: int) -> None:
    if status is not None:
        status.Set_source(source)
        status.Set_tag(tag)


def _reduction_identity(
    reduction: Reduction | Callable[[Any, Any], Any] | None,
) -> object:
    if isinstance(reduction, Reduction):
        return reduction.name
    return reduction


def _validate_time(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number"
        raise TypeError(msg)
    if value < 0 or not math.isfinite(value):
        msg = f"{name} must be finite and nonnegative"
        raise ValueError(msg)


def _validate_tag(tag: int) -> None:
    if isinstance(tag, bool) or not isinstance(tag, int) or tag < 0:
        msg = "tag must be a nonnegative integer"
        raise ValueError(msg)
