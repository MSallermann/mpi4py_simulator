"""Private deterministic event engine for synchronous virtual ranks."""

from __future__ import annotations

import heapq
import json
import math
import operator
import pickle
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

from ._status import ANY_SOURCE, ANY_TAG, Status


class _Task(Protocol):
    """A suspended rank program driven by the simulation event loop."""

    @property
    def python_seconds(self) -> float:
        """Wall time spent in the most recent rank-program slice."""
        ...

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

_STATS_FORMAT = "mpi4py-simulator.stats"
_STATS_FORMAT_VERSION = 1
_STATS_DOCUMENT_FIELDS = frozenset({"format", "version", "stats"})
_STATS_VALUE_FIELDS = frozenset(
    {
        "time",
        "messages_sent",
        "messages_received",
        "pending_messages",
        "rank_finish_seconds",
        "rank_compute_seconds",
        "rank_python_seconds",
        "rank_communication_seconds",
        "rank_compute_idle_seconds",
        "rank_last_compute_finish_seconds",
    },
)
_STATS_RANK_FIELDS = (
    "rank_finish_seconds",
    "rank_compute_seconds",
    "rank_python_seconds",
    "rank_communication_seconds",
    "rank_compute_idle_seconds",
    "rank_last_compute_finish_seconds",
)


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
        rank_compute_seconds: Total modeled computation explicitly charged through
            :meth:`Comm.compute` by each rank.
        rank_python_seconds: Python execution time measured automatically for each
            rank. It is zero when ``measure_python=False``.
        rank_communication_seconds: Sender and receiver CPU time charged to each
            rank. This includes fixed overhead and configured serialization or
            deserialization time, but excludes network transit and collective delay.
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
    rank_python_seconds: tuple[float, ...]
    rank_communication_seconds: tuple[float, ...]
    rank_compute_idle_seconds: tuple[float, ...]
    rank_last_compute_finish_seconds: tuple[float, ...]

    def to_dict(self) -> dict[str, object]:
        """Return the versioned, JSON-compatible statistics document.

        Returns:
            A new dictionary containing the format identifier, format version, and
            every statistics field. Per-rank tuples are represented as lists.

        Notes:
            The format deliberately excludes :attr:`SimulationResult.rank_results`,
            whose application-defined values may not be JSON serializable.

        """
        return {
            "format": _STATS_FORMAT,
            "version": _STATS_FORMAT_VERSION,
            "stats": {
                "time": self.time,
                "messages_sent": self.messages_sent,
                "messages_received": self.messages_received,
                "pending_messages": self.pending_messages,
                "rank_finish_seconds": list(self.rank_finish_seconds),
                "rank_compute_seconds": list(self.rank_compute_seconds),
                "rank_python_seconds": list(self.rank_python_seconds),
                "rank_communication_seconds": list(
                    self.rank_communication_seconds,
                ),
                "rank_compute_idle_seconds": list(self.rank_compute_idle_seconds),
                "rank_last_compute_finish_seconds": list(
                    self.rank_last_compute_finish_seconds,
                ),
            },
        }

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize these statistics to versioned JSON.

        Args:
            indent: Optional indentation passed to :func:`json.dumps`. ``None``
                produces compact JSON; a value such as ``2`` produces readable
                output.

        Returns:
            A JSON string accepted by :meth:`from_json`.

        Raises:
            ValueError: If a manually constructed instance contains NaN or infinity,
                which JSON cannot represent portably.

        """
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, document: Mapping[str, object]) -> SimulationStats:
        """Restore statistics from a versioned dictionary.

        Args:
            document: Mapping produced by :meth:`to_dict` or an equivalent decoded
                JSON document.

        Returns:
            An immutable :class:`SimulationStats` instance.

        Raises:
            TypeError: If the document or one of its values has the wrong type.
            ValueError: If the format, version, fields, lengths, or numeric values
                are invalid.

        """
        if not isinstance(document, Mapping):
            msg = "statistics document must be a mapping"
            raise TypeError(msg)

        _validate_stats_fields(
            set(document),
            _STATS_DOCUMENT_FIELDS,
            "statistics document",
        )

        format_name = document.get("format")
        if format_name != _STATS_FORMAT:
            msg = f"unsupported statistics format: {format_name!r}"
            raise ValueError(msg)

        version = document.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            msg = "statistics format version must be an integer"
            raise TypeError(msg)
        if version != _STATS_FORMAT_VERSION:
            msg = f"unsupported statistics format version: {version}"
            raise ValueError(msg)

        values = document.get("stats")
        if not isinstance(values, Mapping):
            msg = "statistics document must contain a 'stats' mapping"
            raise TypeError(msg)
        _validate_stats_fields(set(values), _STATS_VALUE_FIELDS, "statistics")

        rank_values = {
            name: _deserialize_stats_times(name, values[name])
            for name in _STATS_RANK_FIELDS
        }
        rank_counts = {len(items) for items in rank_values.values()}
        if len(rank_counts) != 1:
            msg = "per-rank statistics fields must have equal lengths"
            raise ValueError(msg)
        if rank_counts == {0}:
            msg = "per-rank statistics fields must contain at least one rank"
            raise ValueError(msg)

        return cls(
            time=_deserialize_stats_time("time", values["time"]),
            messages_sent=_deserialize_stats_count(
                "messages_sent",
                values["messages_sent"],
            ),
            messages_received=_deserialize_stats_count(
                "messages_received",
                values["messages_received"],
            ),
            pending_messages=_deserialize_stats_count(
                "pending_messages",
                values["pending_messages"],
            ),
            rank_finish_seconds=rank_values["rank_finish_seconds"],
            rank_compute_seconds=rank_values["rank_compute_seconds"],
            rank_python_seconds=rank_values["rank_python_seconds"],
            rank_communication_seconds=rank_values["rank_communication_seconds"],
            rank_compute_idle_seconds=rank_values["rank_compute_idle_seconds"],
            rank_last_compute_finish_seconds=rank_values[
                "rank_last_compute_finish_seconds"
            ],
        )

    @classmethod
    def from_json(cls, document: str | bytes | bytearray) -> SimulationStats:
        """Restore statistics from a versioned JSON document.

        Args:
            document: UTF-8 JSON text produced by :meth:`to_json`.

        Returns:
            An immutable :class:`SimulationStats` instance.

        Raises:
            json.JSONDecodeError: If ``document`` is not valid JSON.
            TypeError: If the decoded document has invalid value types.
            ValueError: If its format, version, fields, lengths, or values are
                invalid.

        """
        decoded = json.loads(document)
        if not isinstance(decoded, Mapping):
            msg = "statistics JSON must contain an object at the top level"
            raise TypeError(msg)
        return cls.from_dict(cast("Mapping[str, object]", decoded))


def _deserialize_stats_time(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"statistics field {name!r} must be a number"
        raise TypeError(msg)
    result = float(value)
    if result < 0.0 or not math.isfinite(result):
        msg = f"statistics field {name!r} must be finite and nonnegative"
        raise ValueError(msg)
    return result


def _validate_stats_fields(
    fields: set[object],
    expected: frozenset[str],
    subject: str,
) -> None:
    missing = {field for field in expected if field not in fields}
    unexpected = {field for field in fields if field not in expected}
    if not missing and not unexpected:
        return
    details = []
    if missing:
        details.append(f"missing {sorted(missing)!r}")
    if unexpected:
        details.append(f"unexpected {sorted(map(str, unexpected))!r}")
    msg = f"invalid {subject} fields: {', '.join(details)}"
    raise ValueError(msg)


def _deserialize_stats_count(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"statistics field {name!r} must be an integer"
        raise TypeError(msg)
    if value < 0:
        msg = f"statistics field {name!r} must be nonnegative"
        raise ValueError(msg)
    return value


def _deserialize_stats_times(name: str, value: object) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        msg = f"statistics field {name!r} must be an array"
        raise TypeError(msg)
    return tuple(_deserialize_stats_time(name, item) for item in value)


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

    def __init__(
        self,
        owner: int,
        *,
        receive: bool,
        defer_completion: bool = False,
    ) -> None:
        self._owner = owner
        self._receive = receive
        self._defer_completion = defer_completion
        self._complete = False
        self._completion_scheduled = False
        self._value: Any = None
        self._matched_message: _Message | None = None
        self._source = ANY_SOURCE
        self._tag = ANY_TAG
        self._waiters: list[tuple[int, Status | None, bool]] = []
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
    size_bytes: int
    pickled: bool
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
        bandwidth: float | None = None,
        serialization_bandwidth: float | None = None,
        deserialization_bandwidth: float | None = None,
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
        _validate_rate("bandwidth", bandwidth)
        _validate_rate("serialization_bandwidth", serialization_bandwidth)
        _validate_rate("deserialization_bandwidth", deserialization_bandwidth)
        if not isinstance(copy_messages, bool):
            msg = "copy_messages must be a boolean"
            raise TypeError(msg)

        self.size = size
        self.latency = float(latency)
        self.send_overhead = float(send_overhead)
        self.receive_overhead = float(receive_overhead)
        self.collective_latency = float(collective_latency)
        self.bandwidth = bandwidth
        self.serialization_bandwidth = serialization_bandwidth
        self.deserialization_bandwidth = deserialization_bandwidth
        self.copy_messages = copy_messages
        self._measure_payloads = any(
            rate is not None
            for rate in (
                bandwidth,
                serialization_bandwidth,
                deserialization_bandwidth,
            )
        )

        self._time = 0.0
        self._sequence = 0
        self._actions: list[_ScheduledAction] = []
        self._tasks: list[_Task | None] = []
        self._results: list[Any] = []
        self._rank_times: list[float] = []
        self._rank_compute: list[float] = []
        self._rank_python: list[float] = []
        self._rank_communication: list[float] = []
        self._rank_cpu_available: list[float] = []
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
            rank_python_seconds=tuple(self._rank_python),
            rank_communication_seconds=tuple(self._rank_communication),
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
        self._rank_python = [0.0] * self.size
        self._rank_communication = [0.0] * self.size
        self._rank_cpu_available = [0.0] * self.size
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
            result = stop.value
            self._after_python(
                rank,
                task.python_seconds,
                lambda: self._finish_rank(rank, result),
            )
            return
        self._after_python(
            rank,
            task.python_seconds,
            lambda: self._handle(rank, operation),
        )

    def _after_python(
        self,
        rank: int,
        duration: float,
        callback: Callable[[], None],
    ) -> None:
        if duration <= 0.0:
            callback()
            return
        start = max(self._time, self._rank_cpu_available[rank])
        finish = start + duration
        self._rank_python[rank] += duration
        self._rank_cpu_available[rank] = finish
        self._waiting[rank] = "executing Python"
        self._schedule(finish, callback)

    def _finish_rank(self, rank: int, result: Any) -> None:
        self._rank_times[rank] = self._time
        self._results[rank] = result
        self._tasks[rank] = None
        self._completed += 1

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
            start = max(self._time, self._rank_cpu_available[rank])
            if self._rank_has_computed[rank]:
                self._rank_compute_idle[rank] += max(
                    0.0,
                    start - self._rank_last_compute_finish[rank],
                )
            finish = start + operation.duration
            self._rank_cpu_available[rank] = finish
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
        value, size_bytes, pickled = self._copy_and_measure(operation.value)
        message = _Message(
            source=rank,
            destination=operation.destination,
            tag=operation.tag,
            value=value,
            size_bytes=size_bytes,
            pickled=pickled,
            sequence=self._sequence,
        )
        sender_cost = self.send_overhead + _byte_cost(
            size_bytes,
            self.serialization_bandwidth,
        )
        send_done = max(self._time, self._rank_cpu_available[rank]) + sender_cost
        self._rank_cpu_available[rank] = send_done
        self._rank_communication[rank] += sender_cost
        arrival = send_done + self.latency + _byte_cost(size_bytes, self.bandwidth)
        self._messages_sent += 1
        self._schedule(arrival, lambda: self._deliver(message))

        if operation.nonblocking:
            request = _RequestState(rank, receive=False)
            self._schedule(send_done, lambda: self._complete_request(request, None))
            self._schedule(send_done, lambda: self._resume(rank, request))
        else:
            self._waiting[rank] = f"sending to rank {operation.destination}"
            self._schedule(send_done, lambda: self._resume(rank))

    def _handle_receive(self, rank: int, operation: _Receive) -> None:
        self._validate_source(operation.source)
        self._validate_tag_filter(operation.tag)
        request = _RequestState(
            rank,
            receive=True,
            defer_completion=operation.nonblocking,
        )
        if operation.status is not None:
            request._completion_statuses.append(operation.status)
        self._post_receive(request, operation.source, operation.tag)
        if operation.nonblocking:
            self._schedule(self._time, lambda: self._resume(rank, request))
        else:
            request._waiters.append((rank, None, False))
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
        self._match_receive(request, message)

    def _deliver(self, message: _Message) -> None:
        requests = self._posted_receives[message.destination]
        for index, request in enumerate(requests):
            if self._request_matches(request, message):
                requests.pop(index)
                self._match_receive(request, message)
                return
        self._mailboxes[message.destination].append(message)
        self._wake_matching_probes(message.destination)

    def _match_receive(
        self,
        request: _RequestState,
        message: _Message,
    ) -> None:
        request._matched_message = message
        if not request._defer_completion:
            self._schedule_receive_completion(request)

    def _schedule_receive_completion(
        self,
        request: _RequestState,
    ) -> None:
        """Reserve receiver CPU time and schedule delivery of one message."""
        if request._completion_scheduled:
            return
        message = request._matched_message
        if message is None:
            return
        request._completion_scheduled = True
        rank = request._owner
        receiver_cost = self.receive_overhead + _byte_cost(
            message.size_bytes,
            self.deserialization_bandwidth,
        )
        completion = max(self._time, self._rank_cpu_available[rank]) + receiver_cost
        self._rank_cpu_available[rank] = completion
        self._rank_communication[rank] += receiver_cost
        self._schedule(
            completion,
            lambda: self._complete_receive(request, message),
        )

    def _complete_receive(
        self,
        request: _RequestState,
        message: _Message,
    ) -> None:
        request._source = message.source
        request._tag = message.tag
        request._matched_message = None
        self._messages_received += 1
        value = (
            pickle.loads(message.value)  # noqa: S301
            if message.pickled
            else message.value
        )
        self._complete_request(request, value)

    def _complete_request(self, request: _RequestState, value: Any) -> None:
        request._complete = True
        request._value = value
        waiters = request._waiters
        request._waiters = []
        for status in request._completion_statuses:
            self._update_status(request, status)
        request._completion_statuses = []
        for rank, status, is_test in waiters:
            self._update_status(request, status)
            result = (True, value) if is_test else value
            self._schedule(
                self._time,
                lambda rank=rank, result=result: self._resume(rank, result),
            )

    def _handle_wait(self, rank: int, operation: _WaitRequest) -> None:
        request = operation.request
        self._validate_request_owner(rank, request)
        if request._complete:
            self._update_status(request, operation.status)
            self._schedule(self._time, lambda: self._resume(rank, request._value))
            return
        request._waiters.append((rank, operation.status, False))
        self._waiting[rank] = "waiting for a request"
        self._schedule_receive_completion(request)

    def _handle_test(self, rank: int, operation: _TestRequest) -> None:
        request = operation.request
        self._validate_request_owner(rank, request)
        if request._complete:
            self._update_status(request, operation.status)
            result = True, request._value
            self._schedule(self._time, lambda: self._resume(rank, result))
        elif request._matched_message is not None:
            request._waiters.append((rank, operation.status, True))
            self._waiting[rank] = "deserializing a request"
            self._schedule_receive_completion(request)
        else:
            self._schedule(self._time, lambda: self._resume(rank, (False, None)))

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

    def _copy_and_measure(self, value: Any) -> tuple[Any, int, bool]:
        if not self._measure_payloads:
            return self._copy(value), 0, False
        payload = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        if self.copy_messages:
            return payload, len(payload), True
        return value, len(payload), False

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


def _validate_rate(name: str, value: float | None) -> None:
    if value is None:
        return
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        msg = f"{name} must be a number or None"
        raise TypeError(msg)
    if value <= 0 or not math.isfinite(value):
        msg = f"{name} must be finite and positive"
        raise ValueError(msg)


def _byte_cost(size_bytes: int, rate: float | None) -> float:
    return 0.0 if rate is None else size_bytes / rate


def _validate_tag(tag: int) -> None:
    if isinstance(tag, bool) or not isinstance(tag, int) or tag < 0:
        msg = "tag must be a nonnegative integer"
        raise ValueError(msg)
