"""Blocking MPI-style rank programs on the single-process event engine."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from time import perf_counter
from typing import Any, cast

from greenlet import getcurrent, greenlet

from ._engine import (
    MAX,
    MIN,
    PROD,
    SUM,
    Reduction,
    SimulationResult,
    _Collective,
    _Compute,
    _EventEngine,
    _Probe,
    _Receive,
    _RequestState,
    _Send,
    _Task,
    _TestRequest,
    _WaitRequest,
)
from ._status import ANY_SOURCE, ANY_TAG, Status

RankProgram = Callable[["Comm"], Any]


class Request:
    """Track a nonblocking send or receive.

    Requests are created by :meth:`Comm.isend` and :meth:`Comm.irecv`; applications
    should not instantiate this class directly. Waiting suspends only the current
    virtual rank. It never blocks the simulator process or waits in wall-clock time.

    A send request produces ``None`` when it completes. A receive request produces
    the received Python object.
    """

    def __init__(self, request: _RequestState) -> None:
        """Wrap request state created by the simulator.

        Args:
            request: Internal state returned when the communicator posts a
                nonblocking operation. Applications receive :class:`Request` objects
                from :meth:`Comm.isend` and :meth:`Comm.irecv` instead of calling
                this constructor.

        """
        self._request = request

    @property
    def complete(self) -> bool:
        """Whether the operation has completed on the simulated timeline.

        Reading this property does not advance simulated time. Prefer :meth:`test`
        when the received value or status metadata is also needed. A nonblocking
        receive whose payload has arrived remains incomplete until ``wait`` or
        ``test`` charges any receiver-side processing and materializes the object.
        """
        return self._request.complete

    def wait(self, status: Status | None = None) -> Any:
        """Suspend the current virtual rank until this request completes.

        Args:
            status: Optional status object updated with the source and tag of a
                completed receive. It is unchanged for send requests.

        Returns:
            The received Python object for a receive request, or ``None`` for a send
            request.

        Raises:
            ValueError: If a rank tries to wait for a request created by another
                virtual rank.

        """
        return _suspend(_WaitRequest(self._request, status))

    def Wait(self, status: Status | None = None) -> Any:  # noqa: N802
        """Wait for completion using mpi4py's capitalized method name.

        Args:
            status: Optional receive-status object. See :meth:`wait`.

        Returns:
            The same value as :meth:`wait`.

        """
        return self.wait(status)

    def test(self, status: Status | None = None) -> tuple[bool, Any]:
        """Check whether this request can complete.

        Args:
            status: Optional status object updated when a receive has completed.

        Returns:
            A ``(complete, value)`` tuple. ``value`` is ``None`` while the request is
            incomplete and for completed sends. For a completed receive it is the
            received Python object.

        Raises:
            ValueError: If the request belongs to another virtual rank.

        Notes:
            An unmatched request returns immediately without advancing simulated
            time. For a matched receive, this call charges configured receiver and
            deserialization CPU time before returning the value.

        """
        return cast("tuple[bool, Any]", _suspend(_TestRequest(self._request, status)))

    def Test(self, status: Status | None = None) -> tuple[bool, Any]:  # noqa: N802
        """Test for completion using mpi4py's capitalized method name.

        Args:
            status: Optional receive-status object. See :meth:`test`.

        Returns:
            The same ``(complete, value)`` tuple as :meth:`test`.

        """
        return self.test(status)

    @staticmethod
    def waitall(requests: Sequence[Request]) -> list[Any]:
        """Wait until every request in a sequence has completed.

        Args:
            requests: Send and receive requests owned by the current virtual rank.

        Returns:
            Request results in input order. Send requests contribute ``None`` and
            receive requests contribute their received objects.

        Notes:
            The implementation waits in sequence, but all operations were already
            posted by ``isend`` or ``irecv`` and can progress concurrently in the
            simulation.

        """
        return [request.wait() for request in requests]

    Waitall = waitall


class Comm:
    """MPI-style communicator supplied to one virtual rank.

    :class:`Simulator` creates one communicator per rank and passes it to the rank
    program. Application code normally receives this object as an argument rather
    than constructing it. The communicator supports mpi4py's Python-object style
    operations; capitalized buffer operations and MPI datatypes are not modeled.

    """

    def __init__(self, simulator: _EventEngine, rank: int) -> None:
        """Create the rank-specific view used by the simulator.

        Args:
            simulator: Internal event engine that owns messages and virtual time.
            rank: Zero-based rank represented by this communicator.

        Notes:
            Applications do not construct communicators directly. :meth:`Simulator.run`
            creates one for each rank and passes it to the rank program.

        """
        self._simulator = simulator
        self._rank = rank
        self._collective_sequence = 0

    @property
    def rank(self) -> int:
        """Return the zero-based rank represented by this communicator."""
        return self._rank

    @property
    def size(self) -> int:
        """Return the total number of virtual ranks, including rank zero."""
        return self._simulator.size

    def Get_rank(self) -> int:  # noqa: N802
        """Return this rank using mpi4py's method naming.

        Returns:
            The same zero-based integer as :attr:`rank`.

        """
        return self.rank

    def Get_size(self) -> int:  # noqa: N802
        """Return the communicator size using mpi4py's method naming.

        Returns:
            The same total rank count as :attr:`size`.

        """
        return self.size

    def send(self, value: Any, dest: int, tag: int = 0) -> None:
        """Send a Python object and suspend until the modeled send completes.

        Args:
            value: Arbitrary Python object to send. It is copied when the send is
                posted if ``copy_messages=True`` was passed to :class:`Simulator`.
                Configured payload byte rates use its pickled representation.
            dest: Zero-based destination rank in ``range(comm.size)``.
            tag: Nonnegative integer used to match the message at the receiver.

        Raises:
            ValueError: If ``dest`` is outside the communicator or ``tag`` is
                negative.

        Notes:
            Sends use buffered semantics and do not wait for a matching receive. The
            sender first pays its fixed overhead and configured serialization cost.
            Transport then adds latency and any payload transfer time.

        """
        _suspend(_Send(value, dest, tag, nonblocking=False))

    def isend(self, value: Any, dest: int, tag: int = 0) -> Request:
        """Post a nonblocking Python-object send.

        Args:
            value: Python object to send. Message-copy behavior is controlled by the
                simulator's ``copy_messages`` option.
            dest: Zero-based destination rank.
            tag: Nonnegative message tag.

        Returns:
            A :class:`Request` after sender overhead and configured serialization
            have been charged. This local posting work cannot overlap other work on
            the same rank. Network transport may proceed while the rank continues.

        Raises:
            ValueError: If the destination or tag is invalid.

        """
        request = _suspend(_Send(value, dest, tag, nonblocking=True))
        return Request(cast("_RequestState", request))

    def recv(
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> Any:
        """Receive the first available object matching a source and tag.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE` to accept any rank.
            tag: Required message tag, or :data:`ANY_TAG` to accept any tag.
            status: Optional object updated with the actual sender and tag.

        Returns:
            The Python object carried by the matched message.

        Raises:
            ValueError: If ``source`` or ``tag`` is invalid.
            DeadlockError: Indirectly from :meth:`Simulator.run` if no future event
                can produce a matching message and the simulated program cannot make
                progress.

        Notes:
            A receive suspends only this virtual rank. After a message matches, the
            rank resumes after its fixed receive overhead and configured
            deserialization cost. Receiver CPU costs are serialized on that rank.

        """
        return _suspend(_Receive(source, tag, status, nonblocking=False))

    def irecv(
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> Request:
        """Post a nonblocking receive for a source and tag filter.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE`.
            tag: Required message tag, or :data:`ANY_TAG`.
            status: Optional object updated when the receive completes, even if the
                request is never passed another status object.

        Returns:
            A :class:`Request` immediately. Use :meth:`Request.wait` or
            :meth:`Request.test` to retrieve the received object.

        Raises:
            ValueError: If ``source`` or ``tag`` is invalid.

        Notes:
            Network transfer may finish while the rank does other work. Configured
            receiver and deserialization CPU time is charged when ``wait`` or
            ``test`` materializes the matched Python object.

        """
        request = _suspend(_Receive(source, tag, status, nonblocking=True))
        return Request(cast("_RequestState", request))

    def probe(
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> bool:
        """Wait until a matching message is available without consuming it.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE`.
            tag: Required message tag, or :data:`ANY_TAG`.
            status: Optional object updated with the matched sender and tag.

        Returns:
            ``True`` once a matching message has arrived. The message remains queued
            for a later :meth:`recv` or :meth:`irecv`.

        Raises:
            ValueError: If ``source`` or ``tag`` is invalid.
            DeadlockError: Indirectly from :meth:`Simulator.run` if no event can make
                a matching message available.

        """
        return cast("bool", _suspend(_Probe(source, tag, status, blocking=True)))

    def iprobe(
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> bool:
        """Check for a matching message without waiting or consuming it.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE`.
            tag: Required message tag, or :data:`ANY_TAG`.
            status: Optional object updated only when a match is found.

        Returns:
            ``True`` if a matching message is available at the current simulated
            time, otherwise ``False``. Simulated time is never advanced.

        Raises:
            ValueError: If ``source`` or ``tag`` is invalid.

        """
        return cast("bool", _suspend(_Probe(source, tag, status, blocking=False)))

    def Probe(  # noqa: N802
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> bool:
        """Wait for a message using mpi4py's capitalized probe name.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE`.
            tag: Required message tag, or :data:`ANY_TAG`.
            status: Optional object populated with the matched sender and tag.

        Returns:
            The same result as :meth:`probe`.

        """
        return self.probe(source, tag, status)

    def Iprobe(  # noqa: N802
        self,
        source: int = ANY_SOURCE,
        tag: int = ANY_TAG,
        status: Status | None = None,
    ) -> bool:
        """Test for a message using mpi4py's capitalized probe name.

        Args:
            source: Required sender rank, or :data:`ANY_SOURCE`.
            tag: Required message tag, or :data:`ANY_TAG`.
            status: Optional object populated when a match is found.

        Returns:
            The same nonblocking result as :meth:`iprobe`.

        """
        return self.iprobe(source, tag, status)

    def compute(self, duration: float) -> None:
        """Occupy this rank with modeled computation without doing real work.

        Args:
            duration: Finite, nonnegative simulated duration in seconds.

        Raises:
            TypeError: If ``duration`` is not a real number.
            ValueError: If ``duration`` is negative, infinite, or NaN.

        Notes:
            Only this rank is suspended. Other virtual ranks continue to process
            communication and computation events. Ordinary Python code consumes
            wall-clock time and is charged only when ``measure_python=True``. Even in
            that mode, expensive kernels should usually be replaced by this call in
            large timing experiments so they are not really executed once per rank.

        """
        _suspend(_Compute(duration))

    sleep = compute

    def barrier(self) -> None:
        """Wait until every virtual rank reaches the same barrier.

        All ranks must enter collectives in the same order. The barrier completes
        ``collective_latency`` seconds after the final rank arrives.

        Raises:
            DeadlockError: Indirectly from :meth:`Simulator.run` when another rank
                never reaches this collective.
            ValueError: If ranks enter different collective operations at the same
                sequence position.

        """
        self._collective("barrier")

    def Barrier(self) -> None:  # noqa: N802
        """Enter a barrier using mpi4py's capitalized method name.

        This has the same synchronization and timing behavior as :meth:`barrier`.
        """
        self.barrier()

    def reset_measurement(self) -> None:
        """Synchronize all ranks and discard statistics collected so far.

        After every rank reaches this operation, reported simulation time starts at
        zero and the per-rank compute, measured-Python, communication, idle-time, and
        last-compute fields are reset. Point-to-point message counters also restart.
        The synchronization and its ``collective_latency`` are excluded from the new
        interval.

        Communicator state is preserved: queued messages and nonblocking requests
        remain valid, and rank programs continue normally. Message counters include
        only messages posted after the most recent reset. Complete setup exchanges
        before calling this method when their later delivery or completion should not
        delay the measured phase.

        All ranks must call this method in the same collective sequence. It may be
        called repeatedly; :class:`SimulationResult` reports only the interval after
        the final call.

        Raises:
            DeadlockError: Indirectly from :meth:`Simulator.run` if another rank
                never reaches the boundary.
            ValueError: If ranks enter different collective operations at this
                sequence position.

        """
        self._collective("reset_measurement")

    def ResetMeasurement(self) -> None:  # noqa: N802
        """Reset statistics using a capitalized MPI-style method name."""
        self.reset_measurement()

    def bcast(self, value: Any = None, root: int = 0) -> Any:
        """Return the root rank's value on every rank.

        Args:
            value: Object contributed by ``root``. Values supplied by other ranks are
                ignored and conventionally set to ``None``.
            root: Zero-based rank that provides the broadcast value.

        Returns:
            A copy of the root value on every rank.

        Raises:
            ValueError: If ``root`` is invalid or ranks disagree about the
                collective kind or root.

        """
        return self._collective("bcast", value=value, root=root)

    def scatter(self, values: Sequence[Any] | None = None, root: int = 0) -> Any:
        """Distribute one root-provided value to each rank.

        Args:
            values: On ``root``, a sequence containing exactly ``comm.size`` items.
                Other ranks normally pass ``None``.
            root: Zero-based rank that owns ``values``.

        Returns:
            The item at index ``comm.rank`` in the root sequence.

        Raises:
            ValueError: If ``root`` is invalid, if the root value is not a sequence
                of the required length, or if ranks disagree about the collective.

        """
        return self._collective("scatter", value=values, root=root)

    def gather(self, value: Any, root: int = 0) -> list[Any] | None:
        """Collect one value from every rank on a root rank.

        Args:
            value: This rank's contribution.
            root: Zero-based rank that receives the collected values.

        Returns:
            On ``root``, a list ordered by source rank. On every other rank,
            ``None``.

        Raises:
            ValueError: If ``root`` is invalid or ranks disagree about the
                collective.

        """
        return cast("list[Any] | None", self._collective("gather", value, root))

    def allgather(self, value: Any) -> list[Any]:
        """Collect one value from every rank and return the list everywhere.

        Args:
            value: This rank's contribution.

        Returns:
            A rank-ordered list of all contributions on every rank.

        Notes:
            The simulator materializes a separate copied result for every rank.
            Memory use can therefore grow quadratically with communicator size.

        """
        return cast("list[Any]", self._collective("allgather", value))

    def alltoall(self, values: Sequence[Any]) -> list[Any]:
        """Exchange one value from every source with every destination.

        Args:
            values: A sequence of exactly ``comm.size`` objects. Item ``i`` is sent
                to rank ``i``.

        Returns:
            A list ordered by source rank containing the object each source selected
            for this rank.

        Raises:
            ValueError: If any rank supplies a value of the wrong length or enters a
                different collective.

        Notes:
            All input and output lists are materialized in memory. Large simulated
            worlds can therefore require quadratic memory.

        """
        return cast("list[Any]", self._collective("alltoall", values))

    def reduce(
        self,
        value: Any,
        op: Reduction | Callable[[Any, Any], Any] = SUM,
        root: int = 0,
    ) -> Any:
        """Reduce rank values and return the result on one root rank.

        Args:
            value: This rank's contribution.
            op: ``SUM``, ``PROD``, ``MIN``, ``MAX``, the corresponding predefined
                mpi4py operation, or a two-argument callable. Values are combined in
                ascending rank order.
            root: Zero-based rank that receives the reduced value.

        Returns:
            The reduced value on ``root`` and ``None`` on other ranks.

        Raises:
            TypeError: If ``op`` is not a supported operation or callable.
            ValueError: If ``root`` is invalid or ranks disagree about the
                collective configuration.

        """
        return self._collective("reduce", value, root, _coerce_reduction(op))

    def allreduce(
        self,
        value: Any,
        op: Reduction | Callable[[Any, Any], Any] = SUM,
    ) -> Any:
        """Reduce rank values and return the result on every rank.

        Args:
            value: This rank's contribution.
            op: ``SUM``, ``PROD``, ``MIN``, ``MAX``, the corresponding predefined
                mpi4py operation, or a two-argument callable. Values are combined in
                ascending rank order.

        Returns:
            The reduced value on every rank.

        Raises:
            TypeError: If ``op`` is not supported or callable.
            ValueError: If ranks disagree about the collective or reduction.

        """
        return self._collective(
            "allreduce",
            value,
            reduction=_coerce_reduction(op),
        )

    @staticmethod
    def waitall(requests: Sequence[Request]) -> list[Any]:
        """Wait for a sequence of requests owned by the current rank.

        Args:
            requests: Previously posted send and receive requests.

        Returns:
            Results in request order. Receive requests contribute received objects;
            send requests contribute ``None``.

        See Also:
            :meth:`Request.waitall`: Equivalent request-level helper.

        """
        return [request.wait() for request in requests]

    def _collective(
        self,
        kind: str,
        value: Any = None,
        root: int | None = None,
        reduction: Reduction | Callable[[Any, Any], Any] | None = None,
    ) -> Any:
        sequence = self._collective_sequence
        self._collective_sequence += 1
        return _suspend(_Collective(sequence, kind, value, root, reduction))


class _RankTask(_Task):
    def __init__(
        self,
        program: RankProgram,
        comm: Comm,
        *,
        measure_python: bool,
    ) -> None:
        self._greenlet = greenlet(lambda: program(comm))
        self._started = False
        self._measure_python = measure_python
        self._python_seconds = 0.0

    @property
    def python_seconds(self) -> float:
        return self._python_seconds

    def send(self, value: Any, /) -> object:
        started = perf_counter() if self._measure_python else 0.0
        try:
            if self._started:
                result = self._greenlet.switch(value)
            else:
                self._started = True
                result = self._greenlet.switch()
        finally:
            self._python_seconds = (
                perf_counter() - started if self._measure_python else 0.0
            )
        if self._greenlet.dead:
            raise StopIteration(result)
        return result

    def close(self) -> None:
        if not self._greenlet.dead:
            with suppress(GeneratorExit):
                self._greenlet.throw(GeneratorExit)


class Simulator:
    """Run ordinary blocking MPI-style rank code in one Python process.

    The simulator invokes the same ``program(comm)`` function once for every virtual
    rank. A lightweight greenlet preserves each rank's blocking call stack while a
    deterministic event engine advances communication and computation events. No
    subprocesses, OS threads, MPI runtime, or asyncio event loop are created.

    Args:
        size: Total number of virtual ranks, including rank zero. For a coordinator
            and 10,000 workers, pass ``size=10_001``.
        latency: One-way point-to-point message latency in simulated seconds. It is
            charged after the sender's send overhead.
        send_overhead: Simulated seconds for a rank to post or complete a buffered
            send. This sender-side CPU cost is serialized with computation and other
            communication work on that rank.
        receive_overhead: Simulated seconds between matching a message and resuming
            its receiver. This receiver-side CPU cost is serialized per rank.
        collective_latency: Simulated seconds between the last rank entering a
            collective and all participants resuming. When omitted, ``latency`` is
            used.
        bandwidth: Optional point-to-point payload bandwidth in bytes per simulated
            second. A message of ``n`` pickled bytes adds ``n / bandwidth`` to its
            transport delay. ``None`` disables payload transfer cost.
        serialization_bandwidth: Optional point-to-point sender serialization rate
            in bytes per simulated second. Serialization time is charged to the
            sending rank before both blocking and nonblocking sends return.
        deserialization_bandwidth: Optional point-to-point receiver deserialization
            rate in bytes per simulated second. Deserialization time is charged to
            and serialized on the receiving rank.
        measure_python: Whether to measure each rank's ordinary Python execution
            between communicator calls and add it to that rank's simulated clock.
            This captures coordinator bookkeeping without source instrumentation but
            makes results dependent on the simulator host and current system load.
        copy_messages: If ``True``, copy communicated objects so later mutation by a
            sender cannot change a received value. Point-to-point messages use a
            pickle round trip when payload byte rates are configured; otherwise the
            simulator uses ``deepcopy``. Collective values use ``deepcopy``.
        trace: Whether to retain rank-local compute, measured-Python, communication,
            blocking-wait, and collective intervals for serialization and timeline
            visualization. Disabled by default to avoid event-level memory growth.

    Raises:
        TypeError: If a timing or bandwidth argument has the wrong type, or a boolean
            option is not a boolean.
        ValueError: If ``size`` is not positive, a duration is negative, or a
            configured bandwidth is not finite and positive.

    Notes:
        Timing parameters use seconds. The model has fixed latency and overhead; it
        can model payload size and calibrated byte rates, but does not model network
        congestion, topology, finite buffering, or eager/rendezvous transitions.
        Ordinary Python work runs sequentially in the real interpreter. It adds no
        simulated time unless ``measure_python=True``. Use :meth:`Comm.compute` for
        calibrated or synthetic computation on rank zero as well as worker ranks.

    """

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
        measure_python: bool = False,
        copy_messages: bool = True,
        trace: bool = False,
    ) -> None:
        """Configure rank count, message costs, and object-copy behavior.

        Args:
            size: Total virtual rank count, including rank zero.
            latency: Fixed one-way message latency in seconds.
            send_overhead: Fixed buffered-send overhead in seconds.
            receive_overhead: Fixed receive-completion overhead in seconds.
            collective_latency: Fixed collective completion cost in seconds, or
                ``None`` to reuse ``latency``.
            bandwidth: Optional point-to-point payload bytes transferred per second.
            serialization_bandwidth: Optional point-to-point payload bytes serialized
                per second on the sender.
            deserialization_bandwidth: Optional point-to-point payload bytes
                deserialized per second on the receiver.
            measure_python: Whether to charge measured rank-program execution between
                communicator calls to the corresponding virtual rank.
            copy_messages: Whether to copy communicated Python objects.
            trace: Whether to retain detailed simulated event intervals.

        """
        if not isinstance(measure_python, bool):
            msg = "measure_python must be a boolean"
            raise TypeError(msg)
        if not isinstance(trace, bool):
            msg = "trace must be a boolean"
            raise TypeError(msg)
        self._engine = _EventEngine(
            size,
            latency=latency,
            send_overhead=send_overhead,
            receive_overhead=receive_overhead,
            collective_latency=collective_latency,
            bandwidth=bandwidth,
            serialization_bandwidth=serialization_bandwidth,
            deserialization_bandwidth=deserialization_bandwidth,
            copy_messages=copy_messages,
            trace=trace,
        )
        self._measure_python = measure_python

    @property
    def size(self) -> int:
        """Return the configured total number of virtual ranks."""
        return self._engine.size

    def run(self, program: RankProgram) -> SimulationResult:
        """Execute one synchronous rank function across the virtual world.

        Args:
            program: Callable receiving one :class:`Comm`. It is invoked once per
                rank and may use ordinary blocking MPI-style operations. Its return
                value is recorded for that rank.

        Returns:
            A :class:`SimulationResult` containing rank return values and a snapshot
            of timing and message statistics.

        Raises:
            TypeError: If ``program`` is not callable or yields an unsupported
                internal operation.
            DeadlockError: If unfinished ranks are blocked and no scheduled event can
                resume any of them. The exception message lists each blocked rank and
                operation.
            BaseException: Any exception raised by the rank program is propagated
                after the other suspended rank stacks are closed.

        Notes:
            A ``Simulator`` may be run again; each call starts with a fresh event
            queue and zeroed statistics. Module globals and mutable closures are
            shared between virtual ranks because all ranks live in one interpreter.

        """
        if not callable(program):
            msg = "program must be callable"
            raise TypeError(msg)

        return self._engine.run(
            lambda rank: _RankTask(
                program,
                Comm(self._engine, rank),
                measure_python=self._measure_python,
            ),
        )


def _suspend(operation: object) -> Any:
    current = getcurrent()
    if current.parent is None:
        msg = "blocking communicator methods must be called inside a simulation"
        raise RuntimeError(msg)
    return current.parent.switch(operation)


def _coerce_reduction(
    reduction: Reduction | Callable[[Any, Any], Any] | object,
) -> Reduction | Callable[[Any, Any], Any]:
    if isinstance(reduction, Reduction):
        return reduction
    if type(reduction).__module__.startswith("mpi4py"):
        mpi = importlib.import_module("mpi4py.MPI")
        for name, simulated in (
            ("SUM", SUM),
            ("PROD", PROD),
            ("MIN", MIN),
            ("MAX", MAX),
        ):
            if reduction == getattr(mpi, name):
                return simulated
    if callable(reduction):
        return reduction
    msg = "op must be SUM, PROD, MIN, MAX, or a two-argument callable"
    raise TypeError(msg)


class _MPINamespace:
    """Group MPI-like constants and helper classes for dependency injection.

    Attributes:
        ANY_SOURCE: Wildcard accepted by receive and probe source filters.
        ANY_TAG: Wildcard accepted by receive and probe tag filters.
        MAX: Predefined maximum reduction.
        MIN: Predefined minimum reduction.
        PROD: Predefined product reduction.
        SUM: Predefined sum reduction.
        Request: Simulated nonblocking request class, including ``Waitall``.
        Status: Simulated message-status class.

    Notes:
        This deliberately small namespace is not a replacement for the complete
        ``mpi4py.MPI`` module. It is useful when a dependency already accepts a
        simulated communicator but obtains status or request helpers from a module-
        level MPI object.

    """

    ANY_SOURCE = ANY_SOURCE
    ANY_TAG = ANY_TAG
    MAX = MAX
    MIN = MIN
    PROD = PROD
    SUM = SUM
    Request = Request
    Status = Status


MPI = _MPINamespace()
