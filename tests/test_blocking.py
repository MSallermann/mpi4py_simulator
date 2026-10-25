from __future__ import annotations

import pickle

import pytest

import mpi4py_simulator.simulator as simulator_module
from mpi4py_simulator import (
    ANY_SOURCE,
    MAX,
    MPI,
    SUM,
    Comm,
    DeadlockError,
    Simulator,
    Status,
)


def test_existing_style_blocking_rank_program_runs_unchanged() -> None:
    def program(comm: Comm) -> object:
        if comm.Get_rank() == 0:
            for worker in range(1, comm.Get_size()):
                comm.send(worker * 2, dest=worker, tag=10)
            return [comm.recv(source=ANY_SOURCE, tag=11) for _ in range(3)]

        value = comm.recv(source=0, tag=10)
        comm.compute(float(comm.rank))
        comm.send((comm.rank, value), dest=0, tag=11)
        return None

    result = Simulator(4, latency=0.5).run(program)

    assert result.rank_results[0] == [(1, 2), (2, 4), (3, 6)]
    assert result.stats.messages_sent == 6
    assert result.stats.messages_received == 6
    assert result.time == pytest.approx(4.0)


def test_blocking_requests_status_and_collectives() -> None:
    def program(comm: Comm) -> object:
        destination = (comm.rank + 1) % comm.size
        source = (comm.rank - 1) % comm.size
        send = comm.isend(comm.rank, destination, tag=3)
        status = Status()
        receive = comm.irecv(source, tag=3, status=status)
        values = MPI.Request.Waitall([send, receive])
        total = comm.allreduce(comm.rank + 1, op=SUM)
        return values[1], status.source, total

    result = Simulator(3).run(program)

    assert result.rank_results == ((2, 2, 6), (0, 0, 6), (1, 1, 6))


def test_blocking_program_deadlock_is_reported() -> None:
    def program(comm: Comm) -> None:
        comm.recv(source=(comm.rank + 1) % comm.size)

    with pytest.raises(DeadlockError, match=r"rank 0.*rank 1"):
        Simulator(2).run(program)


def test_messages_are_copied_at_send_time() -> None:
    payload: list[str] = []

    def program(comm: Comm) -> object:
        if comm.rank == 0:
            comm.send(payload, dest=1)
            payload.append("changed after send")
            return None
        return comm.recv(source=0)

    result = Simulator(2).run(program)

    assert result.rank_results == (None, [])
    assert payload == ["changed after send"]


def test_probe_does_not_consume_a_message() -> None:
    def program(comm: Comm) -> object:
        if comm.rank == 0:
            comm.send(42, dest=1, tag=9)
            return None
        assert not comm.iprobe(source=0, tag=9)
        status = Status()
        assert comm.probe(source=0, tag=9, status=status)
        return comm.recv(source=0, tag=9), status.source, status.tag

    result = Simulator(2, latency=1.0).run(program)

    assert result.rank_results == (None, (42, 0, 9))


def test_collectives_and_virtual_compute() -> None:
    def program(comm: Comm) -> object:
        comm.compute(float(comm.size - comm.rank))
        comm.barrier()
        shared = comm.bcast("root value" if comm.rank == 2 else None, root=2)
        return comm.gather((comm.rank, shared), root=1)

    result = Simulator(4, collective_latency=0.5).run(program)

    assert result.rank_results == (
        None,
        [
            (0, "root value"),
            (1, "root value"),
            (2, "root value"),
            (3, "root value"),
        ],
        None,
        None,
    )
    assert result.time == pytest.approx(5.5)
    assert result.stats.rank_compute_seconds == (4.0, 3.0, 2.0, 1.0)


def test_scatter_allgather_alltoall_and_reductions() -> None:
    def program(comm: Comm) -> object:
        scattered = comm.scatter([10, 20, 30] if comm.rank == 0 else None)
        gathered = comm.allgather(scattered + comm.rank)
        exchanged = comm.alltoall(
            [f"{comm.rank}->{destination}" for destination in range(comm.size)],
        )
        total = comm.allreduce(comm.rank + 1, op=SUM)
        maximum = comm.reduce(comm.rank, op=MAX, root=2)
        return scattered, gathered, exchanged, total, maximum

    result = Simulator(3).run(program)

    assert result.rank_results[0] == (
        10,
        [10, 21, 32],
        ["0->0", "1->0", "2->0"],
        6,
        None,
    )
    assert result.rank_results[2] == (
        30,
        [10, 21, 32],
        ["0->2", "1->2", "2->2"],
        6,
        2,
    )


def test_mismatched_collectives_are_rejected() -> None:
    def program(comm: Comm) -> None:
        if comm.rank == 0:
            comm.barrier()
        else:
            comm.bcast(None)

    with pytest.raises(ValueError, match="differs between ranks"):
        Simulator(2).run(program)


@pytest.mark.parametrize("size", [0, -1, 1.5, True])
def test_invalid_rank_count_is_rejected(size: object) -> None:
    with pytest.raises(ValueError, match="size"):
        Simulator(size)  # type: ignore[arg-type]


def test_ten_thousand_rank_smoke_test() -> None:
    result = Simulator(10_000).run(lambda comm: comm.rank)

    assert len(result.rank_results) == 10_000
    assert result.rank_results[-1] == 9_999


def test_payload_costs_charge_sender_network_and_receiver() -> None:
    value = {"payload": "x" * 100}
    size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def program(comm: Comm) -> object:
        if comm.rank == 0:
            comm.send(value, dest=1)
            return None
        return comm.recv(source=0)

    result = Simulator(
        2,
        latency=1.0,
        send_overhead=0.5,
        receive_overhead=0.25,
        bandwidth=size / 2,
        serialization_bandwidth=size / 3,
        deserialization_bandwidth=size / 4,
    ).run(program)

    assert result.rank_results == (None, value)
    assert result.stats.rank_finish_seconds == pytest.approx((3.5, 10.75))
    assert result.stats.rank_communication_seconds == pytest.approx((3.5, 4.25))
    assert result.time == pytest.approx(10.75)


def test_nonblocking_send_serialization_is_serial_per_rank() -> None:
    value = "payload"
    size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def program(comm: Comm) -> object:
        if comm.rank == 0:
            requests = [comm.isend(value, dest=1) for _ in range(2)]
            MPI.Request.Waitall(requests)
            return None
        return [comm.recv(source=0) for _ in range(2)]

    result = Simulator(2, serialization_bandwidth=size).run(program)

    assert result.rank_results == (None, [value, value])
    assert result.stats.rank_finish_seconds == pytest.approx((2.0, 2.0))
    assert result.stats.rank_communication_seconds == pytest.approx((2.0, 0.0))


def test_nonblocking_receive_deserialization_waits_for_rank_cpu() -> None:
    value = "payload"
    size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def program(comm: Comm) -> object:
        if comm.rank == 0:
            comm.send(value, dest=1)
            return None
        request = comm.irecv(source=0)
        comm.compute(5.0)
        return request.wait()

    result = Simulator(2, deserialization_bandwidth=size / 2).run(program)

    assert result.rank_results == (None, value)
    assert result.stats.rank_finish_seconds == pytest.approx((0.0, 7.0))
    assert result.stats.rank_compute_seconds == pytest.approx((0.0, 5.0))
    assert result.stats.rank_communication_seconds == pytest.approx((0.0, 2.0))


def test_rank_zero_pays_fan_in_deserialization_costs_serially() -> None:
    value = "result"
    size = len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))

    def program(comm: Comm) -> object:
        if comm.rank == 0:
            return [comm.recv(source=ANY_SOURCE) for _ in range(comm.size - 1)]
        comm.send(value, dest=0)
        return None

    result = Simulator(3, deserialization_bandwidth=size).run(program)

    assert result.rank_results[0] == [value, value]
    assert result.stats.rank_communication_seconds == pytest.approx((2.0, 0.0, 0.0))
    assert result.time == pytest.approx(2.0)


def test_python_work_can_be_measured_automatically(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 10.25, 20.0, 20.0))
    monkeypatch.setattr(simulator_module, "perf_counter", lambda: next(ticks))

    def program(comm: Comm) -> int:
        if comm.rank == 0:
            return 42
        return 0

    result = Simulator(2, measure_python=True).run(program)

    assert result.rank_results == (42, 0)
    assert result.stats.rank_python_seconds == pytest.approx((0.25, 0.0))
    assert result.stats.rank_compute_seconds == pytest.approx((0.0, 0.0))
    assert result.stats.rank_finish_seconds == pytest.approx((0.25, 0.0))


def test_measured_python_slices_accumulate_across_operations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter((10.0, 10.25, 20.0, 20.5))
    monkeypatch.setattr(simulator_module, "perf_counter", lambda: next(ticks))

    def program(comm: Comm) -> str:
        comm.compute(1.0)
        return "done"

    result = Simulator(1, measure_python=True).run(program)

    assert result.rank_results == ("done",)
    assert result.stats.rank_python_seconds == pytest.approx((0.75,))
    assert result.stats.rank_compute_seconds == pytest.approx((1.0,))
    assert result.stats.rank_finish_seconds == pytest.approx((1.75,))


def test_measure_python_must_be_boolean() -> None:
    with pytest.raises(TypeError, match="measure_python"):
        Simulator(1, measure_python=1)  # type: ignore[arg-type]


@pytest.mark.parametrize("rate", [0.0, -1.0, float("inf"), float("nan")])
def test_invalid_bandwidth_is_rejected(rate: float) -> None:
    with pytest.raises(ValueError, match="bandwidth"):
        Simulator(2, bandwidth=rate)
