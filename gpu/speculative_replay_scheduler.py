"""Exact one-thread CPU speculation alongside one GPU decoder attempt.

The GPU and CPU callback receive immutable source-shot indices. GPU status 1
and status 2 are accepted. Status 3, malformed output, and backend failure use
the exact CPU result. CPU work completed for a GPU-accepted shot is discarded;
it can affect latency, never the decoded answer.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Callable, Protocol, Sequence


GPU_EXACT = 1
GPU_LOW_CONFIDENCE = 2
GPU_WORKSPACE_OVERFLOW = 3
GPU_ACCEPTED = frozenset((GPU_EXACT, GPU_LOW_CONFIDENCE))

DECISION_FIELDS = (
    "low_confidence",
    "predicted_errors_populated",
    "total_cost_bits",
    "predicted_errors",
    "predictions",
)


class GpuAttempt(Protocol):
    def decode_indices(self, shot_indices: Sequence[int]) -> tuple[list[dict], dict]: ...


CpuDecodeOne = Callable[[int], dict]


def _gpu_decision_is_accepted(decision: object) -> bool:
    if not isinstance(decision, dict):
        return False
    if any(field not in decision for field in DECISION_FIELDS):
        return False
    try:
        status = int(decision.get("status", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if status not in GPU_ACCEPTED:
        return False
    return bool(decision["low_confidence"]) == (status == GPU_LOW_CONFIDENCE)


def _validate_cpu_decision(shot: int, decision: object) -> dict:
    if not isinstance(decision, dict):
        raise TypeError(f"CPU replay for shot {shot} did not return a dictionary")
    missing = [field for field in DECISION_FIELDS if field not in decision]
    if missing:
        raise ValueError(f"CPU replay for shot {shot} is missing fields: {missing}")
    return decision


def _normalize_unique_shots(values: Sequence[int], *, name: str) -> list[int]:
    result = [int(value) for value in values]
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate source-shot indices")
    return result


def decode_speculative_attempt(
    shot_indices: Sequence[int],
    gpu_runner: GpuAttempt,
    cpu_decode_one: CpuDecodeOne,
    *,
    cpu_priority: Sequence[int] | None = None,
) -> tuple[list[dict], dict]:
    """Decode one batch with exact CPU speculation on a single worker thread.

    ``cpu_priority`` is a performance-only ordering of zero or more source
    shots. The default speculates every requested shot in request order. Once
    GPU statuses arrive, pending work for accepted shots is cancelled, missing
    fallback shots are appended, and every required CPU result is awaited.

    Returned decisions follow ``shot_indices`` exactly, even when the CPU
    priority and completion order differ. The runner stays open for reuse.
    """
    requested = _normalize_unique_shots(shot_indices, name="shot_indices")
    requested_set = set(requested)
    priority = (
        list(requested)
        if cpu_priority is None
        else _normalize_unique_shots(cpu_priority, name="cpu_priority")
    )
    unknown = [shot for shot in priority if shot not in requested_set]
    if unknown:
        raise ValueError(f"cpu_priority contains unrequested shots: {unknown[:20]}")

    positions = {shot: position for position, shot in enumerate(requested)}
    final: list[dict | None] = [None] * len(requested)
    routes = ["" for _ in requested]
    started_order: list[int] = []
    completed_order: list[int] = []
    cpu_durations_ms: dict[int, float] = {}
    cpu_errors: dict[int, str] = {}
    record_lock = threading.Lock()
    policy_lock = threading.Lock()
    gpu_status_known = False
    required_after_gpu: set[int] = set()
    canceled_before_start: list[int] = []

    class CpuNoLongerNeeded(Exception):
        pass

    def run_cpu(shot: int) -> dict:
        with policy_lock:
            if gpu_status_known and shot not in required_after_gpu:
                with record_lock:
                    canceled_before_start.append(shot)
                raise CpuNoLongerNeeded
        with record_lock:
            started_order.append(shot)
        start = time.perf_counter()
        try:
            decision = _validate_cpu_decision(shot, cpu_decode_one(shot))
        except BaseException as ex:
            with record_lock:
                cpu_errors[shot] = f"{type(ex).__name__}: {ex}"
                cpu_durations_ms[shot] = (time.perf_counter() - start) * 1000.0
            raise
        with record_lock:
            completed_order.append(shot)
            cpu_durations_ms[shot] = (time.perf_counter() - start) * 1000.0
        return decision

    overall_start = time.perf_counter()
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tesseract-cpu")
    futures: dict[int, Future[dict]] = {}
    gpu_stats: dict = {}
    gpu_error: str | None = None
    gpu_decisions: list[dict] | None = None
    gpu_wall_ms = 0.0
    try:
        for shot in priority:
            futures[shot] = executor.submit(run_cpu, shot)

        gpu_start = time.perf_counter()
        try:
            candidate_decisions, candidate_stats = gpu_runner.decode_indices(requested)
            if len(candidate_decisions) != len(requested):
                raise ValueError("GPU runner returned the wrong decision count")
            gpu_decisions = candidate_decisions
            gpu_stats = candidate_stats
        except Exception as ex:
            gpu_error = f"{type(ex).__name__}: {ex}"
        gpu_wall_ms = (time.perf_counter() - gpu_start) * 1000.0

        accepted: list[int] = []
        fallback: list[int] = []
        with policy_lock:
            # Holding this lock while classifying the returned batch prevents
            # the serial worker from beginning another candidate between GPU
            # completion and publication of the required fallback set.
            if gpu_decisions is None:
                fallback = list(requested)
                for shot in requested:
                    routes[positions[shot]] = "gpu:backend-error->cpu"
            else:
                for shot, decision in zip(requested, gpu_decisions, strict=True):
                    position = positions[shot]
                    if _gpu_decision_is_accepted(decision):
                        status = int(decision["status"])
                        final[position] = decision
                        routes[position] = f"gpu:status-{status}"
                        accepted.append(shot)
                    else:
                        try:
                            status = int(decision.get("status", 0))
                        except (AttributeError, TypeError, ValueError, OverflowError):
                            status = 0
                        route_status = (
                            "status-3"
                            if status == GPU_WORKSPACE_OVERFLOW
                            else f"invalid-{status}"
                        )
                        routes[position] = f"gpu:{route_status}->cpu"
                        fallback.append(shot)
            required_after_gpu.update(fallback)
            gpu_status_known = True

        # Futures are submitted in priority order to one worker. Cancelling a
        # pending accepted future removes it from the serial stream. One
        # already-running callback cannot be interrupted safely and is simply
        # ignored if the GPU answer was accepted.
        for shot in accepted:
            future = futures.get(shot)
            if future is not None and future.cancel():
                with record_lock:
                    canceled_before_start.append(shot)

        for shot in fallback:
            if shot not in futures:
                futures[shot] = executor.submit(run_cpu, shot)

        for shot in fallback:
            decision = futures[shot].result()
            position = positions[shot]
            if final[position] is not None:
                raise RuntimeError(f"shot {shot} has both accepted GPU and CPU output")
            final[position] = decision
            routes[position] += "|cpu:replayed"
    finally:
        # No unneeded queued task may start after GPU status is known. Required
        # fallbacks have already been awaited; this only joins an in-flight
        # speculative callback and cancels any remaining accepted work.
        executor.shutdown(wait=True, cancel_futures=True)
    total_wall_ms = (time.perf_counter() - overall_start) * 1000.0

    unresolved = [requested[i] for i, decision in enumerate(final) if decision is None]
    if unresolved:
        raise RuntimeError(f"speculative scheduler left unresolved shots: {unresolved}")
    fallback_set = set(fallback)
    completed_set = set(completed_order)
    return [decision for decision in final if decision is not None], {
        "requested_shots": requested,
        "cpu_priority": priority,
        "routes": routes,
        "gpu_wall_ms": gpu_wall_ms,
        "total_wall_ms": total_wall_ms,
        "gpu_backend_error": gpu_error,
        "gpu_stats": gpu_stats,
        "cpu_required_shots": fallback,
        "cpu_used_shots": [shot for shot in fallback if shot in completed_set],
        "cpu_started_order": started_order,
        "cpu_completed_order": completed_order,
        "cpu_canceled_before_start": canceled_before_start,
        "cpu_completed_but_not_used": [
            shot for shot in completed_order if shot not in fallback_set
        ],
        "cpu_durations_ms": cpu_durations_ms,
        "cpu_errors": cpu_errors,
    }


def compare_exact_decisions(reference: Sequence[dict], actual: Sequence[dict]) -> list[dict]:
    """Compare all externally visible decoder fields, including list order."""
    if len(reference) != len(actual):
        raise ValueError("decision list lengths differ")
    mismatches = []
    for position, (expected, observed) in enumerate(
        zip(reference, actual, strict=True)
    ):
        differing = [
            field
            for field in DECISION_FIELDS
            if expected.get(field) != observed.get(field)
        ]
        if differing:
            mismatches.append({"position": position, "fields": differing})
    return mismatches
