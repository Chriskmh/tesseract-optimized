"""Device-aware resource policy; no search or arithmetic parameters are changed.

SM counts are read from the current CUDA context (including MPS/MIG restrictions).
Batch sizing is a throughput heuristic, not a reservation of compute resources.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable


PROFILES = ("auto", "legacy", "dedicated", "shared")


@dataclass(frozen=True)
class DeviceInfo:
    device_id: int
    name: str
    sm_count: int
    compute_major: int
    compute_minor: int

    def __post_init__(self):
        if self.device_id < 0 or self.sm_count < 1 or self.compute_major < 1:
            raise ValueError("invalid CUDA device topology")


def cuda_device_info() -> DeviceInfo:
    """Query only the selected device; do not poll other users' processes."""
    import cupy as cp

    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    # Device attributes reflect the calling context's MPS resource provision.
    return DeviceInfo(device.id, str(name), int(device.attributes["MultiProcessorCount"]),
                      int(props["major"]), int(props["minor"]))


@dataclass(frozen=True)
class GpuResources:
    profile: str = "auto"
    memory_limit_bytes: int | None = None
    memory_share: float | None = None
    compute_share: float | None = None
    max_chunk_size: int | None = None
    pause_ms: float | None = None

    def __post_init__(self):
        if self.profile not in PROFILES:
            raise ValueError(f"GPU profile must be one of {PROFILES}")
        for name in ("memory_limit_bytes", "max_chunk_size"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)
                                      or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        for name in ("memory_share", "compute_share"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or not 0 < value <= 1):
                raise ValueError(f"{name} must be in (0, 1]")
        if self.pause_ms is not None and (not math.isfinite(self.pause_ms) or self.pause_ms < 0):
            raise ValueError("pause_ms must be finite and nonnegative")

    @property
    def effective_memory_share(self) -> float | None:
        return 0.25 if self.profile == "shared" and self.memory_share is None else self.memory_share

    @property
    def effective_compute_share(self) -> float:
        if self.compute_share is not None:
            return self.compute_share
        return 0.5 if self.profile == "shared" else 1.0

    @property
    def effective_pause_ms(self) -> float:
        if self.pause_ms is not None:
            return self.pause_ms
        return 1.0 if self.profile == "shared" else 0.0

    @property
    def limits_memory(self) -> bool:
        return self.memory_limit_bytes is not None or self.effective_memory_share is not None


def resolve_resources(
    resources: GpuResources,
    preferred_chunk_size: int,
    *,
    device_info: DeviceInfo | Callable[[], DeviceInfo] | None = None,
) -> dict:
    """Keep measured small-device defaults; scale only independent shots.

    The two-wave policy on modern >64-SM devices is unbenchmarked. Thread-block
    geometry, heap cache, workspace candidates and decoder selection stay fixed.
    """
    device = None
    error = None
    if resources.profile != "legacy" or resources.compute_share is not None:
        try:
            device = cuda_device_info() if device_info is None else (
                device_info() if callable(device_info) else device_info
            )
            if not isinstance(device, DeviceInfo):
                raise TypeError("device_info must return DeviceInfo")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
    desired = int(preferred_chunk_size)
    reason = "measured-default"
    effective_sms = None
    if device is not None:
        effective_sms = max(1, math.ceil(device.sm_count * resources.effective_compute_share))
        if resources.profile == "shared":
            # At most one block per advisory share of visible SMs. This does
            # not bind blocks to particular SMs or cap memory bandwidth.
            desired = min(desired, effective_sms)
            reason = "shared-concurrency-budget"
        elif resources.profile == "dedicated" or (
            resources.profile == "auto" and device.compute_major >= 8 and device.sm_count > 64
        ):
            desired = max(desired, min(1024, 1 << (2 * effective_sms - 1).bit_length()))
            reason = "unbenchmarked-two-wave-policy"
        if resources.compute_share is not None:
            desired = min(desired, effective_sms)
            reason = "explicit-concurrency-budget"
    elif resources.profile == "shared":
        desired = 1
        reason = "shared-no-topology-conservative"
    elif resources.compute_share is not None:
        desired = 1
        reason = "explicit-share-no-topology-conservative"
    if resources.max_chunk_size is not None:
        desired = min(desired, resources.max_chunk_size)
    return {
        "requested": asdict(resources),
        "device": asdict(device) if device is not None else None,
        "device_info_error": error,
        "reason": reason,
        "effective_sm_target": effective_sms,
        "preferred_chunk_size": desired,
        "memory_share": resources.effective_memory_share,
        "memory_limit_bytes": resources.memory_limit_bytes,
        "pause_ms": resources.effective_pause_ms,
        "compute_control": "advisory-shot-concurrency; use MPS/MIG for isolation",
    }


__all__ = ["DeviceInfo", "GpuResources", "PROFILES", "cuda_device_info", "resolve_resources"]
