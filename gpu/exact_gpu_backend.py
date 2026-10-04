"""Usable exact hybrid dispatcher for the validated CUDA decoder variants.

The dispatcher selects a resident GPU decoder from immutable model features,
keeps its compiled module and workspace alive across calls, and treats CUDA as
an exact attempt rather than as an oracle.  Status 1 and status 2 decisions are
accepted only after structural validation.  Status 3, malformed output, and
any backend failure are discarded and replayed through an injected exact CPU
callback in original shot order.

No status-3 payload can escape this module.  When replay is required and no
CPU callback exists, decoding fails explicitly instead of returning a partial
or approximate result.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from speculative_replay_scheduler import (
    DECISION_FIELDS,
    GPU_ACCEPTED,
    GPU_EXACT,
    GPU_LOW_CONFIDENCE,
    GPU_WORKSPACE_OVERFLOW,
    _gpu_decision_is_accepted,
    _validate_cpu_decision,
)
from resident_chunk_runner import ResidentChunkedDecoder, max_chunk_detections


MIB = 1 << 20
DEFAULT_MEMORY_RESERVE_BYTES = 512 * MIB
DEFAULT_MEMORY_FRACTION = 0.75
MAX_OUTPUT = 1024
WORKSPACE_LAYOUT_LEGACY = "legacy"
WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT = "bb-singlehash-compact"
SURFACE_ROW_LIMIT = 128
LARGE_SURFACE_DETECTOR_THRESHOLD = 6000
LARGE_SURFACE_WORKSPACE = 40000
ALLOWED_CTA_THREADS = (32, 64, 128, 256, 512, 544, 576, 608, 640)

BACKEND_FINAL = "final"
BACKEND_SIBLING = "sibling"
BACKEND_SIBLING_BASE = "sibling-base"
BACKEND_PREFETCH = "prefetch"
BACKEND_FUSED = "fused"
BACKEND_KEYS = (
    BACKEND_FINAL, BACKEND_SIBLING, BACKEND_SIBLING_BASE,
    BACKEND_PREFETCH, BACKEND_FUSED,
)


@dataclass(frozen=True)
class BackendDefaults:
    """GPU-validated launch defaults and ordered workspace preferences."""

    module: str
    cta_threads: int
    workspace_candidates: tuple[int, ...]
    preferred_chunk_size: int
    heap_cache: int | None = None
    stamp_layout: str | None = None


BACKEND_DEFAULTS: dict[str, BackendDefaults] = {
    # Color d=11: production no-telemetry prefix fold, checked CSR32, and the
    # audited hot-pointer restrict contract. CTA256/heap128 is the measured
    # configuration; 100000 is preferred to avoid CPU replay.
    BACKEND_FINAL: BackendDefaults(
        module="cta_compact_final_nodebug_restrict_hot_decoder",
        cta_threads=256,
        heap_cache=128,
        workspace_candidates=(100000, 60000, 32768, 8192),
        preferred_chunk_size=256,
    ),
    # Surface d=15/d=17/d=19: production no-telemetry sibling rank32 with
    # exact monotonic stop prefetch. CTA256/W32768 keeps replay enabled.
    BACKEND_SIBLING: BackendDefaults(
        module="cta_sibling_rank32_prefetch_nodebug_decoder",
        cta_threads=256,
        stamp_layout="packed",
        workspace_candidates=(32768, 8192, 4096),
        preferred_chunk_size=256,
    ),
    # Retain the pre-prefetch Surface implementation for explicit A/B use.
    BACKEND_SIBLING_BASE: BackendDefaults(
        module="cta_sibling_rank32_decoder",
        cta_threads=256,
        stamp_layout="packed",
        workspace_candidates=(32768, 8192, 4096),
        preferred_chunk_size=128,
    ),
    # BB d=18 winner: exact single-hash collision checks, narrow restrict,
    # compact visited table, checked CSR32, CTA640/heap1024. Prefer full 100000
    # arena; status 3 remains an exact CPU replay request.
    BACKEND_PREFETCH: BackendDefaults(
        module="cta_compact_fused_rank32_nodebug_csr32_singlehash_restrict_hot_compactvisited_cta640_decoder",
        cta_threads=640,
        heap_cache=1024,
        workspace_candidates=(100000, 60000, 38000, 32768, 4096),
        preferred_chunk_size=64,
    ),
    # Retain the pre-prefetch implementation as an explicit A/B override.
    BACKEND_FUSED: BackendDefaults(
        module="cta_compact_fused_rank32_decoder",
        cta_threads=512,
        heap_cache=1024,
        workspace_candidates=(100000, 60000, 38000, 32768, 4096),
        preferred_chunk_size=64,
    ),
}


BACKEND_ALIASES = {
    "auto": "auto",
    "color": BACKEND_FINAL,
    "prefixfold": BACKEND_FINAL,
    "csr32": BACKEND_FINAL,
    "cta_compact_final_decoder": BACKEND_FINAL,
    "cta_compact_final_nodebug_decoder": BACKEND_FINAL,
    "cta_compact_final_nodebug_restrict_hot_decoder": BACKEND_FINAL,
    BACKEND_FINAL: BACKEND_FINAL,
    "surface": BACKEND_SIBLING,
    "surface-prefetch": BACKEND_SIBLING,
    "cta_sibling_rank32_prefetch_decoder": BACKEND_SIBLING,
    "cta_sibling_rank32_prefetch_nodebug_decoder": BACKEND_SIBLING,
    BACKEND_SIBLING: BACKEND_SIBLING,
    "surface-base": BACKEND_SIBLING_BASE,
    "cta_sibling_rank32_decoder": BACKEND_SIBLING_BASE,
    BACKEND_SIBLING_BASE: BACKEND_SIBLING_BASE,
    "bb": BACKEND_PREFETCH,
    "prefetch": BACKEND_PREFETCH,
    "fused-prefetch": BACKEND_PREFETCH,
    "cta_compact_fused_rank32_prefetch_decoder": BACKEND_PREFETCH,
    "cta_compact_fused_rank32_nodebug_csr32_decoder": BACKEND_PREFETCH,
    "cta_compact_fused_rank32_nodebug_csr32_singlehash_restrict_hot_compactvisited_cta640_decoder": BACKEND_PREFETCH,
    BACKEND_PREFETCH: BACKEND_PREFETCH,
    "fused-base": BACKEND_FUSED,
    "cta_compact_fused_rank32_decoder": BACKEND_FUSED,
    BACKEND_FUSED: BACKEND_FUSED,
}


class Decoder(Protocol):
    def decode(self) -> tuple[list[dict], dict]: ...


DecoderFactory = Callable[..., Decoder]
CpuExactCallback = Callable[[int], dict]
MemoryInfoProvider = Callable[[], tuple[int, int]]


class ExactFallbackRequiredError(RuntimeError):
    """Raised when exact CPU replay is necessary but unavailable."""

    def __init__(self, shots: Sequence[int], reason: str):
        self.shots = tuple(int(shot) for shot in shots)
        self.reason = str(reason)
        preview = list(self.shots[:20])
        suffix = "..." if len(self.shots) > len(preview) else ""
        super().__init__(
            "exact CPU fallback is required for shots "
            f"{preview}{suffix}: {self.reason}"
        )


@dataclass(frozen=True)
class ModelFeatures:
    shots: int
    num_detectors: int
    num_errors: int
    rank_rows_monotonic: bool
    max_d2e_row: int


@dataclass(frozen=True)
class BackendSelection:
    backend: str
    reason: str
    features: ModelFeatures
    overridden: bool


@dataclass(frozen=True)
class WorkspacePlan:
    backend: str
    workspace_capacity: int
    cta_threads: int
    heap_cache: int | None
    stamp_layout: str | None
    source_shots: int
    chunk_size: int
    detection_capacity: int
    free_bytes: int | None
    total_bytes: int | None
    usable_bytes: int | None
    graph_bytes: int
    chunk_input_bytes: int
    workspace_bytes: int
    estimated_device_bytes: int
    memory_info_error: str | None
    source: str
    workspace_layout: str = WORKSPACE_LAYOUT_LEGACY


def inspect_model(model: Any) -> ModelFeatures:
    """Extract and cross-check only the immutable selector inputs."""
    metadata = model.metadata
    shots = int(metadata["shots"])
    num_detectors = int(metadata["num_detectors"])
    num_errors = int(metadata["num_errors"])
    if min(shots, num_detectors, num_errors) < 0:
        raise ValueError("model dimensions must be nonnegative")
    offsets = np.asarray(model.d2e_offsets)
    if offsets.ndim != 1 or len(offsets) != num_detectors + 1:
        raise ValueError("d2e_offsets do not match num_detectors")
    if offsets.dtype.kind not in "iu":
        raise ValueError("d2e_offsets must be integer")
    if offsets.dtype.kind == "i" and bool(np.any(offsets < 0)):
        raise ValueError("d2e_offsets must be nonnegative")
    wide = offsets.astype(np.uint64, copy=False)
    if bool(np.any(wide[1:] < wide[:-1])):
        raise ValueError("d2e_offsets must be monotonic")
    max_row = int(np.diff(wide).max(initial=0))
    monotonic = metadata.get("rank_rows_monotonic")
    if not isinstance(monotonic, (bool, np.bool_)):
        raise ValueError("rank_rows_monotonic must be a validated boolean")
    return ModelFeatures(
        shots=shots,
        num_detectors=num_detectors,
        num_errors=num_errors,
        rank_rows_monotonic=bool(monotonic),
        max_d2e_row=max_row,
    )


def normalize_backend_override(value: str | None) -> str:
    if value is None:
        return "auto"
    key = str(value).strip().lower()
    try:
        return BACKEND_ALIASES[key]
    except KeyError as exc:
        raise ValueError(
            f"unknown backend override {value!r}; expected auto or one of "
            f"{BACKEND_KEYS}"
        ) from exc


def select_backend(model: Any, override: str | None = None) -> BackendSelection:
    """Select solely from rank-row semantics and maximum detector-row width."""
    features = inspect_model(model)
    normalized = normalize_backend_override(override)
    if normalized != "auto":
        return BackendSelection(
            backend=normalized,
            reason=f"explicit user override: {normalized}",
            features=features,
            overridden=True,
        )
    if not features.rank_rows_monotonic:
        return BackendSelection(
            backend=BACKEND_FINAL,
            reason="nonmonotonic rank rows require exact prefix fold",
            features=features,
            overridden=False,
        )
    if features.max_d2e_row <= SURFACE_ROW_LIMIT:
        return BackendSelection(
            backend=BACKEND_SIBLING,
            reason=(
                f"monotonic ranks and max d2e row {features.max_d2e_row} "
                f"<= {SURFACE_ROW_LIMIT}"
            ),
            features=features,
            overridden=False,
        )
    return BackendSelection(
        backend=BACKEND_PREFETCH,
        reason=(
            f"monotonic ranks but max d2e row {features.max_d2e_row} "
            f"> {SURFACE_ROW_LIMIT}"
        ),
        features=features,
        overridden=False,
    )


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (int(value) - 1).bit_length())


def estimate_workspace_bytes(
    features: ModelFeatures,
    *,
    backend: str,
    workspace_capacity: int,
    cta_threads: int,
    shot_count: int | None = None,
    max_output: int = MAX_OUTPUT,
    workspace_layout: str = WORKSPACE_LAYOUT_LEGACY,
) -> int:
    """Conservatively bound one resident decoder's workspace allocation.

    The compact BB option matches the isolated single-hash/restrict/compact-
    visited production backend.  Custom factories retain the legacy bound.
    The chunk-input estimate separately includes
    its resident ``shot_hash2`` ABI buffer; graph estimation still charges for
    the original full ``error_hash2`` buffer, which is conservative.
    """
    workspace_capacity = int(workspace_capacity)
    cta_threads = int(cta_threads)
    if workspace_capacity < 2:
        raise ValueError("workspace_capacity must be at least 2")
    if cta_threads not in ALLOWED_CTA_THREADS:
        raise ValueError(f"cta_threads must be one of {ALLOWED_CTA_THREADS}")
    if backend not in BACKEND_KEYS:
        raise ValueError(f"unknown backend: {backend}")
    if workspace_layout not in (
        WORKSPACE_LAYOUT_LEGACY, WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT,
    ):
        raise ValueError(f"unknown workspace layout: {workspace_layout}")
    if (
        workspace_layout == WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT
        and backend != BACKEND_PREFETCH
    ):
        raise ValueError("bb-singlehash-compact requires the prefetch backend")
    resident_shots = features.shots if shot_count is None else int(shot_count)
    if resident_shots < 0:
        raise ValueError("shot_count must be nonnegative")
    words = (features.num_detectors + 63) // 64
    if workspace_layout == WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT:
        # The exact single-hash allocator has eleven per-chain arrays totaling
        # 52 bytes, one uint64 hash plus one uint32 chain per visited slot,
        # and distinct one-element hash2 placeholders for the restrict ABI.
        # W+1 states can be inserted; W+2 preserves an empty probe slot.
        visited_capacity = _next_power_of_two(workspace_capacity + 2)
        if visited_capacity > np.iinfo(np.int32).max:
            raise ValueError("compact visited capacity exceeds signed int32")
        arena = workspace_capacity * 52
        visited = visited_capacity * (8 + 4)
        placeholders = 2 * np.dtype(np.uint64).itemsize
        # CuPy's pool rounds small allocations.  Thirty distinct workspace
        # arrays each get a full extra 512-byte allowance, independent of
        # source shots, so this remains above requested device bytes.
        allocation_allowance = 30 * 512
    else:
        visited_capacity = _next_power_of_two(2 * workspace_capacity)
        arena = workspace_capacity * 60
        visited = visited_capacity * (8 + 8 + 4)
        placeholders = 0
        allocation_allowance = 0
    if backend in (BACKEND_SIBLING, BACKEND_SIBLING_BASE):
        warps = cta_threads // 32
        error_state = features.num_errors * (5 + warps)
    else:
        error_state = features.num_errors * 4
    detector_cache = features.num_detectors * (8 + 4)
    syndrome_scratch = words * 8 * 4
    outputs = max_output * 4 + 96
    per_shot = (
        arena + visited + error_state + detector_cache
        + syndrome_scratch + outputs
    )
    pool_only = 4 if backend in (BACKEND_SIBLING, BACKEND_SIBLING_BASE) else 0
    return resident_shots * per_shot + pool_only + placeholders + allocation_allowance


STATIC_DEVICE_MODEL_ARRAYS = (
    "d2e_offsets", "d2e", "edet_offsets", "edets",
    "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
    "weights", "error_to_dem", "detector_order", "error_hash1",
    "error_hash2", "count_values", "count_ranks16",
)


def estimate_graph_bytes(model: Any) -> int:
    """Conservatively total immutable arrays uploaded once per decoder."""
    total = 0
    for name in STATIC_DEVICE_MODEL_ARRAYS:
        value = getattr(model, name, None)
        if value is not None:
            total += int(np.asarray(value).nbytes)
    ranked = getattr(model, "d2e_ranked", None)
    if ranked is not None:
        total += len(ranked) * np.dtype(np.uint32).itemsize
    return total


def estimate_chunk_input_bytes(
    model: Any,
    chunk_size: int,
    *,
    detection_capacity: int | None = None,
) -> int:
    """Estimate the five mutable resident shot buffers for one chunk.

    Offset storage is counted as uint64 even though the final Color backend
    uses checked uint32 offsets.  That keeps memory admission conservative.
    """
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    required = max_chunk_detections(model, chunk_size)
    capacity = required if detection_capacity is None else int(detection_capacity)
    if capacity < required:
        raise ValueError(
            f"detection_capacity {capacity} is below required {required}"
        )
    return _chunk_input_bytes_for_capacity(
        int(model.metadata["num_detectors"]), chunk_size, capacity
    )


def _chunk_input_bytes_for_capacity(
    num_detectors: int, chunk_size: int, detection_capacity: int
) -> int:
    words = (int(num_detectors) + 63) // 64
    return (
        (chunk_size + 1) * np.dtype(np.uint64).itemsize
        + detection_capacity * np.dtype(np.uint32).itemsize
        + chunk_size * words * np.dtype(np.uint64).itemsize
        + 2 * chunk_size * np.dtype(np.uint64).itemsize
    )


def cuda_mem_get_info() -> tuple[int, int]:
    """Read `(free, total)` lazily so host-only tests never import CUDA."""
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


def _read_memory_info(
    provider: MemoryInfoProvider | tuple[int, int] | None,
) -> tuple[tuple[int, int] | None, str | None]:
    try:
        raw = cuda_mem_get_info() if provider is None else (
            provider() if callable(provider) else provider
        )
        free, total = int(raw[0]), int(raw[1])
        if free <= 0 or total <= 0 or free > total:
            raise ValueError(f"invalid memGetInfo values: free={free}, total={total}")
        return (free, total), None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def plan_workspace(
    model: Any,
    selection: BackendSelection,
    *,
    memory_info: MemoryInfoProvider | tuple[int, int] | None = None,
    workspace_override: int | None = None,
    chunk_size_override: int | None = None,
    cta_threads_override: int | None = None,
    heap_cache_override: int | None = None,
    stamp_layout_override: str | None = None,
    reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    workspace_layout: str = WORKSPACE_LAYOUT_LEGACY,
) -> WorkspacePlan:
    """Choose an arena first, then a resident chunk fitting free memory.

    Total source shots do not scale the allocation.  The planner preserves the
    preferred exact-search arena whenever at least one resident shot fits, and
    streams an arbitrary source batch through that fixed allocation.
    """
    defaults = BACKEND_DEFAULTS[selection.backend]
    cta_threads = int(
        defaults.cta_threads if cta_threads_override is None
        else cta_threads_override
    )
    if cta_threads not in ALLOWED_CTA_THREADS:
        raise ValueError(f"cta_threads must be one of {ALLOWED_CTA_THREADS}")
    heap_cache = (
        defaults.heap_cache if heap_cache_override is None
        else int(heap_cache_override)
    )
    sibling_backend = selection.backend in (
        BACKEND_SIBLING, BACKEND_SIBLING_BASE,
    )
    if sibling_backend:
        if heap_cache_override is not None:
            raise ValueError("sibling backend does not have a heap_cache option")
        heap_cache = None
    elif heap_cache not in (128, 256, 512, 1024):
        raise ValueError("heap_cache must be one of 128, 256, 512, 1024")
    stamp_layout = (
        defaults.stamp_layout if stamp_layout_override is None
        else str(stamp_layout_override)
    )
    if sibling_backend:
        if stamp_layout not in ("packed", "separate"):
            raise ValueError("stamp_layout must be packed or separate")
    elif stamp_layout_override is not None:
        raise ValueError("stamp_layout applies only to sibling backend")
    else:
        stamp_layout = None
    if reserve_bytes < 0:
        raise ValueError("reserve_bytes must be nonnegative")
    if not 0.0 < memory_fraction <= 1.0:
        raise ValueError("memory_fraction must be in (0, 1]")
    source_shots = int(selection.features.shots)
    if chunk_size_override is None:
        desired_chunk_size = min(
            max(1, source_shots), defaults.preferred_chunk_size
        )
    else:
        desired_chunk_size = int(chunk_size_override)
        if desired_chunk_size < 1:
            raise ValueError("chunk_size_override must be positive")
        if desired_chunk_size > np.iinfo(np.int32).max:
            raise ValueError("chunk_size_override exceeds the kernel int32 ABI")

    info, info_error = _read_memory_info(memory_info)
    free = info[0] if info else None
    total = info[1] if info else None
    usable = None
    if free is not None:
        usable = max(
            0,
            min(free - int(reserve_bytes), int(free * memory_fraction)),
        )
    graph_bytes = estimate_graph_bytes(model)
    if workspace_override is not None:
        candidates = (int(workspace_override),)
        source = "user-override"
    elif info is None:
        # Measured conservative fallbacks when memGetInfo is unavailable.
        candidates = {
            BACKEND_FINAL: (60000,),
            BACKEND_SIBLING: (32768,),
            BACKEND_SIBLING_BASE: (32768,),
            BACKEND_PREFETCH: (38000,),
            BACKEND_FUSED: (38000,),
        }[selection.backend]
        source = "conservative-no-memory-info"
    else:
        candidates = defaults.workspace_candidates
        # The d=19 acceptance export crosses a search-frontier step at W=40000:
        # CPU replays fall from seven to four and replay-adjusted wall time is
        # lower.  d=15/d=17 do not benefit, so keep their measured W=32768
        # default and key this policy only from immutable graph size.
        if (
            selection.backend == BACKEND_SIBLING
            and selection.features.num_detectors
            >= LARGE_SURFACE_DETECTOR_THRESHOLD
        ):
            candidates = tuple(dict.fromkeys(
                (LARGE_SURFACE_WORKSPACE, *candidates)
            ))
        source = "memGetInfo"

    chunk_sizes = (
        (desired_chunk_size,)
        if chunk_size_override is not None or usable is None
        else tuple(range(desired_chunk_size, 0, -1))
    )
    detection_capacities: dict[int, int] = {}
    chunk_bytes: dict[int, int] = {}
    rejected: list[tuple[int, int, int]] = []
    for capacity in candidates:
        smallest_estimate = 0
        for chunk_size in chunk_sizes:
            if chunk_size not in detection_capacities:
                detection_capacities[chunk_size] = max_chunk_detections(
                    model, chunk_size
                )
                chunk_bytes[chunk_size] = _chunk_input_bytes_for_capacity(
                    selection.features.num_detectors,
                    chunk_size,
                    detection_capacities[chunk_size],
                )
            workspace_bytes = estimate_workspace_bytes(
                selection.features,
                backend=selection.backend,
                workspace_capacity=capacity,
                cta_threads=cta_threads,
                shot_count=chunk_size,
                workspace_layout=workspace_layout,
            )
            estimate = graph_bytes + chunk_bytes[chunk_size] + workspace_bytes
            smallest_estimate = estimate
            if usable is None or estimate <= usable:
                return WorkspacePlan(
                    backend=selection.backend,
                    workspace_capacity=capacity,
                    cta_threads=cta_threads,
                    heap_cache=heap_cache,
                    stamp_layout=stamp_layout,
                    source_shots=source_shots,
                    chunk_size=chunk_size,
                    detection_capacity=detection_capacities[chunk_size],
                    free_bytes=free,
                    total_bytes=total,
                    usable_bytes=usable,
                    graph_bytes=graph_bytes,
                    chunk_input_bytes=chunk_bytes[chunk_size],
                    workspace_bytes=workspace_bytes,
                    estimated_device_bytes=estimate,
                    memory_info_error=info_error,
                    source=source,
                    workspace_layout=workspace_layout,
                )
        rejected.append((int(capacity), chunk_sizes[-1], smallest_estimate))
    raise MemoryError(
        f"no {selection.backend} workspace fits conservative GPU budget "
        f"{usable}; rejected (workspace, smallest chunk, bytes) {rejected}"
    )


def load_default_decoder_class(
    backend: str, *, cta_threads: int | None = None, heap_cache: int | None = None,
) -> type:
    """Load only the selected CUDA implementation."""
    if backend == BACKEND_FINAL:
        from cta_compact_final_nodebug_restrict_hot_decoder import ExactCtaDecoder
    elif backend == BACKEND_SIBLING:
        from cta_sibling_rank32_prefetch_nodebug_decoder import ExactCtaDecoder
    elif backend == BACKEND_SIBLING_BASE:
        from cta_sibling_rank32_decoder import ExactCtaDecoder
    elif backend == BACKEND_PREFETCH:
        if cta_threads not in (None, 640) or heap_cache not in (None, 1024):
            # The composed kernel is specialized to the measured launch.  Keep
            # the prior checked CSR32 implementation for explicit tuning.
            from cta_compact_fused_rank32_nodebug_csr32_decoder import ExactCtaDecoder
        else:
            from cta_compact_fused_rank32_nodebug_csr32_singlehash_restrict_hot_compactvisited_cta640_decoder import ExactCtaDecoder
    elif backend == BACKEND_FUSED:
        from cta_compact_fused_rank32_decoder import ExactCtaDecoder
    else:
        raise ValueError(f"unknown backend: {backend}")
    return ExactCtaDecoder


def _decoder_kwargs(plan: WorkspacePlan) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "workspace_capacity": plan.workspace_capacity,
        "max_output": MAX_OUTPUT,
        "max_registers": 0,
        "cta_threads": plan.cta_threads,
    }
    if plan.backend in (BACKEND_SIBLING, BACKEND_SIBLING_BASE):
        kwargs["stamp_layout"] = plan.stamp_layout
    else:
        kwargs["heap_cache"] = plan.heap_cache
    return kwargs


def _format_error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _status_for_route(decision: object) -> str:
    try:
        status = int(decision.get("status", 0))
    except (AttributeError, TypeError, ValueError, OverflowError):
        return "malformed"
    if status == GPU_WORKSPACE_OVERFLOW:
        return "status-3"
    return f"invalid-status-{status}"


def gpu_decision_is_accepted(decision: object) -> bool:
    """Apply the proven scheduler checks plus strict payload shape checks."""
    if not _gpu_decision_is_accepted(decision):
        return False
    assert isinstance(decision, dict)
    status = decision["status"]
    if isinstance(status, (bool, np.bool_)) or not isinstance(
        status, (int, np.integer)
    ):
        return False
    if not isinstance(decision["low_confidence"], (bool, np.bool_)):
        return False
    if not isinstance(decision["predicted_errors_populated"], (bool, np.bool_)):
        return False
    if not bool(decision["predicted_errors_populated"]):
        return False
    cost_bits = decision["total_cost_bits"]
    if isinstance(cost_bits, (bool, np.bool_)) or not isinstance(
        cost_bits, (int, np.integer)
    ):
        return False
    if not 0 <= int(cost_bits) <= np.iinfo(np.uint64).max:
        return False
    for field in ("predicted_errors", "predictions"):
        values = decision[field]
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            return False
    for field in ("predicted_errors", "predictions"):
        if any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            for value in decision[field]
        ):
            return False
    return True


def _validate_cpu_exact(shot: int, decision: object) -> dict:
    validated = _validate_cpu_decision(shot, decision)
    if "status" in validated:
        try:
            status = int(validated["status"])
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                f"CPU replay for shot {shot} has malformed status"
            ) from exc
        if status == GPU_WORKSPACE_OVERFLOW:
            raise ValueError(
                f"CPU replay for shot {shot} returned forbidden status 3"
            )
        if status in GPU_ACCEPTED and (
            bool(validated["low_confidence"])
            != (status == GPU_LOW_CONFIDENCE)
        ):
            raise ValueError(
                f"CPU replay for shot {shot} has inconsistent status/low_confidence"
            )
    return validated


_CALLBACK_UNSET = object()


class ExactGpuBackend:
    """Resident selected GPU decoder with mandatory exact replay on failure."""

    def __init__(
        self,
        model: Any,
        *,
        cpu_exact_callback: CpuExactCallback | None = None,
        backend_override: str | None = None,
        workspace_override: int | None = None,
        chunk_size_override: int | None = None,
        cta_threads_override: int | None = None,
        heap_cache_override: int | None = None,
        stamp_layout_override: str | None = None,
        memory_info: MemoryInfoProvider | tuple[int, int] | None = None,
        reserve_bytes: int = DEFAULT_MEMORY_RESERVE_BYTES,
        memory_fraction: float = DEFAULT_MEMORY_FRACTION,
        decoder_class: DecoderFactory | None = None,
        decoder_classes: Mapping[str, DecoderFactory] | None = None,
    ):
        self.model = model
        self.cpu_exact_callback = cpu_exact_callback
        self.selection = select_backend(model, backend_override)
        self.plan: WorkspacePlan | None = None
        self.decoder: Decoder | None = None
        self.runner: ResidentChunkedDecoder | None = None
        self.backend_error: str | None = None
        self._decode_calls = 0

        # Injected factories can allocate different arrays.  Apply the tighter
        # bound only to the validated default implementation; their planning
        # keeps the conservative legacy layout.
        default_factory = decoder_class is None and (
            decoder_classes is None
            or decoder_classes.get(self.selection.backend) is None
        )
        workspace_layout = (
            WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT
            if default_factory and self.selection.backend == BACKEND_PREFETCH
            and cta_threads_override in (None, 640)
            and heap_cache_override in (None, 1024)
            else WORKSPACE_LAYOUT_LEGACY
        )

        try:
            self.plan = plan_workspace(
                model,
                self.selection,
                memory_info=memory_info,
                workspace_override=workspace_override,
                chunk_size_override=chunk_size_override,
                cta_threads_override=cta_threads_override,
                heap_cache_override=heap_cache_override,
                stamp_layout_override=stamp_layout_override,
                reserve_bytes=reserve_bytes,
                memory_fraction=memory_fraction,
                workspace_layout=workspace_layout,
            )
        except MemoryError as exc:
            # An insufficient conservative budget is a runtime availability
            # failure.  Preserve exactness by making the batch CPU-only.
            self.backend_error = _format_error(exc)
            return

        try:
            factory = decoder_class
            if factory is None and decoder_classes is not None:
                factory = decoder_classes.get(self.selection.backend)
            if factory is None:
                factory = load_default_decoder_class(
                    self.selection.backend,
                    cta_threads=self.plan.cta_threads,
                    heap_cache=self.plan.heap_cache,
                )
            self.runner = ResidentChunkedDecoder(
                model,
                factory,
                _decoder_kwargs(self.plan),
                chunk_size=self.plan.chunk_size,
                detection_capacity=self.plan.detection_capacity,
            )
            # Public compatibility handle for launch configuration, workspace
            # inspection, and existing callers.  Decode orchestration belongs
            # to the resident runner.
            self.decoder = self.runner.decoder
        except Exception as exc:
            self.backend_error = _format_error(exc)

    @property
    def resident(self) -> bool:
        return self.runner is not None and self.runner.resident

    def decode(
        self,
        *,
        cpu_exact_callback: CpuExactCallback | None | object = _CALLBACK_UNSET,
    ) -> tuple[list[dict], dict]:
        """Decode every model shot and scatter exact CPU fallbacks in order."""
        self._decode_calls += 1
        callback = (
            self.cpu_exact_callback
            if cpu_exact_callback is _CALLBACK_UNSET
            else cpu_exact_callback
        )
        shots = self.selection.features.shots
        final: list[dict | None] = [None] * shots
        routes = ["" for _ in range(shots)]
        gpu_stats: dict = {}
        call_error = self.backend_error
        gpu_decisions: list[dict] | None = None

        backend_failed_shots: set[int] = set()
        if self.runner is not None:
            try:
                result = self.runner.decode()
                if not isinstance(result, tuple) or len(result) != 2:
                    raise TypeError("GPU decoder must return (decisions, stats)")
                candidate_decisions, candidate_stats = result
                if not isinstance(candidate_decisions, Sequence) or isinstance(
                    candidate_decisions, (str, bytes)
                ):
                    raise TypeError("GPU decisions must be a sequence")
                if len(candidate_decisions) != shots:
                    raise ValueError(
                        f"GPU decoder returned {len(candidate_decisions)} decisions "
                        f"for {shots} shots"
                    )
                gpu_decisions = list(candidate_decisions)
                if not isinstance(candidate_stats, dict):
                    raise TypeError("GPU stats must be a dictionary")
                gpu_stats = candidate_stats
                raw_failed = candidate_stats.get("backend_failed_shots", ())
                try:
                    backend_failed_shots = {int(shot) for shot in raw_failed}
                except (TypeError, ValueError, OverflowError) as exc:
                    raise TypeError(
                        "resident runner backend_failed_shots must be integers"
                    ) from exc
                if any(shot < 0 or shot >= shots for shot in backend_failed_shots):
                    raise ValueError(
                        "resident runner reported an out-of-range failed shot"
                    )
                runner_error = candidate_stats.get("backend_error")
                if runner_error:
                    call_error = str(runner_error)
            except Exception as exc:
                call_error = _format_error(exc)
                gpu_decisions = None
                backend_failed_shots = set(range(shots))

        fallback: list[int] = []
        if gpu_decisions is None:
            fallback = list(range(shots))
            reason = call_error or "GPU backend is unavailable"
            for shot in fallback:
                routes[shot] = "gpu:backend-failure->cpu"
        else:
            reason = call_error or "GPU output requires exact replay"
            for shot, decision in enumerate(gpu_decisions):
                if shot in backend_failed_shots:
                    fallback.append(shot)
                    routes[shot] = "gpu:backend-failure->cpu"
                elif gpu_decision_is_accepted(decision):
                    final[shot] = decision
                    routes[shot] = f"gpu:status-{int(decision['status'])}"
                else:
                    fallback.append(shot)
                    routes[shot] = f"gpu:{_status_for_route(decision)}->cpu"

        if fallback and callback is None:
            raise ExactFallbackRequiredError(fallback, reason)
        cpu_calls: list[int] = []
        for shot in fallback:
            # Increasing source-shot order is deliberate.  A stateful exact
            # CPU decoder therefore observes the same batch order every time.
            decision = _validate_cpu_exact(shot, callback(shot))
            if final[shot] is not None:
                raise RuntimeError(f"shot {shot} has both GPU and CPU output")
            final[shot] = decision
            routes[shot] += "|cpu:replayed"
            cpu_calls.append(shot)

        unresolved = [shot for shot, decision in enumerate(final) if decision is None]
        if unresolved:
            raise RuntimeError(f"hybrid dispatcher left unresolved shots: {unresolved}")
        decisions = [decision for decision in final if decision is not None]
        for shot, decision in enumerate(decisions):
            try:
                status = int(decision.get("status", -1))
            except (AttributeError, TypeError, ValueError, OverflowError):
                status = -1
            if status == GPU_WORKSPACE_OVERFLOW:
                raise RuntimeError(f"forbidden status 3 escaped for shot {shot}")

        fallback_set = set(fallback)
        return decisions, {
            "backend": self.selection.backend,
            "selection_reason": self.selection.reason,
            "selection_overridden": self.selection.overridden,
            "features": asdict(self.selection.features),
            "workspace_plan": asdict(self.plan) if self.plan is not None else None,
            "resident": self.resident,
            "decode_call": self._decode_calls,
            "gpu_backend_error": call_error,
            "gpu_stats": gpu_stats,
            "routes": routes,
            "gpu_accepted_shots": [
                shot for shot in range(shots) if shot not in fallback_set
            ],
            "cpu_fallback_shots": fallback,
            "cpu_calls": cpu_calls,
        }


__all__ = [
    "BACKEND_DEFAULTS",
    "BACKEND_FINAL",
    "BACKEND_FUSED",
    "BACKEND_PREFETCH",
    "BACKEND_SIBLING",
    "BACKEND_SIBLING_BASE",
    "BackendSelection",
    "ExactFallbackRequiredError",
    "ExactGpuBackend",
    "ModelFeatures",
    "WorkspacePlan",
    "WORKSPACE_LAYOUT_BB_SINGLEHASH_COMPACT",
    "WORKSPACE_LAYOUT_LEGACY",
    "estimate_chunk_input_bytes",
    "estimate_graph_bytes",
    "estimate_workspace_bytes",
    "gpu_decision_is_accepted",
    "inspect_model",
    "load_default_decoder_class",
    "plan_workspace",
    "select_backend",
]
