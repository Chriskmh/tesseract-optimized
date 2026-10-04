"""Exact BB candidate with a provably smaller visited hash table.

The CUDA translation unit is byte-for-byte the accepted checked-CSR32,
no-debug source.  This candidate changes only the host allocation policy for
the open-addressed visited table.  The compact policy keeps one empty slot
after the largest state set the bounded workspace can represent; the kernel's
existing exhaustive-probe failure remains an exact status-3 fallback.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp
import numpy as np

import cta_compact_fused_rank32_decoder as rank32
import cta_compact_fused_rank32_nodebug_csr32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


CTA_KERNEL = base.CTA_KERNEL
COMMON = base.COMMON
ALLOWED_CTA_THREADS = base.ALLOWED_CTA_THREADS
ALLOWED_HEAP_CACHE = base.ALLOWED_HEAP_CACHE
VISITED_CAPACITY_POLICIES = ("compact", "legacy")
INT32_MAX = int(np.iinfo(np.int32).max)


@dataclass(frozen=True)
class VisitedCapacityPlan:
    """Host proof and allocation result for one per-shot visited table."""

    policy: str
    workspace_capacity: int
    maximum_inserted_states: int
    minimum_safe_capacity: int
    capacity: int
    empty_slots_at_maximum: int

    @property
    def bytes_per_shot(self) -> int:
        # Two uint64 hashes and one uint32 encoded chain per slot.
        return self.capacity * (8 + 8 + 4)


def plan_visited_capacity(
    workspace_capacity: int,
    *,
    policy: str = "compact",
    explicit_capacity: int | None = None,
) -> VisitedCapacityPlan:
    """Return a power-of-two capacity that cannot fill before fallback.

    At most ``workspace_capacity`` non-root chains can be allocated.  Each
    inserted non-root state owns one such chain, and the root contributes at
    most one more entry.  Therefore the table can contain at most ``W + 1``
    entries.  Open addressing needs one additional empty slot so that every
    absent-key probe terminates without relying on full-table exhaustion.

    The seemingly natural ``next_pow2(W + 1)`` is unsafe when ``W + 1`` is
    already a power of two: it permits the table to become completely full.
    ``compact`` consequently rounds ``W + 2``.  For the production BB
    workspace W=100000 this is still 131072, versus 262144 for ``legacy``.
    """
    workspace = int(workspace_capacity)
    if workspace < 2:
        raise ValueError("workspace_capacity must be at least 2")
    if policy not in VISITED_CAPACITY_POLICIES:
        raise ValueError(
            f"visited capacity policy must be one of {VISITED_CAPACITY_POLICIES}"
        )

    maximum_inserted = workspace + 1  # root plus at most W chain states
    minimum_safe = maximum_inserted + 1  # retain one empty open-address slot
    if explicit_capacity is None:
        target = minimum_safe if policy == "compact" else 2 * workspace
        capacity = rank32._next_power_of_two(target)  # pylint: disable=protected-access
        selected_policy = policy
    else:
        capacity = int(explicit_capacity)
        selected_policy = "explicit"

    if capacity <= 0 or capacity & (capacity - 1):
        raise ValueError("visited_capacity must be a positive power of two")
    if capacity < minimum_safe:
        raise ValueError(
            "visited_capacity must exceed the maximum W+1 inserted states "
            f"and retain one empty slot; need at least {minimum_safe}, got {capacity}"
        )
    # The kernel ABI takes a signed int and uses table_capacity - 1.
    if capacity > INT32_MAX:
        raise ValueError("visited_capacity must fit a positive signed int32")

    return VisitedCapacityPlan(
        policy=selected_policy,
        workspace_capacity=workspace,
        maximum_inserted_states=maximum_inserted,
        minimum_safe_capacity=minimum_safe,
        capacity=capacity,
        empty_slots_at_maximum=capacity - maximum_inserted,
    )


def nvrtc_compile_check(
    *, monotonic: bool = True, cta_threads: int = 640, heap_cache: int = 1024
) -> object:
    """Compile the unchanged exact translation unit without launching it."""
    return base.nvrtc_compile_check(
        monotonic=monotonic, cta_threads=cta_threads, heap_cache=heap_cache
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Accepted CSR32 kernel with a configurable, proved visited capacity."""

    kernel_source = CTA_KERNEL
    common_source = COMMON

    def __init__(
        self,
        model: HostModel,
        workspace_capacity: int = 8192,
        max_output: int = 1024,
        max_registers: int = 0,
        cta_threads: int = 128,
        heap_cache: int = 128,
        *,
        visited_capacity_policy: str = "compact",
        visited_capacity: int | None = None,
    ):
        self.visited_capacity_plan = plan_visited_capacity(
            workspace_capacity,
            policy=visited_capacity_policy,
            explicit_capacity=visited_capacity,
        )
        self.csr32_offsets = base.csr32.pack_model_csr32_offsets(model)
        self.model = model
        self.workspace_capacity = self.visited_capacity_plan.workspace_capacity
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.heap_cache = int(heap_cache)
        self.visited_capacity = self.visited_capacity_plan.capacity
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("packed state requires max_output < 65535")
        if self.cta_threads not in ALLOWED_CTA_THREADS:
            raise ValueError(f"cta_threads must be one of {ALLOWED_CTA_THREADS}")
        if self.heap_cache not in ALLOWED_HEAP_CACHE:
            raise ValueError(f"heap_cache must be one of {ALLOWED_HEAP_CACHE}")
        self.max_error_degree, self.max_row_length = (
            rank32._validate_fused_state_model(model)  # pylint: disable=protected-access
        )
        if not np.array_equal(
            model.detector_order,
            np.arange(len(model.detector_order), dtype=np.uint32),
        ):
            raise ValueError("CTA prototype currently requires identity detector order")
        self.d2e_ranked32, self.rank32_layout = rank32.repack_d2e_ranked32(model)

        options = [
            "--std=c++11", "--fmad=false", "--prec-div=true",
            "--prec-sqrt=true", "--ftz=false",
        ]
        if max_registers:
            options.append(f"--maxrregcount={int(max_registers)}")
        options.extend((
            f"-DTESSERACT_CTA_THREADS={self.cta_threads}",
            f"-DTESSERACT_CTA_HEAP_CACHE={self.heap_cache}",
            "-DTESSERACT_MONOTONIC_RANK_ROWS="
            + ("1" if model.rank_rows_monotonic else "0"),
        ))
        layout = self.rank32_layout
        options.extend((
            f"-DTESSERACT_RANK32_ERROR_MASK=0x{layout.error_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_MASK=0x{layout.group_mask:08x}U",
            f"-DTESSERACT_RANK32_LOWER_MASK=0x{layout.lower_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_SHIFT={layout.group_shift}",
            f"-DTESSERACT_RANK32_LOWER_SHIFT={layout.lower_shift}",
        ))
        self.module = cp.RawModule(
            code=self.common_source + self.kernel_source,
            options=tuple(options),
            name_expressions=("decode_exact_cta",),
            backend="nvrtc",
        )
        self.kernel = self.module.get_function("decode_exact_cta")

        packed_offsets = self.csr32_offsets.as_dict()
        self.device = {}
        for name in (
            "d2e_offsets", "d2e", "edet_offsets", "edets",
            "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
            "weights", "error_to_dem", "detector_order", "error_hash1",
            "error_hash2", "count_values", "count_ranks16", "shot_offsets",
            "shot_detections", "shot_bits", "shot_hash1", "shot_hash2",
        ):
            source = packed_offsets.get(name, getattr(model, name))
            self.device[name] = cp.asarray(source)
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        self.workspace = self._allocate_workspace()

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        plan = self.visited_capacity_plan
        stats.update({
            "csr32_offsets": True,
            "visited_capacity_policy": plan.policy,
            "visited_capacity": plan.capacity,
            "visited_maximum_inserted_states": plan.maximum_inserted_states,
            "visited_empty_slots_at_maximum": plan.empty_slots_at_maximum,
            "visited_bytes_per_shot": plan.bytes_per_shot,
        })
        return decisions, stats


def main() -> None:
    rank32._pin_one_cpu()  # pylint: disable=protected-access
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument(
        "--cta-threads", type=int, choices=ALLOWED_CTA_THREADS, default=128
    )
    parser.add_argument(
        "--heap-cache", type=int, choices=ALLOWED_HEAP_CACHE, default=128
    )
    parser.add_argument(
        "--visited-capacity-policy",
        choices=VISITED_CAPACITY_POLICIES,
        default="compact",
    )
    parser.add_argument("--visited-capacity", type=int)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model)
    decoder = ExactCtaDecoder(
        model,
        args.workspace,
        args.max_output,
        args.max_registers,
        args.cta_threads,
        args.heap_cache,
        visited_capacity_policy=args.visited_capacity_policy,
        visited_capacity=args.visited_capacity,
    )
    plan = decoder.visited_capacity_plan
    if args.construct_only:
        payload = json.dumps({
            "model": str(args.model),
            "workspace_capacity": plan.workspace_capacity,
            "visited_capacity_policy": plan.policy,
            "visited_capacity": plan.capacity,
            "maximum_inserted_states": plan.maximum_inserted_states,
            "empty_slots_at_maximum": plan.empty_slots_at_maximum,
            "visited_bytes_per_shot": plan.bytes_per_shot,
            "kernel_attributes": dict(decoder.kernel.attributes),
            "decoder_launched": False,
        }, indent=2)
        print(payload)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload + "\n")
        return

    runs = []
    decisions = None
    for _ in range(args.repeats):
        wall_start = time.perf_counter()
        decisions, stats = decoder.decode()
        stats["wall_ms"] = (time.perf_counter() - wall_start) * 1000
        runs.append(stats)
    assert decisions is not None
    mismatches = compare_with_reference(
        model, decisions, skip_workspace_overflow=args.allow_overflow
    )
    report = {
        "model": str(args.model), "metadata": model.metadata, "runs": runs,
        "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
    }
    payload = json.dumps(report, indent=2)
    print(payload)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
