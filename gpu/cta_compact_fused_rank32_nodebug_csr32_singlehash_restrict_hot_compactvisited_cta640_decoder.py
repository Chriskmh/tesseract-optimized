"""Exact BB candidate adding compact visited capacity to the composed winner.

CUDA source, CSR32 checks, single-hash equality guards, narrow restrict
contracts, CTA640 specialization, and the 50-pointer allocation audit are
inherited unchanged.  This module changes only the host-side visited capacity
from ``next_pow2(2W)`` to the proved ``next_pow2(W + 2)`` policy.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp

import cta_compact_fused_rank32_nodebug_csr32_compactvisited_decoder as compact
import cta_compact_fused_rank32_nodebug_csr32_singlehash_restrict_hot_cta640_decoder as base
from gpu_decoder import HostModel, compare_with_reference


CTA_THREADS = base.CTA_THREADS
HEAP_CACHE = base.HEAP_CACHE
CTA_KERNEL = base.CTA_KERNEL
COMMON = base.COMMON
VISITED_CAPACITY_POLICIES = compact.VISITED_CAPACITY_POLICIES
VisitedCapacityPlan = compact.VisitedCapacityPlan
plan_visited_capacity = compact.plan_visited_capacity

# One uint64 hash and one uint32 encoded chain per real visited slot.  The
# unused restricted hash2 ABI receives one separate uint64 dummy allocation.
VISITED_SLOT_BYTES = 8 + 4
VISITED_HASH2_PLACEHOLDER_BYTES = 8

if base._strip_restrict(CTA_KERNEL) != base.single_hash.CTA_KERNEL:  # pylint: disable=protected-access
    raise RuntimeError("compact composition lost restrict recovery proof")
if base._restore_dual_hash_kernel(CTA_KERNEL) != base.restrict_base.CTA_KERNEL:  # pylint: disable=protected-access
    raise RuntimeError("compact composition lost single-hash recovery proof")


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    """Compile the unchanged composed SM75 source without launching it."""
    return base.nvrtc_compile_check(monotonic=monotonic)


def runtime_kernel_attributes(*, monotonic: bool = True) -> dict[str, int | bool]:
    """Load unchanged composed code and inspect resources without a launch."""
    return base.runtime_kernel_attributes(monotonic=monotonic)


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Composed exact kernel with the proved compact visited-table policy."""

    kernel_source = CTA_KERNEL
    common_source = COMMON

    def __init__(
        self,
        model: HostModel,
        workspace_capacity: int = 8192,
        max_output: int = 1024,
        max_registers: int = 0,
        cta_threads: int = CTA_THREADS,
        heap_cache: int = HEAP_CACHE,
        *,
        visited_capacity_policy: str = "compact",
        visited_capacity: int | None = None,
    ):
        # Plan before the parent constructor reaches dynamic workspace
        # allocation.  The override below installs this capacity immediately
        # before allocating visited_hash1 and visited_chain.
        self.visited_capacity_plan = plan_visited_capacity(
            workspace_capacity,
            policy=visited_capacity_policy,
            explicit_capacity=visited_capacity,
        )
        super().__init__(
            model, workspace_capacity, max_output, max_registers,
            cta_threads, heap_cache,
        )
        if self.visited_capacity != self.visited_capacity_plan.capacity:
            raise AssertionError("workspace did not use planned visited capacity")

    def _allocate_workspace(self) -> dict[str, cp.ndarray]:
        # base.__init__ temporarily assigns the legacy capacity.  No visited
        # allocation occurs before this virtual call, so replacing it here is
        # both early enough and confined to the derived candidate.
        self.visited_capacity = self.visited_capacity_plan.capacity
        return super()._allocate_workspace()

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        plan = self.visited_capacity_plan
        shots = int(self.model.metadata["shots"])
        real_bytes_per_shot = plan.capacity * VISITED_SLOT_BYTES
        stats.update({
            "visited_capacity_policy": plan.policy,
            "visited_capacity": plan.capacity,
            "visited_maximum_inserted_states": plan.maximum_inserted_states,
            "visited_empty_slots_at_maximum": plan.empty_slots_at_maximum,
            "visited_slot_bytes": VISITED_SLOT_BYTES,
            "visited_bytes_per_shot": real_bytes_per_shot,
            "visited_bytes_total": (
                shots * real_bytes_per_shot + VISITED_HASH2_PLACEHOLDER_BYTES
            ),
            "visited_hash2_placeholder_bytes": VISITED_HASH2_PLACEHOLDER_BYTES,
        })
        return decisions, stats


def main() -> None:
    import argparse

    base.single_hash.base.rank32._pin_one_cpu()  # pylint: disable=protected-access
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--cta-threads", type=int, choices=(CTA_THREADS,),
                        default=CTA_THREADS)
    parser.add_argument("--heap-cache", type=int, choices=(HEAP_CACHE,),
                        default=HEAP_CACHE)
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

    model_load_start = time.perf_counter()
    model = HostModel.load(args.model)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000
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
        shots = int(model.metadata["shots"])
        real_bytes_per_shot = plan.capacity * VISITED_SLOT_BYTES
        payload = json.dumps({
            "model": str(args.model),
            "model_load_ms": model_load_ms,
            "workspace_capacity": plan.workspace_capacity,
            "visited_capacity_policy": plan.policy,
            "visited_capacity": plan.capacity,
            "maximum_inserted_states": plan.maximum_inserted_states,
            "empty_slots_at_maximum": plan.empty_slots_at_maximum,
            "visited_slot_bytes": VISITED_SLOT_BYTES,
            "visited_bytes_per_shot": real_bytes_per_shot,
            "visited_bytes_total": (
                shots * real_bytes_per_shot + VISITED_HASH2_PLACEHOLDER_BYTES
            ),
            "visited_hash2_placeholder_bytes": VISITED_HASH2_PLACEHOLDER_BYTES,
            "initialization_ms": decoder.restrict_initialization_ms,
            "alias_audit": decoder.restrict_alias_audit,
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
        "model_load_ms": model_load_ms,
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
