"""Isolated Color candidate with audited hot ``__restrict__`` pointers.

The accepted checked-CSR32, prefix-fold, no-telemetry Color source is changed
only by adding qualifiers to the full kernel pointer ABI and to its two
detector-query definitions.  The candidate retains the fixed 256-thread,
two-block launch target and the 128-entry shared heap.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp

import cta_compact_final_nodebug_decoder as base
import cta_production_restrict_support as restrict_support
from gpu_decoder import HostModel, compare_with_reference


CTA_THREADS = base.CTA_THREADS
HEAP_CACHE = 128
MIN_BLOCKS_PER_SM = base.MIN_BLOCKS_PER_SM

KERNEL_POINTERS = (
    "d2e_offsets", "d2e", "edet_offsets", "edets",
    "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
    "weights", "error_to_dem", "detector_order", "error_hash1",
    "error_hash2", "count_values", "d2e_ranked", "count_ranks16",
    "shot_offsets", "shot_detections", "initial_shot_bits", "shot_hash1",
    "shot_hash2", "heap_cost_all", "heap_num_dets_all", "heap_depth_all",
    "heap_chain_all", "heap_hash1_all", "heap_hash2_all",
    "chain_error_all", "chain_min_detector_all", "chain_parent_all",
    "chain_row_all", "chain_depth_all", "path_stack_all", "error_state_all",
    "counted_bits_all", "parent_cost_cache_all", "parent_cost_stamp_all",
    "state_bits_all", "child_bits_all", "compare_bits_all",
    "visited_hash1_all", "visited_hash2_all", "visited_chain_all",
    "out_status", "out_length", "out_errors", "out_cost_bits",
    "out_pushes", "out_pops", "out_cycles", "out_debug",
)

HOT_RESTRICT_PARAMETERS: dict[str, tuple[str, ...]] = {
    "detector_cost_cta": (
        "error_state", "d2e_offsets", "d2e_ranked", "count_ranks16",
        "count_values", "warp_keys", "warp_types", "shared_best_rank",
        "shared_best_type", "shared_stop",
    ),
    "detector_cost_subwarp": (
        "error_state", "d2e_offsets", "d2e_ranked", "count_ranks16",
        "count_values",
    ),
    "decode_exact_cta": KERNEL_POINTERS,
}


def _build_hot_restrict_kernel(source: str) -> str:
    return restrict_support.qualify_signature_parameters(
        source, HOT_RESTRICT_PARAMETERS
    )


CTA_KERNEL = _build_hot_restrict_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


def nvrtc_compile_check(*, monotonic: bool = False) -> object:
    """Compile the complete SM75 translation unit without launching it."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=base._compile_options(monotonic=monotonic),  # pylint: disable=protected-access
        arch="75",
        filename="cta_compact_final_nodebug_restrict_hot_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(*, monotonic: bool = False) -> dict[str, int | bool]:
    """Load the candidate and return resource attributes without a launch."""
    module = cp.RawModule(
        code=COMMON + CTA_KERNEL,
        options=base._compile_options(monotonic=monotonic),  # pylint: disable=protected-access
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Immutable candidate source with a mandatory live-allocation audit."""

    common_source = COMMON
    kernel_source = CTA_KERNEL

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = CTA_THREADS, heap_cache: int = HEAP_CACHE):
        constructor_start = time.perf_counter()
        if int(heap_cache) != HEAP_CACHE:
            raise ValueError(
                f"Color hot-restrict candidate requires heap_cache={HEAP_CACHE}"
            )
        super().__init__(
            model, workspace_capacity, max_output, max_registers,
            cta_threads, heap_cache,
        )
        audit_start = time.perf_counter()
        self.restrict_alias_audit = restrict_support.audit_kernel_allocations(
            self.device, self.workspace, KERNEL_POINTERS
        )
        audit_ms = (time.perf_counter() - audit_start) * 1000
        self.restrict_initialization_ms = {
            "alias_audit": audit_ms,
            "constructor_total": (time.perf_counter() - constructor_start) * 1000,
        }

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["restrict_profile"] = "kernel_and_detector_cost"
        stats["restrict_alias_audit"] = dict(self.restrict_alias_audit)
        return decisions, stats


def main() -> None:
    base.base.base._pin_one_cpu()  # pylint: disable=protected-access
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--cta-threads", type=int, choices=(CTA_THREADS,),
                        default=CTA_THREADS)
    parser.add_argument("--heap-cache", type=int, choices=(HEAP_CACHE,),
                        default=HEAP_CACHE)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--construct-only", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model_load_start = time.perf_counter()
    model = HostModel.load(args.model)
    model_load_ms = (time.perf_counter() - model_load_start) * 1000
    decoder = ExactCtaDecoder(
        model, args.workspace, args.max_output, args.max_registers,
        args.cta_threads, args.heap_cache,
    )
    if args.construct_only:
        payload = json.dumps({
            "model": str(args.model),
            "model_load_ms": model_load_ms,
            "initialization_ms": decoder.restrict_initialization_ms,
            "alias_audit": decoder.restrict_alias_audit,
            "csr32_max_offsets": {
                name: int(values.max(initial=0))
                for name, values in decoder.csr32_offsets.as_dict().items()
            },
            "kernel_attributes": dict(decoder.kernel.attributes),
            "decoder_launched": False,
        }, indent=2)
        print(payload)
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(payload + "\n", encoding="utf-8")
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
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
