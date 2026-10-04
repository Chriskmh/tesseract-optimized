"""Exact BB candidate with audited CUDA ``__restrict__`` contracts.

This module changes pointer alias information only.  It starts from the
nodebug stop-prefetch winner, qualifies pointer parameters whose live host
allocations and device call sites are disjoint, and checks those allocations
before the first decode.  Search order, integer operations, binary64 folds,
kernel argument order, and argument element types are unchanged.

The source transform is deliberately independent of CSR offset width and CTA
width.  Applying it to the checked CSR32 source or the no-CSR wide-CTA source
therefore produces the same alias contracts without coupling the experiments.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp
import numpy as np

import cta_compact_fused_rank32_decoder as rank32
import cta_compact_fused_rank32_nodebug_decoder as base
from gpu_decoder import HostModel, compare_with_reference


# Every entry is an audited function scope followed by the pointer parameters
# that are non-overlapping for every call made by this kernel.  Keeping this an
# explicit allowlist prevents a later pointer parameter from silently acquiring
# a stronger promise merely because it was added to a signature.
CTA_RESTRICT_PARAMETERS: dict[str, tuple[str, ...]] = {
    "update_blocker_count": ("error_state", "shared_bad"),
    "set_prefix_position": ("error_state",),
    "clear_prefix_position": ("error_state",),
    "cta_compact_load": (
        "cost", "num_dets", "chain",
        "shared_cost", "shared_num_dets", "shared_chain",
    ),
    "cta_compact_store": (
        "cost", "num_dets", "chain",
        "shared_cost", "shared_num_dets", "shared_chain",
    ),
    "cta_compact_push": (
        "cost", "num_dets", "chain",
        "shared_cost", "shared_num_dets", "shared_chain",
    ),
    "cta_compact_pop": (
        "cost", "num_dets", "chain",
        "shared_cost", "shared_num_dets", "shared_chain",
    ),
    "update_count_row_cta": (
        "error_state", "d2e_offsets", "d2e", "shared_bad",
    ),
    "update_both_count_rows_cta": (
        "error_state", "d2e_offsets", "d2e", "shared_bad",
    ),
    "adjust_child_counts_cta": (
        "parent_bits", "error_state", "edet_offsets", "edets",
        "d2e_offsets", "d2e", "shared_bad",
    ),
    "transition_child_counts_cta": (
        "parent_bits", "error_state", "edet_offsets", "edets",
        "d2e_offsets", "d2e", "shared_bad",
    ),
    "detector_cost_cta": (
        "error_state", "d2e_offsets", "d2e_ranked", "count_ranks16",
        "count_values", "warp_keys", "warp_types", "shared_best_rank",
        "shared_best_type", "shared_stop",
    ),
    "detector_cost_subwarp": (
        "error_state", "d2e_offsets", "d2e_ranked", "count_ranks16",
        "count_values",
    ),
    "transition_parent_counts_cta": (
        "state_bits", "counted_bits", "error_state", "d2e_offsets", "d2e",
        "shared_detector", "shared_delta", "shared_bad",
    ),
    "apply_chain_edge_cta": (
        "bits", "error_state", "chain_error", "chain_min_detector",
        "chain_row", "d2e_offsets", "d2e", "edet_offsets", "edets",
        "shared_bad",
    ),
    "state_equals_virtual_child_cta": (
        "parent_bits", "initial_bits", "scratch", "chain_error",
        "chain_parent", "edet_offsets", "edets",
    ),
    "visited_find_virtual_child_cta": (
        "parent_bits", "visited_hash1", "visited_hash2", "visited_chain",
        "initial_bits", "scratch", "chain_error", "chain_parent",
        "edet_offsets", "edets",
    ),
    "decode_exact_cta": (
        "d2e_offsets", "d2e", "edet_offsets", "edets",
        "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
        "weights", "error_to_dem", "detector_order", "error_hash1",
        "error_hash2", "count_values", "d2e_ranked", "count_ranks16",
        "shot_offsets", "shot_detections", "initial_shot_bits", "shot_hash1",
        "shot_hash2", "heap_cost_all", "heap_num_dets_all", "heap_depth_all",
        "heap_chain_all", "heap_hash1_all", "heap_hash2_all",
        "chain_error_all", "chain_min_detector_all", "chain_parent_all",
        "chain_row_all", "chain_depth_all", "path_stack_all",
        "error_state_all", "counted_bits_all", "parent_cost_cache_all",
        "parent_cost_stamp_all", "state_bits_all", "child_bits_all",
        "compare_bits_all", "visited_hash1_all", "visited_hash2_all",
        "visited_chain_all", "out_status", "out_length", "out_errors",
        "out_cost_bits", "out_pushes", "out_pops", "out_cycles", "out_debug",
    ),
}

COMMON_RESTRICT_PARAMETERS: dict[str, tuple[str, ...]] = {
    "state_equals_chain": (
        "candidate", "initial_bits", "scratch", "chain_error", "chain_parent",
        "edet_offsets", "edets",
    ),
    "visited_find": (
        "candidate", "visited_hash1", "visited_hash2", "visited_chain",
        "initial_bits", "scratch", "chain_error", "chain_parent",
        "edet_offsets", "edets",
    ),
}

EXPECTED_DEFINITIONS = {"detector_cost_subwarp": 2}


def _function_pattern(name: str) -> re.Pattern[str]:
    return re.compile(
        r"(?P<head>(?:extern \"C\" )?(?:__device__|__global__)"
        r"[^{};]*?\b" + re.escape(name) + r"\s*\()"
        r"(?P<params>[^)]*)(?P<tail>\)\s*\{)",
        re.DOTALL,
    )


def _qualify_signature_parameters(
    source: str,
    contracts: Mapping[str, tuple[str, ...]],
) -> str:
    """Add restrict to an explicit set of pointer parameters only."""
    transformed = source
    for function, names in contracts.items():
        pattern = _function_pattern(function)
        matches = list(pattern.finditer(transformed))
        expected = EXPECTED_DEFINITIONS.get(function, 1)
        if len(matches) != expected:
            raise RuntimeError(
                f"restrict audit expected {expected} definitions of {function}, "
                f"found {len(matches)}"
            )

        def qualify(match: re.Match[str]) -> str:
            params = match.group("params")
            for name in names:
                declaration = re.compile(
                    r"(?P<star>\*\s*)(?!__restrict__\s+)"
                    + re.escape(name)
                    + r"\b"
                )
                params, count = declaration.subn(
                    lambda item: item.group("star") + "__restrict__ " + name,
                    params,
                    count=1,
                )
                if count != 1:
                    raise RuntimeError(
                        f"{function}.{name} is missing, duplicated, or not a pointer"
                    )
            return match.group("head") + params + match.group("tail")

        # Re-run substitution over the complete source so both conditional
        # detector-cost definitions receive an identical contract.
        transformed = pattern.sub(qualify, transformed)
    return transformed


def _build_restrict_kernel(source: str) -> str:
    transformed = _qualify_signature_parameters(source, CTA_RESTRICT_PARAMETERS)
    expected = sum(
        len(names) * EXPECTED_DEFINITIONS.get(function, 1)
        for function, names in CTA_RESTRICT_PARAMETERS.items()
    )
    if transformed.count("__restrict__") - source.count("__restrict__") != expected:
        raise RuntimeError("unexpected CTA restrict qualifier count")
    return transformed


def _build_restrict_common(source: str) -> str:
    transformed = _qualify_signature_parameters(source, COMMON_RESTRICT_PARAMETERS)
    expected = sum(len(names) for names in COMMON_RESTRICT_PARAMETERS.values())
    if transformed.count("__restrict__") - source.count("__restrict__") != expected:
        raise RuntimeError("unexpected common restrict qualifier count")
    return transformed


CTA_KERNEL = _build_restrict_kernel(base.CTA_KERNEL)
COMMON = _build_restrict_common(base.COMMON)

ALLOWED_CTA_THREADS = (32, 64, 128, 256, 512, 544, 576, 608, 640)
ALLOWED_HEAP_CACHE = (128, 256, 512, 1024, 1536, 2048)


# This mapping is the host-side proof boundary for the kernel-level contracts.
# Each pointer kernel argument must name a live, disjoint CuPy allocation.
DEVICE_POINTER_BINDINGS = {
    "d2e_offsets": "d2e_offsets",
    "d2e": "d2e",
    "edet_offsets": "edet_offsets",
    "edets": "edets",
    "eneighbor_word_offsets": "eneighbor_word_offsets",
    "eneighbor_words": "eneighbor_words",
    "eneighbor_masks": "eneighbor_masks",
    "weights": "weights",
    "error_to_dem": "error_to_dem",
    "detector_order": "detector_order",
    "error_hash1": "error_hash1",
    "error_hash2": "error_hash2",
    "count_values": "count_values",
    "d2e_ranked": "d2e_ranked32",
    "count_ranks16": "count_ranks16",
    "shot_offsets": "shot_offsets",
    "shot_detections": "shot_detections",
    "initial_shot_bits": "shot_bits",
    "shot_hash1": "shot_hash1",
    "shot_hash2": "shot_hash2",
}

WORKSPACE_POINTER_BINDINGS = {
    name: name[:-4] if name.endswith("_all") else name
    for name in CTA_RESTRICT_PARAMETERS["decode_exact_cta"]
    if name not in DEVICE_POINTER_BINDINGS
}


@dataclass(frozen=True)
class AllocationSpan:
    name: str
    begin: int
    end: int


def _array_span(name: str, array: Any) -> AllocationSpan | None:
    nbytes = int(array.nbytes)
    if nbytes == 0:
        return None
    begin = int(array.data.ptr)
    if begin == 0:
        raise ValueError(f"non-empty kernel buffer {name} has a null pointer")
    return AllocationSpan(name=name, begin=begin, end=begin + nbytes)


def assert_nonoverlapping_allocations(
    buffers: Mapping[str, Any],
) -> tuple[AllocationSpan, ...]:
    """Prove that all non-empty byte ranges are pairwise disjoint."""
    spans = tuple(
        sorted(
            (
                span
                for name, array in buffers.items()
                if (span := _array_span(name, array)) is not None
            ),
            key=lambda span: (span.begin, span.end, span.name),
        )
    )
    for left, right in zip(spans, spans[1:]):
        if right.begin < left.end:
            raise ValueError(
                "__restrict__ allocation overlap: "
                f"{left.name}=[0x{left.begin:x},0x{left.end:x}) and "
                f"{right.name}=[0x{right.begin:x},0x{right.end:x})"
            )
    return spans


def audit_kernel_allocations(
    device: Mapping[str, Any], workspace: Mapping[str, Any]
) -> dict[str, int]:
    """Validate the host allocation promise behind ``decode_exact_cta``."""
    buffers: dict[str, Any] = {}
    for parameter, name in DEVICE_POINTER_BINDINGS.items():
        buffers[parameter] = device[name]
    for parameter, name in WORKSPACE_POINTER_BINDINGS.items():
        buffers[parameter] = workspace[name]
    expected = set(CTA_RESTRICT_PARAMETERS["decode_exact_cta"])
    if set(buffers) != expected:
        missing = sorted(expected - set(buffers))
        extra = sorted(set(buffers) - expected)
        raise RuntimeError(
            f"kernel alias audit binding mismatch: missing={missing}, extra={extra}"
        )
    spans = assert_nonoverlapping_allocations(buffers)
    return {
        "pointer_parameters": len(buffers),
        "nonempty_allocations": len(spans),
        "empty_allocations": len(buffers) - len(spans),
    }


def _compile_options(
    *, cta_threads: int, heap_cache: int, monotonic: bool
) -> tuple[str, ...]:
    if cta_threads not in ALLOWED_CTA_THREADS:
        raise ValueError(f"unsupported CTA width {cta_threads}")
    if heap_cache not in ALLOWED_HEAP_CACHE:
        raise ValueError(f"unsupported heap cache {heap_cache}")
    return (
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
        f"-DTESSERACT_CTA_THREADS={cta_threads}",
        f"-DTESSERACT_CTA_HEAP_CACHE={heap_cache}",
        f"-DTESSERACT_MONOTONIC_RANK_ROWS={1 if monotonic else 0}",
        "-DTESSERACT_RANK32_ERROR_MASK=0x0003ffffU",
        "-DTESSERACT_RANK32_GROUP_MASK=0x0000001fU",
        "-DTESSERACT_RANK32_LOWER_MASK=0x0000007fU",
        "-DTESSERACT_RANK32_GROUP_SHIFT=18",
        "-DTESSERACT_RANK32_LOWER_SHIFT=23",
    )


def nvrtc_compile_check(
    *, monotonic: bool = True, cta_threads: int = 512, heap_cache: int = 1024
) -> object:
    """Compile complete SM75 PTX without launching a decoder."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=_compile_options(
            cta_threads=cta_threads, heap_cache=heap_cache, monotonic=monotonic
        ),
        arch="75",
        filename="cta_compact_fused_rank32_restrict_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(
    *, cta_threads: int = 512, heap_cache: int = 1024,
    monotonic: bool = True,
) -> dict[str, int | bool]:
    """Load the compiled kernel and inspect resources without a launch."""
    module = cp.RawModule(
        code=COMMON + CTA_KERNEL,
        options=_compile_options(
            cta_threads=cta_threads, heap_cache=heap_cache, monotonic=monotonic
        ),
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Thread-local source selection plus a mandatory allocation audit."""

    kernel_source = CTA_KERNEL
    common_source = COMMON

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        constructor_start = time.perf_counter()
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.heap_cache = int(heap_cache)
        self.visited_capacity = rank32._next_power_of_two(  # pylint: disable=protected-access
            2 * self.workspace_capacity
        )
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
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

        options = list(_compile_options(
            cta_threads=self.cta_threads,
            heap_cache=self.heap_cache,
            monotonic=bool(model.rank_rows_monotonic),
        ))
        # The layout flags above use the compile-only fixture widths.  Runtime
        # exports can choose another valid rank32 layout, so replace them here.
        options = [option for option in options if not option.startswith(
            ("-DTESSERACT_RANK32_ERROR_MASK=", "-DTESSERACT_RANK32_GROUP_MASK=",
             "-DTESSERACT_RANK32_LOWER_MASK=", "-DTESSERACT_RANK32_GROUP_SHIFT=",
             "-DTESSERACT_RANK32_LOWER_SHIFT="))]
        layout = self.rank32_layout
        options.extend((
            f"-DTESSERACT_RANK32_ERROR_MASK=0x{layout.error_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_MASK=0x{layout.group_mask:08x}U",
            f"-DTESSERACT_RANK32_LOWER_MASK=0x{layout.lower_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_SHIFT={layout.group_shift}",
            f"-DTESSERACT_RANK32_LOWER_SHIFT={layout.lower_shift}",
        ))
        if max_registers:
            options.append(f"--maxrregcount={int(max_registers)}")
        compile_start = time.perf_counter()
        self.module = cp.RawModule(
            code=self.common_source + self.kernel_source,
            options=tuple(options),
            name_expressions=("decode_exact_cta",),
            backend="nvrtc",
        )
        self.kernel = self.module.get_function("decode_exact_cta")
        compile_ms = (time.perf_counter() - compile_start) * 1000

        upload_start = time.perf_counter()
        self.device = {}
        for name in (
            "d2e_offsets", "d2e", "edet_offsets", "edets",
            "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
            "weights", "error_to_dem", "detector_order", "error_hash1",
            "error_hash2", "count_values", "count_ranks16", "shot_offsets",
            "shot_detections", "shot_bits", "shot_hash1", "shot_hash2",
        ):
            self.device[name] = cp.asarray(getattr(model, name))
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        upload_ms = (time.perf_counter() - upload_start) * 1000

        allocation_start = time.perf_counter()
        self.workspace = self._allocate_workspace()
        allocation_ms = (time.perf_counter() - allocation_start) * 1000

        audit_start = time.perf_counter()
        self.restrict_alias_audit = audit_kernel_allocations(
            self.device, self.workspace
        )
        audit_ms = (time.perf_counter() - audit_start) * 1000
        self.restrict_initialization_ms = {
            "nvrtc_and_load": compile_ms,
            "device_upload": upload_ms,
            "workspace_allocation": allocation_ms,
            "alias_audit": audit_ms,
            "constructor_total": (time.perf_counter() - constructor_start) * 1000,
        }

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["restrict_qualified"] = True
        stats["restrict_alias_audit"] = dict(self.restrict_alias_audit)
        stats["restrict_initialization_ms"] = dict(self.restrict_initialization_ms)
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
        "--cta-threads", type=int, choices=ALLOWED_CTA_THREADS, default=512)
    parser.add_argument(
        "--heap-cache", type=int, choices=ALLOWED_HEAP_CACHE, default=1024)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model)
    decoder = ExactCtaDecoder(
        model, args.workspace, args.max_output, args.max_registers,
        args.cta_threads, args.heap_cache,
    )
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
