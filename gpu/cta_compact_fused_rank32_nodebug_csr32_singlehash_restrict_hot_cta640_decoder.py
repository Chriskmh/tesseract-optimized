"""Exact BB composition: CSR32 single hash + narrow restrict + CTA640.

This immutable candidate composes the independently guarded single-hash
visited filter with the narrow restrict-hot transform.  The launch shape is
fixed to the accepted BB configuration: 640 threads and a 1024-entry shared
heap.  The four unused hash2 ABI placeholders receive distinct one-element
buffers, preserving every top-level ``__restrict__`` promise without restoring
the removed per-shot hash2 storage.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp
import numpy as np

import cta_compact_fused_rank32_nodebug_csr32_singlehash_decoder as single_hash
import cta_compact_fused_rank32_restrict_decoder as audited
import cta_compact_fused_rank32_restrict_hot_csr32_cta640_decoder as restrict_base
import cta_compact_fused_rank32_restrict_hot_decoder as hot
from gpu_decoder import HostModel, compare_with_reference


CTA_THREADS = 640
HEAP_CACHE = 1024

_UNUSED_RENAMES = {
    "error_hash2": "unused_error_hash2",
    "shot_hash2": "unused_shot_hash2",
    "heap_hash2_all": "unused_heap_hash2_all",
    "visited_hash2_all": "unused_visited_hash2_all",
}

HOT_RESTRICT_PARAMETERS = {
    "detector_cost_cta": hot.HOT_RESTRICT_PARAMETERS["detector_cost_cta"],
    "detector_cost_subwarp": hot.HOT_RESTRICT_PARAMETERS["detector_cost_subwarp"],
    "decode_exact_cta": tuple(
        _UNUSED_RENAMES.get(name, name)
        for name in hot.HOT_RESTRICT_PARAMETERS["decode_exact_cta"]
    ),
}


def _build_composed_kernel(source: str) -> str:
    """Apply exactly the accepted narrow allowlist to renamed single-hash ABI."""
    transformed = audited._qualify_signature_parameters(  # pylint: disable=protected-access
        source, HOT_RESTRICT_PARAMETERS
    )
    expected = sum(
        len(names) * audited.EXPECTED_DEFINITIONS.get(function, 1)
        for function, names in HOT_RESTRICT_PARAMETERS.items()
    )
    if transformed.count("__restrict__") - source.count("__restrict__") != expected:
        raise RuntimeError("unexpected composed hot restrict qualifier count")
    return transformed


def _strip_restrict(source: str) -> str:
    return source.replace("__restrict__ ", "")


def _restore_dual_hash_kernel(source: str) -> str:
    """Exact inverse of the single-hash edits, retaining restrict qualifiers.

    This exists as an executable composition proof.  It is not used to build
    or select runtime source.
    """
    replace = single_hash._replace_exact  # pylint: disable=protected-access
    transformed = replace(
        source,
        single_hash.SINGLE_HASH_FIND + single_hash.SINGLE_VIRTUAL_FIND,
        single_hash.DUAL_VIRTUAL_FIND,
    )
    for old, new in _UNUSED_RENAMES.items():
        transformed = replace(transformed, new, old)
    replacements = (
        (
            "  u64 *chain_hash1 = heap_hash1_all + (u64)shot * workspace_capacity;\n",
            "  u64 *chain_hash1 = heap_hash1_all + (u64)shot * workspace_capacity;\n"
            "  u64 *chain_hash2 = heap_hash2_all + (u64)shot * workspace_capacity;\n",
            1,
        ),
        (
            "  u64 *visited_hash1 = visited_hash1_all + (u64)shot * visited_capacity;\n",
            "  u64 *visited_hash1 = visited_hash1_all + (u64)shot * visited_capacity;\n"
            "  u64 *visited_hash2 = visited_hash2_all + (u64)shot * visited_capacity;\n",
            1,
        ),
        (
            "        s_node.hash1 = shot_hash1[shot];\n",
            "        s_node.hash1 = shot_hash1[shot];\n"
            "        s_node.hash2 = shot_hash2[shot];\n",
            1,
        ),
        (
            "        s_node.hash1 = chain_hash1[compact.chain];\n",
            "        s_node.hash1 = chain_hash1[compact.chain];\n"
            "        s_node.hash2 = chain_hash2[compact.chain];\n",
            1,
        ),
        (
            "    const NodeValue node{s_node.cost, s_node.num_dets, s_node.depth,\n"
            "                         s_node.chain, s_node.hash1, 0ULL};\n",
            "    const NodeValue node = s_node;\n",
            1,
        ),
        (
            "      s_visited = visited_find_single_hash_cta(\n"
            "          node.hash1, state_bits, s_empty_slot,\n"
            "          visited_hash1, visited_chain, visited_capacity,",
            "      s_visited = visited_find(\n"
            "          node.hash1, node.hash2, state_bits, s_empty_slot,\n"
            "          visited_hash1, visited_hash2, visited_chain, visited_capacity,",
            1,
        ),
        (
            "      visited_hash1[s_empty_slot] = node.hash1;\n",
            "      visited_hash1[s_empty_slot] = node.hash1;\n"
            "      visited_hash2[s_empty_slot] = node.hash2;\n",
            1,
        ),
        (
            "        const u64 child_hash1 = node.hash1 ^ error_hash1[error];\n",
            "        const u64 child_hash1 = node.hash1 ^ error_hash1[error];\n"
            "        const u64 child_hash2 = node.hash2 ^ error_hash2[error];\n",
            2,
        ),
        (
            "            child_hash1, state_bits, error, child_empty,\n"
            "            visited_hash1, visited_chain, visited_capacity,",
            "            child_hash1, child_hash2, state_bits, error, child_empty,\n"
            "            visited_hash1, visited_hash2, visited_chain, visited_capacity,",
            1,
        ),
        (
            "        chain_hash1[next_chain] = child_hash1;\n",
            "        chain_hash1[next_chain] = child_hash1;\n"
            "        chain_hash2[next_chain] = child_hash2;\n",
            1,
        ),
    )
    for old, new, expected in replacements:
        transformed = replace(transformed, old, new, expected)
    return transformed


CTA_KERNEL = _build_composed_kernel(single_hash.CTA_KERNEL)
COMMON = single_hash.COMMON

# Two byte-for-byte recovery checks keep both transformation boundaries live.
if _strip_restrict(CTA_KERNEL) != single_hash.CTA_KERNEL:
    raise RuntimeError("removing restrict does not recover single-hash source")
if _restore_dual_hash_kernel(CTA_KERNEL) != restrict_base.CTA_KERNEL:
    raise RuntimeError("removing single-hash does not recover restrict-hot source")


def _compile_options(*, monotonic: bool, layout: object | None = None,
                     max_registers: int = 0) -> tuple[str, ...]:
    options = [
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
    ]
    if max_registers:
        options.append(f"--maxrregcount={int(max_registers)}")
    options.extend((
        f"-DTESSERACT_CTA_THREADS={CTA_THREADS}",
        f"-DTESSERACT_CTA_HEAP_CACHE={HEAP_CACHE}",
        f"-DTESSERACT_MONOTONIC_RANK_ROWS={1 if monotonic else 0}",
    ))
    if layout is None:
        options.extend((
            "-DTESSERACT_RANK32_ERROR_MASK=0x0003ffffU",
            "-DTESSERACT_RANK32_GROUP_MASK=0x0000001fU",
            "-DTESSERACT_RANK32_LOWER_MASK=0x0000007fU",
            "-DTESSERACT_RANK32_GROUP_SHIFT=18",
            "-DTESSERACT_RANK32_LOWER_SHIFT=23",
        ))
    else:
        options.extend((
            f"-DTESSERACT_RANK32_ERROR_MASK=0x{layout.error_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_MASK=0x{layout.group_mask:08x}U",
            f"-DTESSERACT_RANK32_LOWER_MASK=0x{layout.lower_mask:08x}U",
            f"-DTESSERACT_RANK32_GROUP_SHIFT={layout.group_shift}",
            f"-DTESSERACT_RANK32_LOWER_SHIFT={layout.lower_shift}",
        ))
    return tuple(options)


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    """Compile SM75 code without loading or launching the decoder."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=_compile_options(monotonic=monotonic),
        arch="75",
        filename=(
            "cta_compact_fused_rank32_nodebug_csr32_singlehash_"
            "restrict_hot_cta640_decoder.cu"
        ),
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(*, monotonic: bool = True) -> dict[str, int | bool]:
    """Load compiled code and inspect resources without a kernel launch."""
    module = cp.RawModule(
        code=COMMON + CTA_KERNEL,
        options=_compile_options(monotonic=monotonic),
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


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
    "unused_error_hash2": "error_hash2",
    "count_values": "count_values",
    "d2e_ranked": "d2e_ranked32",
    "count_ranks16": "count_ranks16",
    "shot_offsets": "shot_offsets",
    "shot_detections": "shot_detections",
    "initial_shot_bits": "shot_bits",
    "shot_hash1": "shot_hash1",
    "unused_shot_hash2": "shot_hash2",
}

_WORKSPACE_RENAMES = {
    "unused_heap_hash2_all": "heap_hash2",
    "unused_visited_hash2_all": "visited_hash2",
}

WORKSPACE_POINTER_BINDINGS = {
    parameter: _WORKSPACE_RENAMES.get(
        parameter, parameter[:-4] if parameter.endswith("_all") else parameter
    )
    for parameter in HOT_RESTRICT_PARAMETERS["decode_exact_cta"]
    if parameter not in DEVICE_POINTER_BINDINGS
}


def audit_kernel_allocations(
    device: Mapping[str, Any], workspace: Mapping[str, Any]
) -> dict[str, int]:
    """Prove the live allocation contract, including four unique dummies."""
    buffers: dict[str, Any] = {}
    for parameter, name in DEVICE_POINTER_BINDINGS.items():
        buffers[parameter] = device[name]
    for parameter, name in WORKSPACE_POINTER_BINDINGS.items():
        buffers[parameter] = workspace[name]
    expected = set(HOT_RESTRICT_PARAMETERS["decode_exact_cta"])
    if set(buffers) != expected:
        missing = sorted(expected - set(buffers))
        extra = sorted(set(buffers) - expected)
        raise RuntimeError(
            f"composed alias audit binding mismatch: missing={missing}, extra={extra}"
        )
    spans = audited.assert_nonoverlapping_allocations(buffers)
    return {
        "pointer_parameters": len(buffers),
        "nonempty_allocations": len(spans),
        "empty_allocations": len(buffers) - len(spans),
        "unused_hash2_placeholders": 4,
    }


class ExactCtaDecoder(single_hash.ExactCtaDecoder):
    """Fixed composition with checked CSR32 and a mandatory alias audit."""

    kernel_source = CTA_KERNEL
    common_source = COMMON

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = CTA_THREADS, heap_cache: int = HEAP_CACHE):
        constructor_start = time.perf_counter()
        if int(cta_threads) != CTA_THREADS:
            raise ValueError(f"fixed composition requires cta_threads={CTA_THREADS}")
        if int(heap_cache) != HEAP_CACHE:
            raise ValueError(f"fixed composition requires heap_cache={HEAP_CACHE}")

        csr_audit_start = time.perf_counter()
        self.csr32_offsets = single_hash.base.csr32.pack_model_csr32_offsets(model)
        csr_audit_ms = (time.perf_counter() - csr_audit_start) * 1000
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = CTA_THREADS
        self.heap_cache = HEAP_CACHE
        self.visited_capacity = single_hash.base.rank32._next_power_of_two(  # pylint: disable=protected-access
            2 * self.workspace_capacity
        )
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("packed state requires max_output < 65535")
        self.max_error_degree, self.max_row_length = (
            single_hash.base.rank32._validate_fused_state_model(model)  # pylint: disable=protected-access
        )
        if not np.array_equal(
            model.detector_order,
            np.arange(len(model.detector_order), dtype=np.uint32),
        ):
            raise ValueError("CTA prototype currently requires identity detector order")
        self.d2e_ranked32, self.rank32_layout = (
            single_hash.base.rank32.repack_d2e_ranked32(model)
        )

        compile_start = time.perf_counter()
        self.module = cp.RawModule(
            code=self.common_source + self.kernel_source,
            options=_compile_options(
                monotonic=bool(model.rank_rows_monotonic),
                layout=self.rank32_layout,
                max_registers=max_registers,
            ),
            name_expressions=("decode_exact_cta",),
            backend="nvrtc",
        )
        self.kernel = self.module.get_function("decode_exact_cta")
        compile_ms = (time.perf_counter() - compile_start) * 1000

        packed_offsets = self.csr32_offsets.as_dict()
        upload_start = time.perf_counter()
        self.device = {}
        for name in (
            "d2e_offsets", "d2e", "edet_offsets", "edets",
            "eneighbor_word_offsets", "eneighbor_words", "eneighbor_masks",
            "weights", "error_to_dem", "detector_order", "error_hash1",
            "count_values", "count_ranks16", "shot_offsets",
            "shot_detections", "shot_bits", "shot_hash1",
        ):
            source = packed_offsets.get(name, getattr(model, name))
            self.device[name] = cp.asarray(source)
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        # Unique, nonempty dummies satisfy restrict.  The resident chunk
        # runner uploads all five shot buffers, so this unused hash2 buffer
        # must retain the resident shot shape even though CUDA never reads it.
        self.device["error_hash2"] = cp.empty(1, dtype=cp.uint64)
        self.device["shot_hash2"] = cp.empty(
            int(model.metadata["shots"]), dtype=cp.uint64
        )
        upload_ms = (time.perf_counter() - upload_start) * 1000

        allocation_start = time.perf_counter()
        self.workspace = self._allocate_workspace()
        allocation_ms = (time.perf_counter() - allocation_start) * 1000

        alias_start = time.perf_counter()
        self.restrict_alias_audit = audit_kernel_allocations(
            self.device, self.workspace
        )
        alias_ms = (time.perf_counter() - alias_start) * 1000
        self.restrict_initialization_ms = {
            "csr32_validation": csr_audit_ms,
            "nvrtc_and_load": compile_ms,
            "device_upload": upload_ms,
            "workspace_allocation": allocation_ms,
            "alias_audit": alias_ms,
            "constructor_total": (time.perf_counter() - constructor_start) * 1000,
        }

    def _allocate_workspace(self) -> dict[str, cp.ndarray]:
        workspace = super()._allocate_workspace()
        workspace["heap_hash2"] = cp.empty(1, dtype=cp.uint64)
        workspace["visited_hash2"] = cp.empty(1, dtype=cp.uint64)
        return workspace

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["fixed_cta_threads"] = CTA_THREADS
        stats["fixed_heap_cache"] = HEAP_CACHE
        stats["restrict_profile"] = "single_hash_csr32_kernel_and_detector_cost"
        stats["restrict_alias_audit"] = dict(self.restrict_alias_audit)
        stats["restrict_initialization_ms"] = dict(
            self.restrict_initialization_ms
        )
        return decisions, stats


def main() -> None:
    import argparse

    single_hash.base.rank32._pin_one_cpu()  # pylint: disable=protected-access
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
        offsets = decoder.csr32_offsets.as_dict()
        payload = json.dumps({
            "model": str(args.model),
            "model_load_ms": model_load_ms,
            "initialization_ms": decoder.restrict_initialization_ms,
            "alias_audit": decoder.restrict_alias_audit,
            "csr32_max_offsets": {
                name: int(values.max(initial=0))
                for name, values in offsets.items()
            },
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
