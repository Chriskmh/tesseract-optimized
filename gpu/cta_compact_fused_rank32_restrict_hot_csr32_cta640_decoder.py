"""Immutable exact BB composition: CSR32 + narrow restrict + CTA640.

The accepted checked-CSR32 source is transformed once with the narrow
kernel-and-detector restrict allowlist.  Runtime specialization is fixed to
CTA640 and a 1024-entry shared heap, the accepted CSR32 launch shape.  Every
kernel pointer allocation is checked for overlap before decoding can begin.
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
import numpy as np

import cta_compact_fused_rank32_decoder as rank32
import cta_compact_fused_rank32_nodebug_csr32_decoder as csr32_base
import cta_compact_fused_rank32_restrict_decoder as audited
import cta_compact_fused_rank32_restrict_hot_decoder as hot
from gpu_decoder import HostModel, compare_with_reference


CTA_THREADS = 640
HEAP_CACHE = 1024
CTA_KERNEL = hot._build_hot_restrict_kernel(  # pylint: disable=protected-access
    csr32_base.CTA_KERNEL
)
COMMON = csr32_base.COMMON


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
    """Compile and link SM75 code without launching the decoder."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=_compile_options(monotonic=monotonic),
        arch="75",
        filename="cta_compact_fused_rank32_restrict_hot_csr32_cta640_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(*, monotonic: bool = True) -> dict[str, int | bool]:
    """Return fixed-shape driver resources without a kernel launch."""
    module = cp.RawModule(
        code=COMMON + CTA_KERNEL,
        options=_compile_options(monotonic=monotonic),
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


class ExactCtaDecoder(hot.ExactCtaDecoder):
    """Checked CSR32 uploads with immutable fixed-shape CUDA source."""

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
        self.csr32_offsets = csr32_base.csr32.pack_model_csr32_offsets(model)
        csr_audit_ms = (time.perf_counter() - csr_audit_start) * 1000
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = CTA_THREADS
        self.heap_cache = HEAP_CACHE
        self.visited_capacity = rank32._next_power_of_two(  # pylint: disable=protected-access
            2 * self.workspace_capacity
        )
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("packed state requires max_output < 65535")
        self.max_error_degree, self.max_row_length = (
            rank32._validate_fused_state_model(model)  # pylint: disable=protected-access
        )
        if not np.array_equal(
            model.detector_order,
            np.arange(len(model.detector_order), dtype=np.uint32),
        ):
            raise ValueError("CTA prototype currently requires identity detector order")
        self.d2e_ranked32, self.rank32_layout = rank32.repack_d2e_ranked32(model)

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
            "error_hash2", "count_values", "count_ranks16", "shot_offsets",
            "shot_detections", "shot_bits", "shot_hash1", "shot_hash2",
        ):
            source = packed_offsets.get(name, getattr(model, name))
            self.device[name] = cp.asarray(source)
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        upload_ms = (time.perf_counter() - upload_start) * 1000

        allocation_start = time.perf_counter()
        self.workspace = self._allocate_workspace()
        allocation_ms = (time.perf_counter() - allocation_start) * 1000

        alias_start = time.perf_counter()
        self.restrict_alias_audit = audited.audit_kernel_allocations(
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

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["csr32_offsets"] = True
        stats["fixed_cta_threads"] = CTA_THREADS
        stats["fixed_heap_cache"] = HEAP_CACHE
        stats["restrict_profile"] = "csr32_kernel_and_detector_cost"
        return decisions, stats


def main() -> None:
    import argparse

    rank32._pin_one_cpu()  # pylint: disable=protected-access
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
