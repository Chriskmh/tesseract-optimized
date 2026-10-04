"""Production Surface CTA specialization without diagnostic telemetry.

The search, exact result buffers, rank ordering, floating-point operations,
status codes, and fallback contract are inherited from the retained sibling
rank32 stop-prefetch winner.  The generated kernel changes only two things:

* per-shot diagnostic writes and the clock sampling used only by reports are
  removed; and
* the supported 256-thread production launch is declared with two resident
  blocks per SM as its register-allocation target.

The output status, length, ordered error list, and raw binary64 cost bits stay
in the kernel ABI and are written by byte-identical code.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cupy as cp

import cta_sibling_rank32_prefetch_decoder as base
from gpu_decoder import HostModel, compare_with_reference


CTA_THREADS = 256
MIN_BLOCKS_PER_SM = 2
DEBUG_START = "struct CtaDebugRecorder {"
DEBUG_END = "__device__ __forceinline__ bool update_count_row_cta("
EMPTY_DEBUG = r"""struct CtaDebugRecorder {
  __device__ CtaDebugRecorder(u64 *, int, int) {}
};

"""


def _build_kernel(source: str) -> str:
    """Remove diagnostics with count-checked, source-local transformations."""
    if source.count(DEBUG_START) != 1 or source.count(DEBUG_END) != 1:
        raise RuntimeError("Surface debug recorder boundaries are not unique")
    start = source.index(DEBUG_START)
    end = source.index(DEBUG_END, start)
    transformed = source[:start] + EMPTY_DEBUG + source[end:]

    recorder_line = "  CtaDebugRecorder debug(out_debug, shot, tid);\n"
    if transformed.count(recorder_line) != 1:
        raise RuntimeError("Surface debug recorder construction is not unique")
    transformed = transformed.replace(recorder_line, "", 1)

    transformed, debug_lines = re.subn(
        r"^.*\bdebug\.[^\n]*\n", "", transformed, flags=re.MULTILINE
    )
    if debug_lines != 8:
        raise RuntimeError(
            f"expected eight Surface debug counter statements, found {debug_lines}"
        )

    cycle_line = "  const u64 cycle_start = clock64();\n"
    if transformed.count(cycle_line) != 1:
        raise RuntimeError("Surface cycle-start telemetry is not unique")
    transformed = transformed.replace(cycle_line, "", 1)
    transformed, telemetry_lines = re.subn(
        r"^[ \t]+out_(?:pushes|pops|cycles)\[shot\] = [^;\n]+;\n",
        "",
        transformed,
        flags=re.MULTILINE,
    )
    if telemetry_lines != 15:
        raise RuntimeError(
            "expected fifteen Surface telemetry writes, "
            f"found {telemetry_lines}"
        )

    launch = '__global__ __launch_bounds__(CTA_THREADS) void decode_exact_cta('
    bounded = (
        '__global__ __launch_bounds__(CTA_THREADS, 2) '
        'void decode_exact_cta('
    )
    if transformed.count(launch) != 1:
        raise RuntimeError("Surface launch-bounds marker is not unique")
    transformed = transformed.replace(launch, bounded, 1)

    for forbidden in (
        "debug.", "CtaDebugRecorder debug", "cycle_start", "out_pushes[shot] =",
        "out_pops[shot] =", "out_cycles[shot] =",
    ):
        if forbidden in transformed:
            raise RuntimeError(f"diagnostic operation survived: {forbidden}")
    for required in (
        "out_status[shot]", "out_length[shot]", "out_errors[output_index]",
        "out_cost_bits[shot]", "__launch_bounds__(CTA_THREADS, 2)",
    ):
        if required not in transformed:
            raise RuntimeError(f"required exact-result operation missing: {required}")
    return transformed


CTA_KERNEL = _build_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


def _compile_options(*, monotonic: bool) -> tuple[str, ...]:
    return (
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
        "-DTESSERACT_CTA_THREADS=256",
        f"-DTESSERACT_MONOTONIC_RANK_ROWS={1 if monotonic else 0}",
        "-DTESSERACT_PACKED_STAMPS=1",
        "-DTESSERACT_RANK32_ERROR_MASK=0x0001ffffU",
        "-DTESSERACT_RANK32_GROUP_MASK=0x0000003fU",
        "-DTESSERACT_RANK32_LOWER_MASK=0x0000007fU",
        "-DTESSERACT_RANK32_GROUP_SHIFT=17",
        "-DTESSERACT_RANK32_LOWER_SHIFT=23",
    )


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    """Compile the full SM75 translation unit without launching decoding."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=_compile_options(monotonic=monotonic),
        arch="75",
        filename="cta_sibling_rank32_prefetch_nodebug_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(*, production: bool = True) -> dict[str, int | bool]:
    """Load one compiled kernel and return attributes without launching it."""
    source = CTA_KERNEL if production else base.CTA_KERNEL
    module = cp.RawModule(
        code=COMMON + source,
        options=_compile_options(monotonic=True),
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Immutable-source production specialization for the 256-thread winner."""

    common_source = COMMON
    kernel_source = CTA_KERNEL
    diagnostic_telemetry = False

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = CTA_THREADS,
                 stamp_layout: str = "packed"):
        if int(cta_threads) != CTA_THREADS:
            raise ValueError(
                "Surface production launch bounds require cta_threads=256"
            )
        super().__init__(
            model, workspace_capacity, max_output, max_registers,
            cta_threads, stamp_layout,
        )

    def _allocate_workspace(self) -> dict[str, cp.ndarray]:
        workspace = super()._allocate_workspace()
        # The specialized kernel keeps these pointers only as inert ABI
        # placeholders.  One element is sufficient and prevents per-shot
        # diagnostic storage from entering the resident memory plan.
        workspace["out_pushes"] = cp.empty(1, cp.uint32)
        workspace["out_pops"] = cp.empty(1, cp.uint32)
        workspace["out_cycles"] = cp.empty(1, cp.uint64)
        workspace["out_debug"] = cp.empty(1, cp.uint64)
        return workspace

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["diagnostic_telemetry"] = False
        stats["launch_bounds_min_blocks_per_sm"] = MIN_BLOCKS_PER_SM
        return decisions, stats


def main() -> None:
    base.base._pin_one_cpu()  # pylint: disable=protected-access
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--cta-threads", type=int, choices=(256,), default=256)
    parser.add_argument(
        "--stamp-layout", choices=("separate", "packed"), default="packed"
    )
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model)
    decoder = ExactCtaDecoder(
        model, args.workspace, args.max_output, args.max_registers,
        args.cta_threads, args.stamp_layout,
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
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload + "\n", encoding="utf-8")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

