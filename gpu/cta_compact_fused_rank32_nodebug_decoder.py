"""Production-shaped exact BB prefetch kernel without diagnostic counters."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp

import cta_compact_fused_rank32_prefetch_decoder as base
from gpu_decoder import HostModel, compare_with_reference


DEBUG_START = "struct CtaDebugRecorder {"
DEBUG_END = "__device__ __forceinline__ bool update_count_row_cta("

EMPTY_DEBUG = r"""struct CtaDebugRecorder {
  __device__ CtaDebugRecorder(u64 *, int, int) {}
};

"""

DEBUG_STATEMENTS = (
    "    if (tid == 0) ++debug.expanded;\n",
    "      if (tid == 0) ++debug.branch_rows;\n",
    "      if (tid == 0) ++debug.base_pass;\n",
    "      if (tid == 0) ++debug.beam_pass;\n",
    "      if (tid == 0) ++debug.visited_pass;\n",
)

DEBUG_QUERY_BLOCK = r"""      if (tid == 0) {
        debug.affected += s_affected_count;
        for (u32 index = 0; index < s_affected_count; ++index) {
          debug.parent_queries += s_affected_operation[index] & 1U;
          debug.child_queries += (s_affected_operation[index] >> 1) & 1U;
        }
      }
"""


def _build_kernel(source: str) -> str:
    if source.count(DEBUG_START) != 1 or source.count(DEBUG_END) != 1:
        raise RuntimeError("debug recorder boundaries are not unique")
    start = source.index(DEBUG_START)
    end = source.index(DEBUG_END, start)
    source = source[:start] + EMPTY_DEBUG + source[end:]
    for statement in DEBUG_STATEMENTS:
        if source.count(statement) != 1:
            raise RuntimeError(f"debug statement is not unique: {statement!r}")
        source = source.replace(statement, "", 1)
    if source.count(DEBUG_QUERY_BLOCK) != 1:
        raise RuntimeError("debug query block is not unique")
    source = source.replace(DEBUG_QUERY_BLOCK, "", 1)
    if "debug." in source:
        raise RuntimeError("a diagnostic counter survived production transform")
    launch = '__global__ __launch_bounds__(CTA_THREADS) void decode_exact_cta('
    if source.count(launch) != 1:
        raise RuntimeError("kernel launch-bounds marker is not unique")
    source = source.replace(
        launch,
        '__global__ __launch_bounds__(CTA_THREADS, 1) void decode_exact_cta(',
        1,
    )
    return source


CTA_KERNEL = _build_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    options = (
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
        "-DTESSERACT_CTA_THREADS=512", "-DTESSERACT_CTA_HEAP_CACHE=1024",
        f"-DTESSERACT_MONOTONIC_RANK_ROWS={1 if monotonic else 0}",
        "-DTESSERACT_RANK32_ERROR_MASK=0x0003ffffU",
        "-DTESSERACT_RANK32_GROUP_MASK=0x0000001fU",
        "-DTESSERACT_RANK32_LOWER_MASK=0x0000007fU",
        "-DTESSERACT_RANK32_GROUP_SHIFT=18",
        "-DTESSERACT_RANK32_LOWER_SHIFT=23",
    )
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL, options=options, arch="75",
        filename="cta_compact_fused_rank32_nodebug_decoder.cu",
        name_expressions=("decode_exact_cta",), cache_in_memory=True,
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    def __init__(self, *args, **kwargs):
        original = base.CTA_KERNEL
        base.CTA_KERNEL = CTA_KERNEL
        try:
            super().__init__(*args, **kwargs)
        finally:
            base.CTA_KERNEL = original

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["debug_enabled"] = False
        stats["debug_totals"] = {}
        return decisions, stats


def main() -> None:
    base.base._pin_one_cpu()
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--cta-threads", type=int, choices=(32, 64, 128, 256, 512), default=128)
    parser.add_argument("--heap-cache", type=int, choices=(128, 256, 512, 1024), default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
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
    print(json.dumps({
        "model": str(args.model), "metadata": model.metadata, "runs": runs,
        "mismatch_count": len(mismatches), "mismatches": mismatches[:20],
    }, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
