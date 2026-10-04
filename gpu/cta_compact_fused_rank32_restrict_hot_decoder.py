"""Narrow exact BB alias candidate: kernel boundary and detector scans only.

The broader restrict experiment qualifies every audited helper.  This variant
keeps only the contracts most likely to affect BB query traffic:

* every top-level kernel pointer, guarded by the same live-allocation audit;
* immutable/error-state and shared reduction pointers in detector_cost_cta;
* immutable/error-state pointers in both detector_cost_subwarp definitions.

No other helper signature and no executable statement changes.
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

import cta_compact_fused_rank32_nodebug_decoder as base
import cta_compact_fused_rank32_restrict_decoder as audited
from gpu_decoder import HostModel, compare_with_reference


HOT_RESTRICT_PARAMETERS = {
    function: audited.CTA_RESTRICT_PARAMETERS[function]
    for function in (
        "detector_cost_cta",
        "detector_cost_subwarp",
        "decode_exact_cta",
    )
}


def _build_hot_restrict_kernel(source: str) -> str:
    transformed = audited._qualify_signature_parameters(  # pylint: disable=protected-access
        source, HOT_RESTRICT_PARAMETERS
    )
    expected = sum(
        len(names) * audited.EXPECTED_DEFINITIONS.get(function, 1)
        for function, names in HOT_RESTRICT_PARAMETERS.items()
    )
    if transformed.count("__restrict__") - source.count("__restrict__") != expected:
        raise RuntimeError("unexpected hot restrict qualifier count")
    return transformed


CTA_KERNEL = _build_hot_restrict_kernel(base.CTA_KERNEL)
COMMON = base.COMMON
ALLOWED_CTA_THREADS = audited.ALLOWED_CTA_THREADS
ALLOWED_HEAP_CACHE = audited.ALLOWED_HEAP_CACHE


def nvrtc_compile_check(
    *, monotonic: bool = True, cta_threads: int = 512, heap_cache: int = 1024
) -> object:
    """Compile complete SM75 code without launching a decoder."""
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=audited._compile_options(  # pylint: disable=protected-access
            cta_threads=cta_threads, heap_cache=heap_cache, monotonic=monotonic
        ),
        arch="75",
        filename="cta_compact_fused_rank32_restrict_hot_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


def runtime_kernel_attributes(
    *, cta_threads: int = 512, heap_cache: int = 1024,
    monotonic: bool = True,
) -> dict[str, int | bool]:
    """Load and inspect the kernel without launching it."""
    module = cp.RawModule(
        code=COMMON + CTA_KERNEL,
        options=audited._compile_options(  # pylint: disable=protected-access
            cta_threads=cta_threads, heap_cache=heap_cache, monotonic=monotonic
        ),
        name_expressions=("decode_exact_cta",),
        backend="nvrtc",
    )
    return dict(module.get_function("decode_exact_cta").attributes)


class ExactCtaDecoder(audited.ExactCtaDecoder):
    """Narrow source selection with the complete top-level alias audit."""

    kernel_source = CTA_KERNEL
    common_source = COMMON

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["restrict_profile"] = "kernel_and_detector_cost"
        return decisions, stats


def main() -> None:
    import argparse

    audited.rank32._pin_one_cpu()  # pylint: disable=protected-access
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
