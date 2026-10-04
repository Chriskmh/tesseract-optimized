"""Production-shaped BB kernel with checked 32-bit CSR offsets.

This isolated candidate applies the already proved CSR32 address ABI to the
current no-debug, monotonic-stop-prefetch winner.  It changes integer address
widths only; payload order and every search/floating-point operation remain
unchanged.  No dispatcher selects this candidate until a real exact A/B run
shows a benefit.
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

import cta_compact_csr32_decoder as csr32
import cta_compact_fused_rank32_decoder as rank32
import cta_compact_fused_rank32_nodebug_decoder as base
from gpu_decoder import HostModel, compare_with_reference


def _replace_exact(source: str, old: str, new: str, expected: int) -> str:
    observed = source.count(old)
    if observed != expected:
        raise RuntimeError(
            f"nodebug CSR32 audit expected {expected} copies of {old!r}, "
            f"found {observed}"
        )
    return source.replace(old, new)


def _build_csr32_kernel(source: str) -> str:
    """Narrow only checked CSR bounds/cursors in the current winner.

    The stop-prefetch source carries both monotonic and nonmonotonic subwarp
    definitions behind preprocessing, hence the extra function/cursor copies
    relative to the older CSR32 prototype.
    """
    replacements = (
        ("const u64 *d2e_offsets", "const u32 *d2e_offsets", 10),
        ("const u64 *edet_offsets", "const u32 *edet_offsets", 6),
        ("const u64 *eneighbor_word_offsets", "const u32 *eneighbor_word_offsets", 1),
        ("const u64 *shot_offsets", "const u32 *shot_offsets", 1),
        ("for (u64 row =", "for (u32 row =", 5),
        ("for (u64 j =", "for (u32 j =", 5),
        ("for (u64 base =", "for (u32 base =", 3),
        ("for (u64 i =", "for (u32 i =", 2),
        ("for (u64 branch_row =", "for (u32 branch_row =", 1),
        ("for (u64 group =", "for (u32 group =", 1),
        ("u64 old_position =", "u32 old_position =", 1),
        ("const u64 old_end =", "const u32 old_end =", 1),
        ("u64 new_position =", "u32 new_position =", 1),
        ("const u64 new_end =", "const u32 new_end =", 1),
        ("const u64 begin =", "const u32 begin =", 4),
        ("const u64 end =", "const u32 end =", 3),
        ("const u64 position =", "const u32 position =", 4),
        ("const u64 selected =", "const u32 selected =", 1),
        ("const u64 shot_begin =", "const u32 shot_begin =", 1),
        ("const u64 shot_end =", "const u32 shot_end =", 1),
        ("const u64 branch_begin =", "const u32 branch_begin =", 1),
        ("const u64 branch_end =", "const u32 branch_end =", 1),
        ("(u64)tid", "(u32)tid", 2),
        ("(u64)lane", "(u32)lane", 2),
    )
    transformed = source
    for old, new, expected in replacements:
        transformed = _replace_exact(transformed, old, new, expected)
    for forbidden in (
        "const u64 *d2e_offsets",
        "const u64 *edet_offsets",
        "const u64 *eneighbor_word_offsets",
        "const u64 *shot_offsets",
    ):
        if forbidden in transformed:
            raise RuntimeError(f"unconverted nodebug CSR32 ABI fragment: {forbidden}")
    for required in (
        "(u64)shot * workspace_capacity",
        "(u64)shot * num_errors",
        "(u64)shot * num_detectors",
        "(u64)shot * num_words",
        "(u64)shot * visited_capacity",
        "(u64)shot * max_output",
    ):
        if required not in transformed:
            raise RuntimeError(f"required wide pointer product changed: {required}")
    return transformed


CTA_KERNEL = _build_csr32_kernel(base.CTA_KERNEL)
COMMON = csr32._build_csr32_common(base.COMMON)  # pylint: disable=protected-access
ALLOWED_CTA_THREADS = (32, 64, 128, 256, 512, 544, 576, 608, 640)
ALLOWED_HEAP_CACHE = (128, 256, 512, 1024, 1536, 2048)


def nvrtc_compile_check(*, monotonic: bool = True,
                        cta_threads: int = 512,
                        heap_cache: int = 1024) -> object:
    """Compile SM75 PTX without loading or launching the kernel."""
    if cta_threads not in ALLOWED_CTA_THREADS:
        raise ValueError(f"unsupported CTA width {cta_threads}")
    if heap_cache not in ALLOWED_HEAP_CACHE:
        raise ValueError(f"unsupported heap cache {heap_cache}")
    options = (
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
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL, options=options, arch="75",
        filename="cta_compact_fused_rank32_nodebug_csr32_decoder.cu",
        name_expressions=("decode_exact_cta",), cache_in_memory=True,
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Current BB winner with losslessly narrowed hot CSR offsets."""

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        self.csr32_offsets = csr32.pack_model_csr32_offsets(model)
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
            code=COMMON + CTA_KERNEL, options=tuple(options),
            name_expressions=("decode_exact_cta",), backend="nvrtc",
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
        stats["csr32_offsets"] = True
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
        "--cta-threads", type=int, choices=ALLOWED_CTA_THREADS, default=128)
    parser.add_argument(
        "--heap-cache", type=int, choices=ALLOWED_HEAP_CACHE, default=128)
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
