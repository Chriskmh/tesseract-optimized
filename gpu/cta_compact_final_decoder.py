"""Exact fused rank32/prefix-fold CTA decoder with checked uint32 CSR offsets.

The flattened payloads of the four hot search CSRs are already bounded by
uint32 in supported exports.  This isolated experiment validates those bounds
on the host, uploads lossless uint32 copies, and uses uint32 row cursors in the
kernel.  Shot/workspace pointer products remain explicitly uint64.
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

import cta_compact_fused_rank32_prefixfold_decoder as prefixfold
import cta_compact_fused_rank32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


UINT32_MAX = int(np.iinfo(np.uint32).max)


@dataclass(frozen=True)
class Csr32Offsets:
    """Lossless device representations of every converted CSR offset array."""

    d2e_offsets: np.ndarray
    edet_offsets: np.ndarray
    eneighbor_word_offsets: np.ndarray
    shot_offsets: np.ndarray

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "d2e_offsets": self.d2e_offsets,
            "edet_offsets": self.edet_offsets,
            "eneighbor_word_offsets": self.eneighbor_word_offsets,
            "shot_offsets": self.shot_offsets,
        }


def checked_csr32_offsets(
    name: str,
    offsets: np.ndarray,
    *,
    expected_rows: int,
    payload_length: int,
) -> np.ndarray:
    """Validate one CSR completely and return a lossless contiguous uint32 copy."""
    if expected_rows < 0:
        raise ValueError(f"{name}: expected_rows must be nonnegative")
    if payload_length < 0 or payload_length > UINT32_MAX:
        raise ValueError(
            f"{name}: payload length {payload_length} exceeds UINT32_MAX"
        )
    source = np.asarray(offsets)
    if source.ndim != 1:
        raise ValueError(f"{name}: offsets must be one-dimensional")
    if source.dtype.kind not in "iu" or source.dtype.itemsize > 8:
        raise ValueError(f"{name}: offsets must use an integer dtype up to 64 bits")
    if len(source) != expected_rows + 1:
        raise ValueError(
            f"{name}: expected {expected_rows + 1} offsets, got {len(source)}"
        )
    if source.dtype.kind == "i" and bool(np.any(source < 0)):
        raise ValueError(f"{name}: offsets must be nonnegative")

    wide = source.astype(np.uint64, copy=False)
    if int(wide[0]) != 0:
        raise ValueError(f"{name}: first offset must be zero")
    if bool(np.any(wide[1:] < wide[:-1])):
        raise ValueError(f"{name}: offsets must be monotonically nondecreasing")
    terminal = int(wide[-1])
    if terminal != payload_length:
        raise ValueError(
            f"{name}: terminal offset {terminal} does not match payload length "
            f"{payload_length}"
        )
    maximum = int(wide.max(initial=0))
    if maximum > UINT32_MAX:
        raise ValueError(f"{name}: maximum offset {maximum} exceeds UINT32_MAX")

    packed = np.ascontiguousarray(wide.astype(np.uint32, copy=False))
    roundtrip = packed.astype(np.uint64, copy=False)
    if not np.array_equal(roundtrip, wide):
        raise ValueError(f"{name}: uint32 conversion is not lossless")
    return packed


def pack_model_csr32_offsets(model: HostModel) -> Csr32Offsets:
    """Validate shape, monotonicity, terminal, bound, and roundtrip per CSR."""
    metadata = model.metadata
    detectors = int(metadata["num_detectors"])
    errors = int(metadata["num_errors"])
    shots = int(metadata["shots"])
    return Csr32Offsets(
        d2e_offsets=checked_csr32_offsets(
            "d2e_offsets", model.d2e_offsets,
            expected_rows=detectors, payload_length=len(model.d2e),
        ),
        edet_offsets=checked_csr32_offsets(
            "edet_offsets", model.edet_offsets,
            expected_rows=errors, payload_length=len(model.edets),
        ),
        eneighbor_word_offsets=checked_csr32_offsets(
            "eneighbor_word_offsets", model.eneighbor_word_offsets,
            expected_rows=errors, payload_length=len(model.eneighbor_words),
        ),
        shot_offsets=checked_csr32_offsets(
            "shot_offsets", model.shot_offsets,
            expected_rows=shots, payload_length=len(model.shot_detections),
        ),
    )


def _replace_exact(source: str, old: str, new: str, expected: int) -> str:
    observed = source.count(old)
    if observed != expected:
        raise RuntimeError(
            f"CSR32 kernel audit expected {expected} copies of {old!r}, "
            f"found {observed}"
        )
    return source.replace(old, new)


def _build_csr32_kernel(source: str) -> str:
    """Apply an exhaustive, count-checked CSR offset/index ABI conversion."""
    replacements = (
        ("const u64 *d2e_offsets", "const u32 *d2e_offsets", 9),
        ("const u64 *edet_offsets", "const u32 *edet_offsets", 6),
        ("const u64 *eneighbor_word_offsets", "const u32 *eneighbor_word_offsets", 1),
        ("const u64 *shot_offsets", "const u32 *shot_offsets", 1),
        ("for (u64 row =", "for (u32 row =", 5),
        ("for (u64 j =", "for (u32 j =", 5),
        ("for (u64 base =", "for (u32 base =", 2),
        ("for (u64 i =", "for (u32 i =", 2),
        ("for (u64 branch_row =", "for (u32 branch_row =", 1),
        ("for (u64 group =", "for (u32 group =", 1),
        ("u64 old_position =", "u32 old_position =", 1),
        ("const u64 old_end =", "const u32 old_end =", 1),
        ("u64 new_position =", "u32 new_position =", 1),
        ("const u64 new_end =", "const u32 new_end =", 1),
        ("const u64 begin =", "const u32 begin =", 3),
        ("const u64 end =", "const u32 end =", 2),
        ("const u64 position =", "const u32 position =", 3),
        ("const u64 selected =", "const u32 selected =", 1),
        ("const u64 shot_begin =", "const u32 shot_begin =", 1),
        ("const u64 shot_end =", "const u32 shot_end =", 1),
        ("const u64 branch_begin =", "const u32 branch_begin =", 1),
        ("const u64 branch_end =", "const u32 branch_end =", 1),
        ("(u64)tid", "(u32)tid", 2),
        ("(u64)lane", "(u32)lane", 1),
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
            raise RuntimeError(f"unconverted CSR ABI fragment: {forbidden}")
    # Global shot/workspace strides must remain wide even when local rows are u32.
    for required in (
        "(u64)shot * workspace_capacity",
        "(u64)shot * num_errors",
        "(u64)shot * num_words",
        "(u64)shot * max_output",
    ):
        if required not in transformed:
            raise RuntimeError(f"required wide pointer product changed: {required}")
    return transformed


def _build_csr32_common(source: str) -> str:
    """Convert shared helper signatures so CTA calls have one offset ABI.

    ``COMMON`` contains the visited-state equality helpers used by the CTA,
    plus dormant serial helpers parsed by NVRTC in the same translation unit.
    CUDA type-checks all of them even though only ``visited_find`` is reached by
    this kernel, so every d2e/edet signature and local cursor is converted.
    """
    replacements = (
        ("const u64 *d2e_offsets", "const u32 *d2e_offsets", 2),
        ("const u64 *edet_offsets", "const u32 *edet_offsets", 5),
        ("for (u64 j =", "for (u32 j =", 4),
        ("for (u64 i =", "for (u32 i =", 1),
        ("const u64 begin =", "const u32 begin =", 1),
        ("const u64 end =", "const u32 end =", 1),
    )
    transformed = source
    for old, new, expected in replacements:
        transformed = _replace_exact(transformed, old, new, expected)
    for forbidden in (
        "const u64 *d2e_offsets",
        "const u64 *edet_offsets",
    ):
        if forbidden in transformed:
            raise RuntimeError(f"unconverted shared-helper ABI fragment: {forbidden}")
    return transformed


CTA_KERNEL = _build_csr32_kernel(prefixfold.CTA_KERNEL)
COMMON = _build_csr32_common(prefixfold.COMMON)


def nvrtc_compile_check() -> object:
    """Compile the complete generated translation unit without launching CUDA.

    An explicit SM 7.5 virtual architecture avoids querying a device.  This is
    a type-level integration check for helper signatures, call sites, kernel
    ABI, and generated CUDA syntax; it neither loads nor launches the kernel.
    """
    options = (
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
        "-DTESSERACT_CTA_THREADS=128",
        "-DTESSERACT_CTA_HEAP_CACHE=128",
        "-DTESSERACT_MONOTONIC_RANK_ROWS=0",
        "-DTESSERACT_RANK32_ERROR_MASK=0x0003ffffU",
        "-DTESSERACT_RANK32_GROUP_MASK=0x0000001fU",
        "-DTESSERACT_RANK32_LOWER_MASK=0x000001ffU",
        "-DTESSERACT_RANK32_GROUP_SHIFT=18",
        "-DTESSERACT_RANK32_LOWER_SHIFT=23",
    )
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=options,
        arch="75",
        filename="cta_compact_final_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


class ExactCtaDecoder(prefixfold.ExactCtaDecoder):
    """Fused rank32/prefix-fold decoder with a checked uint32 offset ABI."""

    # Allow an audited specialization to select immutable source through normal
    # inheritance.  This avoids the process-global source swapping used by early
    # experiments when decoders are constructed concurrently.
    common_source = COMMON
    kernel_source = CTA_KERNEL

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        self.csr32_offsets = pack_model_csr32_offsets(model)
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.heap_cache = int(heap_cache)
        self.visited_capacity = base._next_power_of_two(  # pylint: disable=protected-access
            2 * self.workspace_capacity
        )
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("packed state requires max_output < 65535")
        if self.cta_threads not in (32, 64, 128, 256, 512):
            raise ValueError("cta_threads must be one of 32, 64, 128, 256, 512")
        if self.heap_cache not in (128, 256, 512, 1024):
            raise ValueError("heap_cache must be one of 128, 256, 512, 1024")
        self.max_error_degree, self.max_row_length = (
            base._validate_fused_state_model(model)  # pylint: disable=protected-access
        )
        if not getattr(self, "supports_detector_orders", False) and not np.array_equal(
            model.detector_order,
            np.arange(len(model.detector_order), dtype=np.uint32),
        ):
            raise ValueError("CTA prototype currently requires identity detector order")
        self.d2e_ranked32, self.rank32_layout = base.repack_d2e_ranked32(model)
        options = [
            "--std=c++11", "--fmad=false", "--prec-div=true",
            "--prec-sqrt=true", "--ftz=false",
        ]
        if max_registers:
            options.append(f"--maxrregcount={int(max_registers)}")
        options.append(f"-DTESSERACT_CTA_THREADS={self.cta_threads}")
        options.append(f"-DTESSERACT_CTA_HEAP_CACHE={self.heap_cache}")
        options.append(
            "-DTESSERACT_MONOTONIC_RANK_ROWS="
            + ("1" if model.rank_rows_monotonic else "0")
        )
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
            host_array = packed_offsets.get(name, getattr(model, name))
            self.device[name] = cp.asarray(host_array)
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        self.workspace = self._allocate_workspace()

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["csr_offset_storage_bits"] = 32
        stats["csr_offset_bytes"] = {
            name: int(array.nbytes)
            for name, array in self.csr32_offsets.as_dict().items()
        }
        return decisions, stats


def main() -> None:
    base._pin_one_cpu()  # pylint: disable=protected-access
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument(
        "--cta-threads", type=int, choices=(32, 64, 128, 256, 512), default=128)
    parser.add_argument(
        "--heap-cache", type=int, choices=(128, 256, 512, 1024), default=128)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model)
    decoder = ExactCtaDecoder(
        model, args.workspace, args.max_output, args.max_registers,
        args.cta_threads, args.heap_cache)
    runs = []
    decisions = None
    for _ in range(args.repeats):
        wall_start = time.perf_counter()
        decisions, stats = decoder.decode()
        stats["wall_ms"] = (time.perf_counter() - wall_start) * 1000
        runs.append(stats)
    assert decisions is not None
    mismatches = compare_with_reference(
        model, decisions, skip_workspace_overflow=args.allow_overflow)
    report = {
        "model": str(args.model),
        "metadata": model.metadata,
        "runs": runs,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:20],
    }
    print(json.dumps(report, indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
