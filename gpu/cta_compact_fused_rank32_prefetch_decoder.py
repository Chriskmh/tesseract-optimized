"""Exact fused-rank32 CTA decoder with coalesced subwarp stop prefetch.

Only the monotonic ``detector_cost_subwarp`` load schedule differs from the
fused-rank32 base.  Every lane first loads its packed incidence record.  Lane
zero extracts the tile lower bound from that already loaded record and
broadcasts the stop decision.  A continuing tile therefore removes the old
serial lane-zero load and its duplicate load in the coalesced warp access.
No error state, rank table, or floating-point value is touched before stop.
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

import cta_compact_fused_rank32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


SUBWARP_START = "__device__ __noinline__ double detector_cost_subwarp("
SUBWARP_END = "__device__ __forceinline__ bool transition_parent_counts_cta("


PREFETCH_MONOTONIC_SUBWARP = r"""__device__ __noinline__ double detector_cost_subwarp(
    u32 detector, const u32 *error_state, u32 count_shift, u32 prefix_limit,
    const u64 *d2e_offsets, const u32 *d2e_ranked,
    const u16 *count_ranks16, const double *count_values, u32 rank_stride) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  const u64 begin = d2e_offsets[detector];
  const u64 end = d2e_offsets[detector + 1];
  u32 best_rank = 0xffffffffU;
  u32 best_type = 0xffffffffU;
  for (u64 base = begin; base < end; base += 32) {
    const u64 position = base + (u64)lane;

    // Load the record once, coalesced across the warp.  Lane zero's record is
    // also the monotonic lower-bound sentinel for this tile.  No error-indexed
    // state or rank data is read until the broadcast stop test has passed.
    const u32 packed = position < end ? d2e_ranked[position] : 0U;
    u32 base_lower = lane == 0 ? rank32_lower(packed) : 0U;
    base_lower = __shfl_sync(full, base_lower, 0);
    if (base_lower >= best_rank) break;

    u32 local_type = 0xffffffffU;
    u32 local_rank = 0xffffffffU;
    if (position < end) {
      const u32 error = rank32_error(packed);
      const u32 group = rank32_group(packed);
      const u32 lower = rank32_lower(packed);
      const u32 state = error_state[error];
      const u32 prefix_position = prefix_position_value(state);
      if (lower < best_rank && blocker_count_value(state) == 0 &&
          (prefix_limit == 0 || prefix_position == 0 ||
           prefix_position > prefix_limit)) {
        local_type =
            group * rank_stride + error_count_value(state, count_shift);
        const u32 rank = count_ranks16[local_type];
        if (rank != 65535U) local_rank = rank;
      }
    }
    u32 reduced_rank = local_rank;
    for (int offset = 16; offset != 0; offset >>= 1) {
      const u32 other = __shfl_down_sync(full, reduced_rank, offset);
      if (other < reduced_rank) reduced_rank = other;
    }
    const u32 chunk_rank = __shfl_sync(full, reduced_rank, 0);
    if (chunk_rank != 0xffffffffU && chunk_rank < best_rank) {
      const u32 winners = __ballot_sync(full, local_rank == chunk_rank);
      const int winner_lane = __ffs(winners) - 1;
      const u32 type = __shfl_sync(full, local_type, winner_lane);
      best_rank = chunk_rank;
      best_type = type;
    }
  }
  return best_type == 0xffffffffU ? inf_value() : count_values[best_type];
}
"""


def _extract_original_subwarp(source: str) -> tuple[int, int, str]:
    if source.count(SUBWARP_START) != 1:
        raise RuntimeError("expected exactly one detector_cost_subwarp definition")
    if source.count(SUBWARP_END) != 1:
        raise RuntimeError("expected exactly one transition_parent_counts_cta definition")
    start = source.index(SUBWARP_START)
    end = source.index(SUBWARP_END, start)
    return start, end, source[start:end]


def _build_prefetch_kernel(source: str) -> str:
    """Select the prefetch function only for monotonic ranked rows."""
    start, end, original = _extract_original_subwarp(source)
    replacement = (
        "#if TESSERACT_MONOTONIC_RANK_ROWS\n"
        + PREFETCH_MONOTONIC_SUBWARP
        + "#else\n"
        + original
        + "#endif\n\n"
    )
    transformed = source[:start] + replacement + source[end:]
    if transformed.count(original) != 1:
        raise RuntimeError("original nonmonotonic subwarp function was not preserved")
    if transformed.count("base_lower = __shfl_sync(full, base_lower, 0)") != 1:
        raise RuntimeError("coalesced stop prefetch was not emitted exactly once")
    return transformed


CTA_KERNEL = _build_prefetch_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


@dataclass(frozen=True)
class ScanSimulation:
    best_rank: int
    best_type: int | None
    packed_loads: int
    processed_tiles: int
    stopped_tiles: int


def monotonic_tile_scan_reference(
    entries: list[tuple[int, int | None, int, bool]],
    *,
    prefetch: bool,
    initial_best_rank: int = 0xffffffff,
) -> ScanSimulation:
    """Model the old and prefetched load schedules with exact decisions.

    Entries are ``(lower_rank, actual_rank_or_none, type, eligible)``.  The
    packed load count distinguishes schedules; every decision field must be
    identical for a valid monotonic row.
    """
    if any(entries[index][0] > entries[index + 1][0]
           for index in range(len(entries) - 1)):
        raise ValueError("lower ranks must be nondecreasing")
    for lower, rank, _candidate_type, _eligible in entries:
        if rank is not None and rank < lower:
            raise ValueError("actual rank must not be below its lower bound")

    best_rank = int(initial_best_rank)
    best_type: int | None = None
    packed_loads = 0
    processed_tiles = 0
    stopped_tiles = 0
    for base_index in range(0, len(entries), 32):
        valid = min(32, len(entries) - base_index)
        # Old: one serialized base load, then all valid lane loads if the tile
        # continues.  New: all valid lane loads before the stop decision.
        packed_loads += valid if prefetch else 1
        if entries[base_index][0] >= best_rank:
            stopped_tiles += 1
            break
        if not prefetch:
            packed_loads += valid
        processed_tiles += 1
        candidates: list[tuple[int, int, int]] = []
        for position in range(base_index, base_index + valid):
            lower, rank, candidate_type, eligible = entries[position]
            if lower < best_rank and eligible and rank is not None:
                candidates.append((rank, position, candidate_type))
        if candidates:
            rank, _position, candidate_type = min(candidates)
            if rank < best_rank:
                best_rank = rank
                best_type = candidate_type
    return ScanSimulation(
        best_rank=best_rank,
        best_type=best_type,
        packed_loads=packed_loads,
        processed_tiles=processed_tiles,
        stopped_tiles=stopped_tiles,
    )


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    """Compile the complete SM75 translation unit without launching a kernel."""
    options = (
        "--std=c++11", "--fmad=false", "--prec-div=true",
        "--prec-sqrt=true", "--ftz=false",
        "-DTESSERACT_CTA_THREADS=512",
        "-DTESSERACT_CTA_HEAP_CACHE=1024",
        f"-DTESSERACT_MONOTONIC_RANK_ROWS={1 if monotonic else 0}",
        "-DTESSERACT_RANK32_ERROR_MASK=0x0003ffffU",
        "-DTESSERACT_RANK32_GROUP_MASK=0x0000001fU",
        "-DTESSERACT_RANK32_LOWER_MASK=0x0000007fU",
        "-DTESSERACT_RANK32_GROUP_SHIFT=18",
        "-DTESSERACT_RANK32_LOWER_SHIFT=23",
    )
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=options,
        arch="75",
        filename="cta_compact_fused_rank32_prefetch_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Fused-rank32 decoder with only the monotonic load schedule changed."""

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
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
        if not np.array_equal(
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
            code=COMMON + CTA_KERNEL,
            options=tuple(options),
            name_expressions=("decode_exact_cta",),
            backend="nvrtc",
        )
        self.kernel = self.module.get_function("decode_exact_cta")
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
        self.workspace = self._allocate_workspace()

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["subwarp_stop_prefetch"] = bool(self.model.rank_rows_monotonic)
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
