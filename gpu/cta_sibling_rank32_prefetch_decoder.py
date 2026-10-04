"""Sibling-parallel rank32 decoder with exact monotonic stop prefetch.

This isolated variant changes only ``detector_cost_subwarp`` when ranked
detector rows are monotonic.  Each lane first loads its packed rank32 record;
lane zero reuses that register as the tile stop sentinel.  Continuing tiles
therefore avoid the base implementation's serialized lane-zero load followed
by a duplicate coalesced load.  No error-indexed state is read before stop.

The complete original function remains selected byte for byte for
nonmonotonic rows.  Kernel ABI, search ordering, status codes, floating-point
operations, and all host-side buffers are inherited from the Surface winner.
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

import cta_sibling_rank32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


SUBWARP_START = "__device__ __noinline__ double detector_cost_subwarp("
SUBWARP_END = "__device__ __noinline__ double cached_parent_cost_subwarp("


PREFETCH_MONOTONIC_SUBWARP = r"""__device__ __noinline__ double detector_cost_subwarp(
    u32 detector, const u8 *error_counts,
    const blocker_stamp_t *base_stamp,
    const prefix_stamp_t *prefix_stamp, u32 prefix_begin, u32 prefix_limit,
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

    // The only speculative access is this packed, coalesced rank32 load.
    // Lane zero reuses its value for exactly the old monotonic stop test.
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
      u32 blocker_count;
      u32 prefix_position;
      blocker_and_prefix_values(
          base_stamp, prefix_stamp, error, blocker_count, prefix_position);
      if (lower < best_rank && blocker_count == 0 &&
          (prefix_limit == 0 || prefix_position == 0 ||
           prefix_position > prefix_limit)) {
        local_type = group * rank_stride + (u32)error_counts[error];
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
  (void)prefix_begin;
  return best_type == 0xffffffffU ? inf_value() : count_values[best_type];
}
"""


def _extract_original_subwarp(source: str) -> tuple[int, int, str]:
    if source.count(SUBWARP_START) != 1:
        raise RuntimeError("expected exactly one detector_cost_subwarp definition")
    if source.count(SUBWARP_END) != 1:
        raise RuntimeError("expected exactly one cached_parent_cost_subwarp definition")
    start = source.index(SUBWARP_START)
    end = source.index(SUBWARP_END, start)
    return start, end, source[start:end]


def _build_prefetch_kernel(source: str) -> str:
    """Select the prefetch scan only for monotonic ranked detector rows."""
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
        raise RuntimeError("original nonmonotonic function was not preserved")
    if transformed.count("base_lower = __shfl_sync(full, base_lower, 0)") != 1:
        raise RuntimeError("coalesced stop prefetch was not emitted exactly once")
    return transformed


CTA_KERNEL = _build_prefetch_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


@dataclass(frozen=True)
class ScanSimulation:
    best_rank: int
    best_type: int | None
    packed_load_order: tuple[int, ...]
    candidate_read_order: tuple[int, ...]
    processed_tiles: int
    stopped_tiles: int

    @property
    def packed_loads(self) -> int:
        return len(self.packed_load_order)


def monotonic_tile_scan_reference(
    entries: list[tuple[int, int | None, int, bool]],
    *,
    prefetch: bool,
    initial_best_rank: int = 0xFFFFFFFF,
) -> ScanSimulation:
    """Model old and prefetched schedules while applying identical decisions.

    Each entry is ``(lower_rank, actual_rank_or_none, type, eligible)``.
    Load order is explicit so tests can distinguish the scheduling change from
    the unchanged candidate evaluation and strict first-winner semantics.
    """
    if any(
        entries[index][0] > entries[index + 1][0]
        for index in range(len(entries) - 1)
    ):
        raise ValueError("lower ranks must be nondecreasing")
    for lower, rank, _candidate_type, _eligible in entries:
        if rank is not None and rank < lower:
            raise ValueError("actual rank must not be below its lower bound")

    best_rank = int(initial_best_rank)
    best_type: int | None = None
    packed_load_order: list[int] = []
    candidate_read_order: list[int] = []
    processed_tiles = 0
    stopped_tiles = 0
    for base_index in range(0, len(entries), 32):
        valid = min(32, len(entries) - base_index)
        if prefetch:
            packed_load_order.extend(range(base_index, base_index + valid))
        else:
            packed_load_order.append(base_index)
        if entries[base_index][0] >= best_rank:
            stopped_tiles += 1
            break
        if not prefetch:
            packed_load_order.extend(range(base_index, base_index + valid))
        processed_tiles += 1
        candidates: list[tuple[int, int, int]] = []
        for position in range(base_index, base_index + valid):
            candidate_read_order.append(position)
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
        packed_load_order=tuple(packed_load_order),
        candidate_read_order=tuple(candidate_read_order),
        processed_tiles=processed_tiles,
        stopped_tiles=stopped_tiles,
    )


def nvrtc_compile_check(*, monotonic: bool = True) -> object:
    """Compile the complete SM75 Surface translation unit without launching."""
    options = (
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
    return cp.cuda.compiler.compile_using_nvrtc(
        COMMON + CTA_KERNEL,
        options=options,
        arch="75",
        filename="cta_sibling_rank32_prefetch_decoder.cu",
        name_expressions=("decode_exact_cta",),
        cache_in_memory=True,
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Surface winner with only the monotonic subwarp schedule changed."""

    # Subclasses may provide an audited kernel specialization without mutating
    # this module's globals.  Keeping the source on the class makes concurrent
    # decoder construction deterministic and thread safe.
    common_source = COMMON
    kernel_source = CTA_KERNEL

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, stamp_layout: str = "packed"):
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.stamp_layout = str(stamp_layout)
        self.visited_capacity = base._next_power_of_two(  # pylint: disable=protected-access
            2 * self.workspace_capacity
        )
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.workspace_capacity >= np.iinfo(np.uint16).max:
            raise ValueError(
                "compact blocker counts require workspace_capacity < 65535")
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("compact state requires max_output < 65535")
        if self.cta_threads not in (32, 64, 128, 256, 512):
            raise ValueError("cta_threads must be one of 32, 64, 128, 256, 512")
        if self.stamp_layout not in ("separate", "packed"):
            raise ValueError("stamp_layout must be 'separate' or 'packed'")
        row_lengths = np.diff(model.d2e_offsets.astype(np.uint64, copy=False))
        max_row_length = int(row_lengths.max(initial=0))
        if max_row_length >= np.iinfo(np.uint16).max:
            raise ValueError(
                "row-relative prefix positions require every d2e row "
                "to contain fewer than 65535 errors; maximum is "
                f"{max_row_length}")
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
            "-DTESSERACT_MONOTONIC_RANK_ROWS="
            + ("1" if model.rank_rows_monotonic else "0"),
            f"-DTESSERACT_PACKED_STAMPS={int(self.stamp_layout == 'packed')}",
            "-DTESSERACT_RANK32_ERROR_MASK="
            f"0x{self.rank32_layout.error_mask:08x}U",
            "-DTESSERACT_RANK32_GROUP_MASK="
            f"0x{self.rank32_layout.group_mask:08x}U",
            "-DTESSERACT_RANK32_LOWER_MASK="
            f"0x{self.rank32_layout.lower_mask:08x}U",
            "-DTESSERACT_RANK32_GROUP_SHIFT="
            f"{self.rank32_layout.group_shift}",
            "-DTESSERACT_RANK32_LOWER_SHIFT="
            f"{self.rank32_layout.lower_shift}",
        ))
        self.module = cp.RawModule(
            code=self.common_source + self.kernel_source,
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
    import argparse

    base._pin_one_cpu()  # pylint: disable=protected-access
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument(
        "--cta-threads", type=int, choices=(32, 64, 128, 256, 512), default=128)
    parser.add_argument(
        "--stamp-layout", choices=("separate", "packed"), default="packed")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model)
    decoder = ExactCtaDecoder(
        model, args.workspace, args.max_output, args.max_registers,
        args.cta_threads, args.stamp_layout)
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
