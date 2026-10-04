"""Exact fused rank32 CTA decoder with a parallel nonmonotonic rank fold.

Only the implementation of the ordered rank-row fold changes from
``cta_compact_fused_rank32_decoder``.  Each warp computes the same first stopping
position and strict minimum rank with an exclusive prefix minimum, a ballot,
and a stable first-lane reduction.  The monotonic path, kernel ABI, search,
floating-point updates, tie rules, budgets, and statuses are inherited
unchanged.
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

import cta_compact_fused_rank32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


PREFIX_FOLD_HELPER = r"""
struct PrefixFoldResult {
  u32 best_rank;
  u32 best_type;
  u32 stopped;
};

// Reproduce the scalar left-to-right fold for one warp without serial source
// shuffles.  local_rank is UINT32_MAX for blocked/invalid rank candidates and
// valid is false for lanes beyond the row end.  A stop lane itself never
// contributes a candidate, matching the scalar lower-bound check before its
// strict-rank update.
__device__ __forceinline__ PrefixFoldResult prefix_fold_nonmonotonic_warp(
    u32 local_lower, u32 local_rank, u32 local_type, bool valid,
    u32 entering_best_rank, u32 entering_best_type) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);

  // Inclusive minimum of candidate ranks, then shift once to obtain the
  // exclusive minimum needed by the lower-bound test at this lane.
  u32 inclusive_rank = local_rank;
#pragma unroll
  for (int offset = 1; offset < 32; offset <<= 1) {
    const u32 prior = __shfl_up_sync(full, inclusive_rank, offset);
    if (lane >= offset && prior < inclusive_rank) inclusive_rank = prior;
  }
  u32 exclusive_best = entering_best_rank;
  const u32 prior_rank = __shfl_up_sync(full, inclusive_rank, 1);
  if (lane != 0 && prior_rank < exclusive_best) exclusive_best = prior_rank;

  const u32 stop_mask = __ballot_sync(
      full, valid && local_lower >= exclusive_best);
  const int stop_lane = stop_mask == 0U ? 32 : __ffs((int)stop_mask) - 1;
  const bool participates = valid && lane < stop_lane;

  // Only lanes strictly before the first stop may update the running rank.
  u32 reduced_rank = participates ? local_rank : 0xffffffffU;
#pragma unroll
  for (int offset = 16; offset != 0; offset >>= 1) {
    const u32 other = __shfl_down_sync(full, reduced_rank, offset);
    if (other < reduced_rank) reduced_rank = other;
  }
  const u32 chunk_rank = __shfl_sync(full, reduced_rank, 0);

  u32 next_rank = entering_best_rank;
  u32 next_type = entering_best_type;
  if (chunk_rank < entering_best_rank) {
    const u32 winner_mask = __ballot_sync(
        full, participates && local_rank == chunk_rank);
    const int winner_lane = __ffs((int)winner_mask) - 1;
    next_rank = chunk_rank;
    next_type = __shfl_sync(full, local_type, winner_lane);
  }
  return PrefixFoldResult{next_rank, next_type, stop_mask != 0U};
}

"""


CTA_NONMONOTONIC = r"""#else
    // Exact lower ranks can decrease in the CPU row's rounded-cost order.
    // Load all candidates in parallel.  Warps remain ordered because the
    // entering best rank of warp w depends on every earlier row position, but
    // the 32 scalar source shuffles inside each warp are an exact prefix fold.
    const u64 position = base + (u64)tid;
    u32 local_lower = 0xffffffffU;
    u32 local_rank = 0xffffffffU;
    u32 local_type = 0xffffffffU;
    const bool local_valid = position < end;
    if (local_valid) {
      const u32 packed = d2e_ranked[position];
      const u32 error = rank32_error(packed);
      const u32 group = rank32_group(packed);
      local_lower = rank32_lower(packed);
      if (local_lower < *shared_best_rank) {
        const u32 state = error_state[error];
        const u32 prefix_position = prefix_position_value(state);
        const bool blocked = blocker_count_value(state) != 0 ||
            (prefix_limit != 0 && prefix_position != 0 &&
             prefix_position <= prefix_limit);
        if (!blocked) {
          local_type =
              group * rank_stride + error_count_value(state, count_shift);
          const u32 rank = count_ranks16[local_type];
          if (rank != 65535U) local_rank = rank;
        }
      }
    }
    if (tid == 0) *shared_stop = 0;
    __syncthreads();
#pragma unroll
    for (int ordered_warp = 0; ordered_warp < CTA_WARPS; ++ordered_warp) {
      if (warp == ordered_warp) {
        const PrefixFoldResult folded = prefix_fold_nonmonotonic_warp(
            local_lower, local_rank, local_type, local_valid,
            *shared_best_rank, *shared_best_type);
        if (lane == 0) {
          *shared_best_rank = folded.best_rank;
          *shared_best_type = folded.best_type;
          if (folded.stopped) *shared_stop = 1;
        }
      }
      __syncthreads();
      if (*shared_stop) break;
    }
    if (*shared_stop) break;
#endif"""


SUBWARP_NONMONOTONIC = r"""#else
    const PrefixFoldResult folded = prefix_fold_nonmonotonic_warp(
        local_lower, local_rank, local_type, position < end,
        best_rank, best_type);
    best_rank = folded.best_rank;
    best_type = folded.best_type;
    if (folded.stopped) break;
#endif"""


def _replace_preprocessor_else(
    source: str, *, function_marker: str, body_marker: str, replacement: str
) -> str:
    """Replace exactly one #else...#endif arm selected by stable markers."""
    function = source.index(function_marker)
    body = source.index(body_marker, function)
    start = source.rfind("#else", function, body)
    if start < function:
        raise RuntimeError(f"missing #else before {body_marker!r}")
    end = source.index("#endif", body) + len("#endif")
    nested = source.find("#if", start + len("#else"), end)
    if nested != -1:
        raise RuntimeError("prefix-fold source arm unexpectedly contains nested #if")
    return source[:start] + replacement + source[end:]


def _build_prefixfold_kernel(source: str) -> str:
    """Apply two audited rank-fold substitutions to the rank32 kernel."""
    helper_marker = "// Reduce a (rank,row-position) key across all four warps."
    if source.count(helper_marker) != 1:
        raise RuntimeError("rank32 kernel helper insertion marker changed")
    transformed = source.replace(
        helper_marker, PREFIX_FOLD_HELPER + helper_marker, 1
    )
    transformed = _replace_preprocessor_else(
        transformed,
        function_marker="double detector_cost_cta(",
        body_marker="// Exact lower ranks can decrease",
        replacement=CTA_NONMONOTONIC,
    )
    transformed = _replace_preprocessor_else(
        transformed,
        function_marker="double detector_cost_subwarp(",
        body_marker="u32 chunk_stop = 0;",
        replacement=SUBWARP_NONMONOTONIC,
    )
    if transformed.count("prefix_fold_nonmonotonic_warp(") != 3:
        raise RuntimeError("expected one prefix-fold definition and two call sites")
    if "for (int source = 0; source < 32; ++source)" in transformed:
        raise RuntimeError("serial nonmonotonic source fold survived transformation")
    return transformed


CTA_KERNEL = _build_prefixfold_kernel(base.CTA_KERNEL)
COMMON = base.COMMON


class ExactCtaDecoder(base.ExactCtaDecoder):
    """Fused rank32 decoder whose nonmonotonic warp fold uses prefix scans."""

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        # Mirror fused base.ExactCtaDecoder.__init__, changing only the kernel text.
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
        stats["nonmonotonic_rank_fold"] = "warp_prefix_min"
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
