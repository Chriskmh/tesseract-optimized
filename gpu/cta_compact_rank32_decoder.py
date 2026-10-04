"""Exact rank32 packed-state CTA prototype.

This experiment keeps all ordering-sensitive search operations on thread zero
of a 128-thread CTA.  The complete CTA cooperates on the two graph operations
that dominate BB instances: ranked detector-cost scans and count/blocker row
updates.  A workspace exhaustion is reported as status 3 and must be replayed
by the CPU; it is never converted into an algorithmic low-confidence result.
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

from gpu_decoder import HostModel, KERNEL, compare_with_reference


# Reuse the bit-exact heap, state comparison, and visited-table primitives.
# The serial entry point is deliberately omitted so this translation unit has
# only the CTA kernel below.
COMMON = KERNEL.split('extern "C" __global__ void decode_exact(', 1)[0]


def _bits_for_inclusive_maximum(maximum: int) -> int:
    if maximum < 0:
        raise ValueError("rank32 field maximum must be nonnegative")
    return max(1, int(maximum).bit_length())


@dataclass(frozen=True)
class Rank32Layout:
    """Bit allocation for one ranked detector incidence."""

    error_bits: int
    group_bits: int
    lower_bits: int

    def __post_init__(self) -> None:
        if min(self.error_bits, self.group_bits, self.lower_bits) < 1:
            raise ValueError("rank32 fields must each use at least one bit")
        if self.total_bits > 32:
            raise ValueError(
                "ranked detector incidence requires "
                f"{self.total_bits} bits, exceeding uint32"
            )

    @property
    def group_shift(self) -> int:
        return self.error_bits

    @property
    def lower_shift(self) -> int:
        return self.error_bits + self.group_bits

    @property
    def total_bits(self) -> int:
        return self.error_bits + self.group_bits + self.lower_bits

    @property
    def error_mask(self) -> int:
        return (1 << self.error_bits) - 1

    @property
    def group_mask(self) -> int:
        return (1 << self.group_bits) - 1

    @property
    def lower_mask(self) -> int:
        return (1 << self.lower_bits) - 1

    def pack_scalar(self, error: int, group: int, lower: int) -> np.uint32:
        fields = (
            ("error", int(error), self.error_mask),
            ("group", int(group), self.group_mask),
            ("lower", int(lower), self.lower_mask),
        )
        for name, value, mask in fields:
            if value < 0 or value > mask:
                raise ValueError(f"{name}={value} does not fit its rank32 field")
        return np.uint32(
            int(error)
            | (int(group) << self.group_shift)
            | (int(lower) << self.lower_shift)
        )

    def unpack_scalar(self, packed: int) -> tuple[int, int, int]:
        value = int(packed)
        return (
            value & self.error_mask,
            (value >> self.group_shift) & self.group_mask,
            (value >> self.lower_shift) & self.lower_mask,
        )


def rank32_layout_for_bounds(
    *, num_errors: int, rank_groups: int, rank_maximum: int
) -> Rank32Layout:
    """Choose widths from validated model bounds, rejecting >32-bit models."""
    if num_errors < 1:
        raise ValueError("num_errors must be positive")
    if rank_groups < 1:
        raise ValueError("rank_groups must be positive")
    return Rank32Layout(
        error_bits=_bits_for_inclusive_maximum(num_errors - 1),
        group_bits=_bits_for_inclusive_maximum(rank_groups - 1),
        lower_bits=_bits_for_inclusive_maximum(rank_maximum),
    )


def repack_d2e_ranked32(model: HostModel) -> tuple[np.ndarray, Rank32Layout]:
    """Losslessly repack the exporter's u64 incidence words into uint32.

    The export format stores ``error`` in bits 0..31, ``rank group`` in bits
    32..39, and ``lower rank`` in bits 40..55. Row offsets and entry order are
    unchanged, preserving first-winner ties and ordered nonmonotonic exits.
    """
    metadata = model.metadata
    num_errors = int(metadata["num_errors"])
    rank_groups = int(metadata["rank_groups"])
    rank_maximum = int(metadata["rank_maximum"])
    rank_stride = int(metadata["rank_stride"])
    if num_errors > 1 << 32 or rank_groups > 1 << 8 or rank_maximum >= 1 << 16:
        raise ValueError("model bounds exceed the validated u64 export fields")
    count_ranks16 = np.asarray(model.count_ranks16)
    if count_ranks16.dtype != np.dtype(np.uint16):
        raise ValueError("rank32 path requires the validated uint16 rank table")
    if len(count_ranks16) != rank_groups * rank_stride:
        raise ValueError("count_ranks16 does not span every rank group and count")
    layout = rank32_layout_for_bounds(
        num_errors=num_errors,
        rank_groups=rank_groups,
        rank_maximum=rank_maximum,
    )
    source = np.asarray(model.d2e_ranked)
    if source.dtype != np.dtype(np.uint64):
        raise ValueError("d2e_ranked must use the validated uint64 export format")
    if source.ndim != 1:
        raise ValueError("d2e_ranked must be one-dimensional")
    if len(source) != len(model.d2e):
        raise ValueError("d2e_ranked length must equal the detector incidence count")
    if len(model.d2e_offsets) == 0 or int(model.d2e_offsets[-1]) != len(source):
        raise ValueError("d2e offsets do not span the ranked incidence table")
    if np.any(source >> np.uint64(56)):
        raise ValueError("d2e_ranked uses bits outside the validated export format")

    errors64 = source & np.uint64(0xFFFFFFFF)
    groups64 = (source >> np.uint64(32)) & np.uint64(0xFF)
    lowers64 = (source >> np.uint64(40)) & np.uint64(0xFFFF)
    if np.any(errors64 >= np.uint64(num_errors)):
        raise ValueError("d2e_ranked contains an out-of-range error id")
    if np.any(groups64 >= np.uint64(rank_groups)):
        raise ValueError("d2e_ranked contains an out-of-range rank group")
    if np.any(lowers64 > np.uint64(rank_maximum)):
        raise ValueError("d2e_ranked contains a lower rank above rank_maximum")

    packed = errors64.astype(np.uint32)
    packed |= groups64.astype(np.uint32) << np.uint32(layout.group_shift)
    packed |= lowers64.astype(np.uint32) << np.uint32(layout.lower_shift)
    roundtrip_errors = packed & np.uint32(layout.error_mask)
    roundtrip_groups = (
        packed >> np.uint32(layout.group_shift)
    ) & np.uint32(layout.group_mask)
    roundtrip_lowers = (
        packed >> np.uint32(layout.lower_shift)
    ) & np.uint32(layout.lower_mask)
    if not (
        np.array_equal(roundtrip_errors.astype(np.uint64), errors64)
        and np.array_equal(roundtrip_groups.astype(np.uint64), groups64)
        and np.array_equal(roundtrip_lowers.astype(np.uint64), lowers64)
    ):
        raise AssertionError("rank32 packing failed its lossless roundtrip")
    return np.ascontiguousarray(packed), layout


CTA_KERNEL = r"""
#ifndef TESSERACT_CTA_THREADS
#define TESSERACT_CTA_THREADS 128
#endif
#ifndef TESSERACT_MONOTONIC_RANK_ROWS
#define TESSERACT_MONOTONIC_RANK_ROWS 0
#endif
#ifndef TESSERACT_RANK32_ERROR_MASK
#error TESSERACT_RANK32_ERROR_MASK must be supplied by the host
#endif
#ifndef TESSERACT_RANK32_GROUP_MASK
#error TESSERACT_RANK32_GROUP_MASK must be supplied by the host
#endif
#ifndef TESSERACT_RANK32_LOWER_MASK
#error TESSERACT_RANK32_LOWER_MASK must be supplied by the host
#endif
#ifndef TESSERACT_RANK32_GROUP_SHIFT
#error TESSERACT_RANK32_GROUP_SHIFT must be supplied by the host
#endif
#ifndef TESSERACT_RANK32_LOWER_SHIFT
#error TESSERACT_RANK32_LOWER_SHIFT must be supplied by the host
#endif
constexpr int CTA_THREADS = TESSERACT_CTA_THREADS;
constexpr int CTA_WARPS = CTA_THREADS / 32;
constexpr int MAX_AFFECTED_DETECTORS = 512;
#ifndef TESSERACT_CTA_HEAP_CACHE
#define TESSERACT_CTA_HEAP_CACHE 128
#endif
constexpr int CTA_HEAP_CACHE = TESSERACT_CTA_HEAP_CACHE;
constexpr u32 PARENT_CACHE_CLAIM_BIT = 0x80000000U;
constexpr u32 RANK32_ERROR_MASK = TESSERACT_RANK32_ERROR_MASK;
constexpr u32 RANK32_GROUP_MASK = TESSERACT_RANK32_GROUP_MASK;
constexpr u32 RANK32_LOWER_MASK = TESSERACT_RANK32_LOWER_MASK;
constexpr int RANK32_GROUP_SHIFT = TESSERACT_RANK32_GROUP_SHIFT;
constexpr int RANK32_LOWER_SHIFT = TESSERACT_RANK32_LOWER_SHIFT;

__device__ __forceinline__ u32 rank32_error(u32 packed) {
  return packed & RANK32_ERROR_MASK;
}

__device__ __forceinline__ u32 rank32_group(u32 packed) {
  return (packed >> RANK32_GROUP_SHIFT) & RANK32_GROUP_MASK;
}

__device__ __forceinline__ u32 rank32_lower(u32 packed) {
  return (packed >> RANK32_LOWER_SHIFT) & RANK32_LOWER_MASK;
}

// One random 32-bit word per error: the high half is the active-chain
// blocker reference count, and the low half is the one-based position in the
// current minimum-detector branch.  The host bounds both halves below 65535.
__device__ __forceinline__ void blocker_and_prefix_values(
    const u32 *compact_state, u32 error,
    u32 &blocker_count, u32 &prefix_position) {
  const u32 value = compact_state[error];
  blocker_count = value >> 16;
  prefix_position = value & 0xffffU;
}

__device__ __forceinline__ u32 blocker_count_value(
    const u32 *compact_state, u32 error) {
  return compact_state[error] >> 16;
}

__device__ __forceinline__ void update_blocker_count(
    u32 *compact_state, u32 error, bool undo) {
  const u32 value = compact_state[error];
  const u32 count = value >> 16;
  const u32 next = undo ? count - 1 : count + 1;
  compact_state[error] = (value & 0xffffU) | (next << 16);
}

__device__ __forceinline__ void set_prefix_position(
    u32 *compact_state, u32 error, u16 position) {
  const u32 value = compact_state[error];
  compact_state[error] = (value & 0xffff0000U) | (u32)position;
}

__device__ __forceinline__ void clear_prefix_position(
    u32 *compact_state, u32 error) {
  compact_state[error] &= 0xffff0000U;
}

struct CtaCompactNode {
  double cost;
  u32 num_dets;
  int chain;
};

__device__ __forceinline__ bool cta_compact_greater(
    const CtaCompactNode &a, const CtaCompactNode &b) {
  return a.cost > b.cost || (a.cost == b.cost && a.num_dets < b.num_dets);
}

__device__ __forceinline__ CtaCompactNode cta_compact_load(
    int index, const double *cost, const u32 *num_dets, const int *chain,
    const double *shared_cost, const u32 *shared_num_dets, const int *shared_chain) {
  if (index < CTA_HEAP_CACHE) {
    return CtaCompactNode{
        shared_cost[index], shared_num_dets[index], shared_chain[index]};
  }
  return CtaCompactNode{cost[index], num_dets[index], chain[index]};
}

__device__ __forceinline__ void cta_compact_store(
    int index, const CtaCompactNode &value,
    double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  if (index < CTA_HEAP_CACHE) {
    shared_cost[index] = value.cost;
    shared_num_dets[index] = value.num_dets;
    shared_chain[index] = value.chain;
  } else {
    cost[index] = value.cost;
    num_dets[index] = value.num_dets;
    chain[index] = value.chain;
  }
}

__device__ __noinline__ void cta_compact_push(
    int &length, const CtaCompactNode &input,
    double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  int hole = length++;
  CtaCompactNode value = input;
  int parent = (hole - 1) / 2;
  while (hole > 0) {
    const CtaCompactNode parent_value = cta_compact_load(
        parent, cost, num_dets, chain,
        shared_cost, shared_num_dets, shared_chain);
    if (!cta_compact_greater(parent_value, value)) break;
    cta_compact_store(hole, parent_value, cost, num_dets, chain,
                      shared_cost, shared_num_dets, shared_chain);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  cta_compact_store(hole, value, cost, num_dets, chain,
                    shared_cost, shared_num_dets, shared_chain);
}

__device__ __noinline__ CtaCompactNode cta_compact_pop(
    int &length, double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  const CtaCompactNode result = cta_compact_load(
      0, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
  CtaCompactNode value = cta_compact_load(
      length - 1, cost, num_dets, chain,
      shared_cost, shared_num_dets, shared_chain);
  const int new_length = --length;
  if (new_length == 0) return result;
  int hole = 0;
  int second_child = 0;
  while (second_child < (new_length - 1) / 2) {
    second_child = 2 * (second_child + 1);
    const CtaCompactNode right = cta_compact_load(
        second_child, cost, num_dets, chain,
        shared_cost, shared_num_dets, shared_chain);
    const CtaCompactNode left = cta_compact_load(
        second_child - 1, cost, num_dets, chain,
        shared_cost, shared_num_dets, shared_chain);
    const bool choose_left = cta_compact_greater(right, left);
    const CtaCompactNode selected = choose_left ? left : right;
    if (choose_left) --second_child;
    cta_compact_store(
        hole, selected,
        cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    hole = second_child;
  }
  if ((new_length & 1) == 0 && second_child == (new_length - 2) / 2) {
    second_child = 2 * (second_child + 1);
    cta_compact_store(
        hole,
        cta_compact_load(second_child - 1, cost, num_dets, chain,
                         shared_cost, shared_num_dets, shared_chain),
        cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    hole = second_child - 1;
  }
  int parent = (hole - 1) / 2;
  while (hole > 0) {
    const CtaCompactNode parent_value = cta_compact_load(
        parent, cost, num_dets, chain,
        shared_cost, shared_num_dets, shared_chain);
    if (!cta_compact_greater(parent_value, value)) break;
    cta_compact_store(hole, parent_value, cost, num_dets, chain,
                      shared_cost, shared_num_dets, shared_chain);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  cta_compact_store(hole, value, cost, num_dets, chain,
                    shared_cost, shared_num_dets, shared_chain);
  return result;
}

struct CtaDebugRecorder {
  u64 *output;
  int shot;
  int tid;
  u64 expanded = 0;
  u64 branch_rows = 0;
  u64 base_pass = 0;
  u64 beam_pass = 0;
  u64 visited_pass = 0;
  u64 affected = 0;
  u64 parent_queries = 0;
  u64 child_queries = 0;
  __device__ CtaDebugRecorder(u64 *output, int shot, int tid)
      : output(output), shot(shot), tid(tid) {}
  __device__ ~CtaDebugRecorder() {
    if (tid != 0) return;
    u64 *row = output + (u64)shot * 8;
    row[0] = expanded;
    row[1] = branch_rows;
    row[2] = base_pass;
    row[3] = beam_pass;
    row[4] = visited_pass;
    row[5] = affected;
    row[6] = parent_queries;
    row[7] = child_queries;
  }
};

__device__ __forceinline__ bool update_count_row_cta(
    u32 detector, int delta, u8 *counts,
    const u64 *d2e_offsets, const u32 *d2e, u32 *shared_bad) {
  const int tid = (int)threadIdx.x;
  for (u64 row = d2e_offsets[detector] + tid;
       row < d2e_offsets[detector + 1]; row += CTA_THREADS) {
    const u32 error = d2e[row];
    counts[error] = (u8)((int)counts[error] + delta);
  }
  __syncthreads();
  (void)shared_bad;
  return true;
}

__device__ __forceinline__ void update_two_count_rows_cta(
    u32 detector, int delta, u8 *first, u8 *second,
    const u64 *d2e_offsets, const u32 *d2e) {
  const int tid = (int)threadIdx.x;
  for (u64 row = d2e_offsets[detector] + tid;
       row < d2e_offsets[detector + 1]; row += CTA_THREADS) {
    const u32 error = d2e[row];
    first[error] = (u8)((int)first[error] + delta);
    second[error] = (u8)((int)second[error] + delta);
  }
  __syncthreads();
}

__device__ __forceinline__ bool adjust_child_counts_cta(
    u32 error, bool apply_child, const u64 *parent_bits, u8 *counts,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *d2e_offsets, const u32 *d2e, u32 *shared_bad) {
  for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
    const u32 detector = edets[j];
    int delta = bit_is_set(parent_bits, detector) ? -1 : 1;
    if (!apply_child) delta = -delta;
    if (!update_count_row_cta(
            detector, delta, counts, d2e_offsets, d2e, shared_bad)) return false;
  }
  return true;
}

// Move directly between two sibling syndromes.  Their sorted supports share
// at least the branch detector; common support detectors stay toggled and do
// not need an undo followed by an apply row update.
__device__ __forceinline__ bool transition_child_counts_cta(
    u32 old_error, u32 new_error, const u64 *parent_bits, u8 *counts,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *d2e_offsets, const u32 *d2e, u32 *shared_bad) {
  u64 old_position = edet_offsets[old_error];
  const u64 old_end = edet_offsets[old_error + 1];
  u64 new_position = edet_offsets[new_error];
  const u64 new_end = edet_offsets[new_error + 1];
  while (old_position < old_end || new_position < new_end) {
    const u32 old_detector = old_position < old_end
        ? edets[old_position] : 0xffffffffU;
    const u32 new_detector = new_position < new_end
        ? edets[new_position] : 0xffffffffU;
    if (old_detector == new_detector) {
      ++old_position;
      ++new_position;
      continue;
    }
    u32 detector;
    int delta;
    if (old_detector < new_detector) {
      detector = old_detector;
      // Undo the old child toggle.
      delta = bit_is_set(parent_bits, detector) ? 1 : -1;
      ++old_position;
    } else {
      detector = new_detector;
      // Apply the new child toggle.
      delta = bit_is_set(parent_bits, detector) ? -1 : 1;
      ++new_position;
    }
    if (!update_count_row_cta(
            detector, delta, counts, d2e_offsets, d2e, shared_bad)) return false;
  }
  return true;
}

// Reduce a (rank,row-position) key across all four warps.  The row position is
// the second component, so equal ranks reproduce the CPU's first-winner rule.
__device__ __noinline__ double detector_cost_cta(
    u32 detector, const u8 *error_counts, const u32 *base_stamp,
    u32 base_generation, const u32 *prefix_stamp,
    u32 prefix_begin, u32 prefix_limit,
    const u64 *d2e_offsets, const u32 *d2e_ranked,
    const u16 *count_ranks16, const double *count_values, u32 rank_stride,
    u64 *warp_keys, u32 *shared_best_rank, u32 *shared_best_type,
    u32 *shared_stop) {
  const u32 full = 0xffffffffU;
  const int tid = (int)threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const u64 begin = d2e_offsets[detector];
  const u64 end = d2e_offsets[detector + 1];
  if (tid == 0) {
    *shared_best_rank = 0xffffffffU;
    *shared_best_type = 0xffffffffU;
  }
  __syncthreads();

  for (u64 base = begin; base < end; base += CTA_THREADS) {
#if TESSERACT_MONOTONIC_RANK_ROWS
    if (tid == 0) {
      *shared_stop = rank32_lower(d2e_ranked[base]) >= *shared_best_rank;
    }
    __syncthreads();
    if (*shared_stop) break;

    const u64 position = base + (u64)tid;
    u64 key = 0xffffffffffffffffULL;
    if (position < end) {
      const u32 packed = d2e_ranked[position];
      const u32 error = rank32_error(packed);
      const u32 group = rank32_group(packed);
      const u32 lower = rank32_lower(packed);
      if (lower < *shared_best_rank) {
        u32 blocker_count;
        u32 prefix_position;
        blocker_and_prefix_values(
            base_stamp, error, blocker_count, prefix_position);
        const bool blocked = blocker_count != 0 ||
            (prefix_limit != 0 && prefix_position != 0 &&
             prefix_position <= prefix_limit);
        if (!blocked) {
          const u32 type = group * rank_stride + (u32)error_counts[error];
          const u32 rank = count_ranks16[type];
          if (rank != 65535U) {
            key = ((u64)rank << 32) | (u32)(position - begin);
          }
        }
      }
    }
    for (int offset = 16; offset != 0; offset >>= 1) {
      const u64 other = __shfl_down_sync(full, key, offset);
      if (other < key) key = other;
    }
    if (lane == 0) warp_keys[warp] = key;
    __syncthreads();
    if (tid == 0) {
      u64 winner = warp_keys[0];
#pragma unroll
      for (int w = 1; w < CTA_WARPS; ++w) {
        if (warp_keys[w] < winner) winner = warp_keys[w];
      }
      if (winner != 0xffffffffffffffffULL) {
        const u32 rank = (u32)(winner >> 32);
        if (rank < *shared_best_rank) {
          const u64 winner_position = begin + (u32)winner;
          const u32 packed = d2e_ranked[winner_position];
          const u32 error = rank32_error(packed);
          const u32 group = rank32_group(packed);
          *shared_best_rank = rank;
          *shared_best_type = group * rank_stride + (u32)error_counts[error];
        }
      }
    }
    __syncthreads();
#else
    // Exact lower ranks can decrease in the CPU row's rounded-cost order.
    // Load in parallel, then fold warps and lanes in exact row order.
    const u64 position = base + (u64)tid;
    u32 local_lower = 0xffffffffU;
    u32 local_rank = 0xffffffffU;
    u32 local_type = 0xffffffffU;
    if (position < end) {
      const u32 packed = d2e_ranked[position];
      const u32 error = rank32_error(packed);
      const u32 group = rank32_group(packed);
      local_lower = rank32_lower(packed);
      if (local_lower < *shared_best_rank) {
        u32 blocker_count;
        u32 prefix_position;
        blocker_and_prefix_values(
            base_stamp, error, blocker_count, prefix_position);
        const bool blocked = blocker_count != 0 ||
            (prefix_limit != 0 && prefix_position != 0 &&
             prefix_position <= prefix_limit);
        if (!blocked) {
          local_type = group * rank_stride + (u32)error_counts[error];
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
        u32 best_rank = *shared_best_rank;
        u32 best_type = *shared_best_type;
        u32 chunk_stop = 0;
#pragma unroll
        for (int source = 0; source < 32; ++source) {
          const u32 lower = __shfl_sync(full, local_lower, source);
          const u32 rank = __shfl_sync(full, local_rank, source);
          const u32 type = __shfl_sync(full, local_type, source);
          const u64 source_position =
              base + (u64)ordered_warp * 32U + (u64)source;
          if (!chunk_stop && source_position < end) {
            if (lower >= best_rank) {
              chunk_stop = 1;
            } else if (rank < best_rank) {
              best_rank = rank;
              best_type = type;
            }
          }
        }
        if (lane == 0) {
          *shared_best_rank = best_rank;
          *shared_best_type = best_type;
          if (chunk_stop) *shared_stop = 1;
        }
      }
      __syncthreads();
      if (*shared_stop) break;
    }
    if (*shared_stop) break;
#endif
  }
  const u32 type = *shared_best_type;
  (void)base_generation;
  return type == 0xffffffffU ? inf_value() : count_values[type];
}

// Independent warp-sized detector-cost query.  Multiple warps can evaluate
// different affected detectors, while thread zero later folds the returned
// binary64 values in the original CPU order.
__device__ __noinline__ double detector_cost_subwarp(
    u32 detector, const u8 *error_counts, const u32 *base_stamp,
    const u32 *prefix_stamp, u32 prefix_begin, u32 prefix_limit,
    const u64 *d2e_offsets, const u32 *d2e_ranked,
    const u16 *count_ranks16, const double *count_values, u32 rank_stride) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  const u64 begin = d2e_offsets[detector];
  const u64 end = d2e_offsets[detector + 1];
  u32 best_rank = 0xffffffffU;
  u32 best_type = 0xffffffffU;
  for (u64 base = begin; base < end; base += 32) {
#if TESSERACT_MONOTONIC_RANK_ROWS
    u32 stop = 0;
    if (lane == 0) stop = rank32_lower(d2e_ranked[base]) >= best_rank;
    stop = __shfl_sync(full, stop, 0);
    if (stop) break;
#endif
    const u64 position = base + (u64)lane;
    u32 local_type = 0xffffffffU;
    u32 local_rank = 0xffffffffU;
    u32 local_lower = 0xffffffffU;
    if (position < end) {
      const u32 packed = d2e_ranked[position];
      const u32 error = rank32_error(packed);
      const u32 group = rank32_group(packed);
      const u32 lower = rank32_lower(packed);
      local_lower = lower;
      u32 blocker_count;
      u32 prefix_position;
      blocker_and_prefix_values(
          base_stamp, error, blocker_count, prefix_position);
      if (lower < best_rank && blocker_count == 0 &&
          (prefix_limit == 0 || prefix_position == 0 ||
           prefix_position > prefix_limit)) {
        local_type = group * rank_stride + (u32)error_counts[error];
        const u32 rank = count_ranks16[local_type];
        if (rank != 65535U) local_rank = rank;
      }
    }
#if TESSERACT_MONOTONIC_RANK_ROWS
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
#else
    u32 chunk_stop = 0;
#pragma unroll
    for (int source = 0; source < 32; ++source) {
      const u32 lower = __shfl_sync(full, local_lower, source);
      const u32 rank = __shfl_sync(full, local_rank, source);
      const u32 type = __shfl_sync(full, local_type, source);
      if (!chunk_stop && base + (u64)source < end) {
        if (lower >= best_rank) {
          chunk_stop = 1;
        } else if (rank < best_rank) {
          best_rank = rank;
          best_type = type;
        }
      }
    }
    if (chunk_stop) break;
#endif
  }
  return best_type == 0xffffffffU ? inf_value() : count_values[best_type];
}

__device__ __forceinline__ bool transition_parent_counts_cta(
    const u64 *state_bits, u64 *counted_bits, int num_words,
    u8 *counts, u8 *mirror_counts,
    const u64 *d2e_offsets, const u32 *d2e,
    u32 *shared_detector, int *shared_delta, u32 *shared_bad) {
  const int tid = (int)threadIdx.x;
  while (true) {
    if (tid == 0) {
      *shared_detector = 0xffffffffU;
      for (int word = 0; word < num_words; ++word) {
        const u64 changed = state_bits[word] ^ counted_bits[word];
        if (changed != 0) {
          const u32 detector = (u32)(word * 64 + __ffsll((long long)changed) - 1);
          *shared_detector = detector;
          *shared_delta = bit_is_set(state_bits, detector) ? 1 : -1;
          toggle_bit(counted_bits, detector);
          break;
        }
      }
    }
    __syncthreads();
    if (*shared_detector == 0xffffffffU) return true;
    update_two_count_rows_cta(
        *shared_detector, *shared_delta, counts, mirror_counts,
        d2e_offsets, d2e);
  }
  (void)shared_bad;
}

__device__ __forceinline__ bool apply_chain_edge_cta(
    int chain_index, bool undo, u64 *bits, u32 *blocker_counts,
    const u32 *chain_error, const u32 *chain_min_detector, const u32 *chain_row,
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets, u32 *shared_bad) {
  const int tid = (int)threadIdx.x;
  const u32 error = chain_error[chain_index];
  const u32 min_detector = chain_min_detector[chain_index];
  const u64 begin = d2e_offsets[min_detector];
  const u64 selected = chain_row[chain_index];
  for (u64 row = begin + tid; row <= selected; row += CTA_THREADS) {
    const u32 blocked_error = d2e[row];
    update_blocker_count(blocker_counts, blocked_error, undo);
  }
  __syncthreads();
  if (tid == 0) {
    for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
      toggle_bit(bits, edets[j]);
    }
  }
  __syncthreads();
  (void)shared_bad;
  return true;
}

__device__ __noinline__ bool state_equals_virtual_child_cta(
    const u64 *parent_bits, u32 error, int existing_chain,
    const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u64 *edet_offsets, const u32 *edets) {
  for (int word = 0; word < num_words; ++word) scratch[word] = initial_bits[word];
  int index = existing_chain;
  while (index != -1) {
    const u32 chain_error_index = chain_error[index];
    for (u64 j = edet_offsets[chain_error_index];
         j < edet_offsets[chain_error_index + 1]; ++j) {
      toggle_bit(scratch, edets[j]);
    }
    index = chain_parent[index];
  }
  for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
    toggle_bit(scratch, edets[j]);
  }
  for (int word = 0; word < num_words; ++word) {
    if (parent_bits[word] != scratch[word]) return false;
  }
  return true;
}

__device__ __noinline__ int visited_find_virtual_child_cta(
    u64 hash1, u64 hash2, const u64 *parent_bits, u32 error, int &empty_slot,
    const u64 *visited_hash1, const u64 *visited_hash2, const u32 *visited_chain,
    int table_capacity, const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u64 *edet_offsets, const u32 *edets) {
  const u64 mixed = hash1 ^ ((hash2 << 23) | (hash2 >> 41));
  int slot = (int)(mixed & (u64)(table_capacity - 1));
  for (int probe = 0; probe < table_capacity; ++probe) {
    const u32 encoded = visited_chain[slot];
    if (encoded == 0) {
      empty_slot = slot;
      return 0;
    }
    if (visited_hash1[slot] == hash1 && visited_hash2[slot] == hash2) {
      const int existing_chain = (int)encoded - 2;
      if (state_equals_virtual_child_cta(
              parent_bits, error, existing_chain, initial_bits, scratch, num_words,
              chain_error, chain_parent, edet_offsets, edets)) {
        empty_slot = slot;
        return 1;
      }
    }
    slot = (slot + 1) & (table_capacity - 1);
  }
  empty_slot = -1;
  return -1;
}

extern "C" __global__ __launch_bounds__(CTA_THREADS) void decode_exact_cta(
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *eneighbor_word_offsets, const u32 *eneighbor_words,
    const u64 *eneighbor_masks,
    const double *weights, const u32 *error_to_dem,
    const u32 *detector_order, const u64 *error_hash1, const u64 *error_hash2,
    const double *count_values, u32 rank_stride,
    const u32 *d2e_ranked, const u16 *count_ranks16,
    const u64 *shot_offsets, const u32 *shot_detections,
    const u64 *initial_shot_bits, const u64 *shot_hash1, const u64 *shot_hash2,
    int num_shots, int num_detectors, int num_errors, int num_words,
    u32 detector_beam, u32 pqlimit,
    int workspace_capacity, int visited_capacity, int max_output,
    double *heap_cost_all, u32 *heap_num_dets_all, u32 *heap_depth_all,
    int *heap_chain_all, u64 *heap_hash1_all, u64 *heap_hash2_all,
    u32 *chain_error_all, u32 *chain_min_detector_all, int *chain_parent_all,
    u32 *chain_row_all, u32 *chain_depth_all, int *path_stack_all,
    u8 *parent_counts_all, u8 *child_counts_all, u64 *counted_bits_all,
    u32 *base_stamp_all, u32 *prefix_stamp_all,
    double *parent_cost_cache_all, u32 *parent_cost_stamp_all,
    u64 *state_bits_all, u64 *child_bits_all, u64 *compare_bits_all,
    u64 *visited_hash1_all, u64 *visited_hash2_all, u32 *visited_chain_all,
    u32 *out_status, u32 *out_length, u32 *out_errors,
    u64 *out_cost_bits, u32 *out_pushes, u32 *out_pops, u64 *out_cycles,
    u64 *out_debug) {
  const int tid = (int)threadIdx.x;
  const int shot = (int)blockIdx.x;
  if (shot >= num_shots) return;
  CtaDebugRecorder debug(out_debug, shot, tid);
  const u64 cycle_start = clock64();

  __shared__ NodeValue s_node;
  __shared__ int s_heap_length;
  __shared__ int s_visited;
  __shared__ int s_empty_slot;
  __shared__ u32 s_min_detector;
  __shared__ u32 s_detector;
  __shared__ int s_delta;
  __shared__ u32 s_bad;
  __shared__ u64 s_warp_keys[CTA_WARPS];
  __shared__ u32 s_neighbor_masks[CTA_WARPS];
  __shared__ u32 s_neighbor_word;
  __shared__ u64 s_neighbor_bits;
  __shared__ u32 s_best_rank;
  __shared__ u32 s_best_type;
  __shared__ u32 s_stop;
  __shared__ double s_initial_cost;
  __shared__ double s_next_cost;
  __shared__ double s_value;
  __shared__ u32 s_cached;
  __shared__ u32 s_affected_count;
  __shared__ u32 s_query_cursor;
  __shared__ u32 s_affected_detector[MAX_AFFECTED_DETECTORS];
  __shared__ u32 s_affected_operation[MAX_AFFECTED_DETECTORS];
  __shared__ double s_parent_values[MAX_AFFECTED_DETECTORS];
  __shared__ double s_child_values[MAX_AFFECTED_DETECTORS];
  __shared__ double s_heap_cost[CTA_HEAP_CACHE];
  __shared__ u32 s_heap_num_dets[CTA_HEAP_CACHE];
  __shared__ int s_heap_chain[CTA_HEAP_CACHE];

  double *heap_cost = heap_cost_all + (u64)shot * workspace_capacity;
  u32 *heap_num_dets = heap_num_dets_all + (u64)shot * workspace_capacity;
  u32 *heap_depth = heap_depth_all + (u64)shot * workspace_capacity;
  int *heap_chain = heap_chain_all + (u64)shot * workspace_capacity;
  u64 *chain_hash1 = heap_hash1_all + (u64)shot * workspace_capacity;
  u64 *chain_hash2 = heap_hash2_all + (u64)shot * workspace_capacity;
  u32 *chain_error = chain_error_all + (u64)shot * workspace_capacity;
  u32 *chain_min_detector = chain_min_detector_all + (u64)shot * workspace_capacity;
  int *chain_parent = chain_parent_all + (u64)shot * workspace_capacity;
  u32 *chain_row = chain_row_all + (u64)shot * workspace_capacity;
  u32 *chain_depth = chain_depth_all + (u64)shot * workspace_capacity;
  int *path_stack = path_stack_all + (u64)shot * workspace_capacity;
  u8 *parent_counts = parent_counts_all + (u64)shot * num_errors;
  u8 *child_counts = child_counts_all + (u64)shot * num_errors;
  u64 *counted_bits = counted_bits_all + (u64)shot * num_words;
  u32 *base_stamp = base_stamp_all + (u64)shot * num_errors;
  // Packed state keeps the branch position in base_stamp's low half.  Keep
  // the second pointer only to preserve the launch ABI.
  u32 *prefix_stamp = prefix_stamp_all;
  double *parent_cost_cache = parent_cost_cache_all + (u64)shot * num_detectors;
  u32 *parent_cost_stamp = parent_cost_stamp_all + (u64)shot * num_detectors;
  u64 *state_bits = state_bits_all + (u64)shot * num_words;
  u64 *child_bits = child_bits_all + (u64)shot * num_words;
  u64 *compare_bits = compare_bits_all + (u64)shot * num_words;
  u64 *visited_hash1 = visited_hash1_all + (u64)shot * visited_capacity;
  u64 *visited_hash2 = visited_hash2_all + (u64)shot * visited_capacity;
  u32 *visited_chain = visited_chain_all + (u64)shot * visited_capacity;
  const u64 *initial_bits = initial_shot_bits + (u64)shot * num_words;

  if (tid == 0) {
    out_status[shot] = 0;
    out_length[shot] = 0;
    out_cost_bits[shot] = 0;
    out_pushes[shot] = 0;
    out_pops[shot] = 0;
  }
  const u64 shot_begin = shot_offsets[shot];
  const u64 shot_end = shot_offsets[shot + 1];
  const u32 initial_num_dets = (u32)(shot_end - shot_begin);
  if (initial_num_dets == 0) {
    if (tid == 0) {
      out_status[shot] = 1;
      out_cycles[shot] = clock64() - cycle_start;
    }
    return;
  }

  for (int word = tid; word < num_words; word += CTA_THREADS) {
    state_bits[word] = initial_bits[word];
    counted_bits[word] = initial_bits[word];
  }
  __syncthreads();
  for (u64 i = shot_begin; i < shot_end; ++i) {
    if (!update_count_row_cta(
            shot_detections[i], 1, parent_counts,
            d2e_offsets, d2e, &s_bad)) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }
  }
  for (int error = tid; error < num_errors; error += CTA_THREADS) {
    child_counts[error] = parent_counts[error];
  }
  __syncthreads();

  if (tid == 0) s_initial_cost = 0.0;
  __syncthreads();
  for (u64 i = shot_begin; i < shot_end; ++i) {
    const double value = detector_cost_cta(
        shot_detections[i], parent_counts, base_stamp, 1,
        prefix_stamp, 0, 0, d2e_offsets, d2e_ranked,
        count_ranks16, count_values, rank_stride,
        s_warp_keys, &s_best_rank, &s_best_type, &s_stop);
    if (tid == 0) s_initial_cost = __dadd_rn(s_initial_cost, value);
    __syncthreads();
  }
  if (s_initial_cost == inf_value()) {
    if (tid == 0) out_status[shot] = 2;
    return;
  }

  if (tid == 0) {
    s_heap_length = 0;
    CtaCompactNode root{s_initial_cost, initial_num_dets, -1};
    cta_compact_push(
        s_heap_length, root, heap_cost, heap_num_dets, heap_chain,
        s_heap_cost, s_heap_num_dets, s_heap_chain);
  }
  __syncthreads();
  int chain_count = 0;
  u32 num_pushed = 1;
  u32 pop_count = 0;
  u32 generation = 1;
  u32 min_num_dets = initial_num_dets;
  u32 max_num_dets = initial_num_dets + detector_beam;
  int current_chain = -1;
  u32 current_depth = 0;

  while (s_heap_length != 0) {
    if (tid == 0) {
      const CtaCompactNode compact = cta_compact_pop(
          s_heap_length, heap_cost, heap_num_dets, heap_chain,
          s_heap_cost, s_heap_num_dets, s_heap_chain);
      s_node.cost = compact.cost;
      s_node.num_dets = compact.num_dets;
      s_node.chain = compact.chain;
      if (compact.chain == -1) {
        s_node.depth = 0;
        s_node.hash1 = shot_hash1[shot];
        s_node.hash2 = shot_hash2[shot];
      } else {
        s_node.depth = chain_depth[compact.chain];
        s_node.hash1 = chain_hash1[compact.chain];
        s_node.hash2 = chain_hash2[compact.chain];
      }
    }
    __syncthreads();
    ++pop_count;
    const NodeValue node = s_node;
    if (node.num_dets > max_num_dets) continue;
    if (node.num_dets == 0) {
      if (tid == 0) {
        if ((int)node.depth > max_output) {
          out_status[shot] = 3;
        } else {
          int walker = node.chain;
          double total_cost = 0.0;
          for (u32 i = 0; i < node.depth; ++i) {
            const u32 error = chain_error[walker];
            out_errors[(u64)shot * max_output + node.depth - 1 - i] = error;
            walker = chain_parent[walker];
          }
          for (u32 i = 0; i < node.depth; ++i) {
            const u64 output_index = (u64)shot * max_output + i;
            const u32 error = out_errors[output_index];
            total_cost = __dadd_rn(total_cost, weights[error]);
            out_errors[output_index] = error_to_dem[error];
          }
          out_length[shot] = node.depth;
          out_cost_bits[shot] = __double_as_longlong(total_cost);
          out_status[shot] = 1;
        }
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
        out_cycles[shot] = clock64() - cycle_start;
      }
      return;
    }

    if ((++generation & PARENT_CACHE_CLAIM_BIT) != 0) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }

    int current_cursor = current_chain;
    int target_cursor = node.chain;
    u32 current_cursor_depth = current_depth;
    u32 target_cursor_depth = node.depth;
    int stack_length = 0;
    bool state_ok = true;
    while (current_cursor_depth > target_cursor_depth) {
      if (!apply_chain_edge_cta(
              current_cursor, true, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets, &s_bad)) {
        state_ok = false;
        break;
      }
      current_cursor = chain_parent[current_cursor];
      --current_cursor_depth;
    }
    while (state_ok && target_cursor_depth > current_cursor_depth) {
      if (stack_length >= workspace_capacity) {
        state_ok = false;
        break;
      }
      if (tid == 0) path_stack[stack_length] = target_cursor;
      ++stack_length;
      target_cursor = chain_parent[target_cursor];
      --target_cursor_depth;
    }
    while (state_ok && current_cursor != target_cursor) {
      if (current_cursor < 0 || target_cursor < 0 ||
          stack_length >= workspace_capacity) {
        state_ok = false;
        break;
      }
      if (!apply_chain_edge_cta(
              current_cursor, true, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets, &s_bad)) {
        state_ok = false;
        break;
      }
      current_cursor = chain_parent[current_cursor];
      if (tid == 0) path_stack[stack_length] = target_cursor;
      ++stack_length;
      target_cursor = chain_parent[target_cursor];
    }
    __syncthreads();
    while (state_ok && stack_length != 0) {
      const int edge = path_stack[--stack_length];
      if (!apply_chain_edge_cta(
              edge, false, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets, &s_bad)) {
        state_ok = false;
      }
    }
    if (!state_ok) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }
    current_chain = node.chain;
    current_depth = node.depth;

    if (tid == 0) {
      s_empty_slot = -1;
      s_visited = visited_find(
          node.hash1, node.hash2, state_bits, s_empty_slot,
          visited_hash1, visited_hash2, visited_chain, visited_capacity,
          initial_bits, compare_bits, num_words, chain_error, chain_parent,
          edet_offsets, edets);
    }
    __syncthreads();
    if (s_visited < 0) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }
    if (s_visited > 0) continue;
    if (tid == 0) ++debug.expanded;
    if (tid == 0) {
      visited_hash1[s_empty_slot] = node.hash1;
      visited_hash2[s_empty_slot] = node.hash2;
      visited_chain[s_empty_slot] = (u32)(node.chain + 2);
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      const u32 tightened = min_num_dets + detector_beam;
      if (tightened < max_num_dets) max_num_dets = tightened;
    }
    if (!transition_parent_counts_cta(
            state_bits, counted_bits, num_words, parent_counts, child_counts,
            d2e_offsets, d2e, &s_detector, &s_delta, &s_bad)) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }
    if (tid == 0) {
      s_min_detector = 0xffffffffU;
      // Exported exact models use the identity detector order.  Scan the
      // packed syndrome directly instead of testing every detector bit.
      for (int word = 0; word < num_words; ++word) {
        const u64 active = state_bits[word];
        if (active != 0) {
          s_min_detector = (u32)(word * 64 + __ffsll((long long)active) - 1);
          break;
        }
      }
    }
    __syncthreads();
    const u32 min_detector = s_min_detector;
    if (min_detector == 0xffffffffU) {
      if (tid == 0) out_status[shot] = 3;
      return;
    }

    const u64 branch_begin = d2e_offsets[min_detector];
    const u64 branch_end = d2e_offsets[min_detector + 1];
    for (u64 branch_row = branch_begin + tid;
         branch_row < branch_end; branch_row += CTA_THREADS) {
      set_prefix_position(
          base_stamp, d2e[branch_row],
          (u16)(branch_row - branch_begin + 1));
    }
    __syncthreads();
    const u32 prefix_begin = (u32)branch_begin;
    int previous_child_error = -1;
    for (u64 row = branch_begin; row < branch_end; ++row) {
      if (tid == 0) ++debug.branch_rows;
      const u32 error = d2e[row];
      if (blocker_count_value(base_stamp, error) != 0) continue;
      if (tid == 0) ++debug.base_pass;
      const u32 prefix_limit = (u32)(row - branch_begin + 1);

      const u32 parent_overlap = parent_counts[error];
      const u32 degree = (u32)(edet_offsets[error + 1] - edet_offsets[error]);
      const u32 next_num_dets = node.num_dets + degree - 2 * parent_overlap;
      if (next_num_dets > max_num_dets) continue;
      if (tid == 0) ++debug.beam_pass;

      if (tid == 0) {
        const u64 child_hash1 = node.hash1 ^ error_hash1[error];
        const u64 child_hash2 = node.hash2 ^ error_hash2[error];
        int child_empty = -1;
        s_visited = visited_find_virtual_child_cta(
            child_hash1, child_hash2, state_bits, error, child_empty,
            visited_hash1, visited_hash2, visited_chain, visited_capacity,
            initial_bits, compare_bits, num_words, chain_error, chain_parent,
            edet_offsets, edets);
      }
      __syncthreads();
      if (s_visited < 0) {
        if (tid == 0) out_status[shot] = 3;
        return;
      }
      if (s_visited > 0) continue;
      if (tid == 0) ++debug.visited_pass;

      bool counts_ok = true;
      if (previous_child_error != -1) {
        counts_ok = transition_child_counts_cta(
            (u32)previous_child_error, error, state_bits, child_counts,
            edet_offsets, edets, d2e_offsets, d2e, &s_bad);
      } else {
        counts_ok = adjust_child_counts_cta(
            error, true, state_bits, child_counts,
            edet_offsets, edets, d2e_offsets, d2e, &s_bad);
      }
      if (!counts_ok) {
        if (tid == 0) out_status[shot] = 3;
        return;
      }
      previous_child_error = (int)error;

      // Gather the exact floating-point fold sequence.  Operations use bit 0
      // for a parent subtraction and bit 1 for a child addition.
      if (tid == 0) {
        s_next_cost = __dadd_rn(node.cost, weights[error]);
        s_affected_count = 0;
        for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
          const u32 detector = edets[j];
          s_affected_detector[s_affected_count] = detector;
          s_affected_operation[s_affected_count] =
              bit_is_set(state_bits, detector) ? 1U : 2U;
          ++s_affected_count;
        }
        for (u64 group = eneighbor_word_offsets[error];
             group < eneighbor_word_offsets[error + 1]; ++group) {
          const u32 word = eneighbor_words[group];
          u64 active_bits = eneighbor_masks[group] & state_bits[word];
          while (active_bits != 0) {
            if (s_affected_count >= MAX_AFFECTED_DETECTORS) break;
            const u32 detector = (word << 6) +
                (u32)(__ffsll((long long)active_bits) - 1);
            s_affected_detector[s_affected_count] = detector;
            s_affected_operation[s_affected_count] = 3U;
            ++s_affected_count;
            active_bits &= active_bits - 1;
          }
        }
      }
      __syncthreads();
      if (tid == 0) {
        debug.affected += s_affected_count;
        for (u32 index = 0; index < s_affected_count; ++index) {
          debug.parent_queries += s_affected_operation[index] & 1U;
          debug.child_queries += (s_affected_operation[index] >> 1) & 1U;
        }
      }
      if (s_affected_count >= MAX_AFFECTED_DETECTORS) {
        if (tid == 0) out_status[shot] = 3;
        return;
      }

      const int lane = tid & 31;
      // Ranked detector rows have unequal lengths, especially for BB models.
      // Let each warp claim another query as soon as it finishes.  Query
      // results remain indexed by the original affected-detector order; the
      // exact binary64 fold below therefore remains byte-for-byte unchanged.
      if (tid == 0) s_query_cursor = 0;
      __syncthreads();
      while (true) {
        u32 affected_index = 0;
        if (lane == 0) affected_index = atomicAdd(&s_query_cursor, 1U);
        affected_index = __shfl_sync(0xffffffffU, affected_index, 0);
        if (affected_index >= s_affected_count) break;
        const u32 detector = s_affected_detector[affected_index];
        const u32 operation = s_affected_operation[affected_index];
        double parent_value = 0.0;
        double child_value = 0.0;
        if ((operation & 1U) != 0) {
          // Detector lists are unique in exported models.  The claim protocol
          // also makes malformed/legacy duplicate entries race-free: one warp
          // publishes the shared cache and every racing warp computes the same
          // pure value locally without writing that cache entry.
          u32 cache_role = 0;  // 1: cached, 2: owner, 3: duplicate compute.
          if (lane == 0) {
            const u32 claim = generation | PARENT_CACHE_CLAIM_BIT;
            const u32 observed = atomicAdd(parent_cost_stamp + detector, 0U);
            if (observed == generation) {
              cache_role = 1;
            } else if (observed == claim) {
              cache_role = 3;
            } else {
              const u32 prior = atomicCAS(
                  parent_cost_stamp + detector, observed, claim);
              if (prior == observed) cache_role = 2;
              else if (prior == generation) cache_role = 1;
              else cache_role = 3;
            }
          }
          cache_role = __shfl_sync(0xffffffffU, cache_role, 0);
          if (cache_role == 1) {
            if (lane == 0) parent_value = parent_cost_cache[detector];
            parent_value = __shfl_sync(0xffffffffU, parent_value, 0);
          } else {
            parent_value = detector_cost_subwarp(
                detector, parent_counts, base_stamp, prefix_stamp, 0, 0,
                d2e_offsets, d2e_ranked, count_ranks16, count_values, rank_stride);
            if (lane == 0 && cache_role == 2) {
              parent_cost_cache[detector] = parent_value;
              __threadfence_block();
              atomicExch(parent_cost_stamp + detector, generation);
            }
          }
        }
        if ((operation & 2U) != 0) {
          child_value = detector_cost_subwarp(
              detector, child_counts, base_stamp, prefix_stamp,
              prefix_begin, prefix_limit,
              d2e_offsets, d2e_ranked, count_ranks16, count_values, rank_stride);
        }
        if (lane == 0) {
          s_parent_values[affected_index] = parent_value;
          s_child_values[affected_index] = child_value;
        }
      }
      __syncthreads();
      if (tid == 0) {
        for (u32 index = 0; index < s_affected_count; ++index) {
          const u32 operation = s_affected_operation[index];
          if ((operation & 1U) != 0) {
            s_next_cost = __dsub_rn(s_next_cost, s_parent_values[index]);
          }
          if ((operation & 2U) != 0) {
            s_next_cost = __dadd_rn(s_next_cost, s_child_values[index]);
          }
        }
      }
      __syncthreads();
      if (s_next_cost == inf_value()) continue;

      if ((u64)num_pushed + 1ULL > (u64)pqlimit) {
        if (tid == 0) {
          out_status[shot] = 2;
          out_pushes[shot] = num_pushed + 1;
          out_pops[shot] = pop_count;
          out_cycles[shot] = clock64() - cycle_start;
        }
        return;
      }
      if (chain_count >= workspace_capacity || s_heap_length >= workspace_capacity) {
        if (tid == 0) {
          out_status[shot] = 3;
          out_pushes[shot] = num_pushed;
          out_pops[shot] = pop_count;
          out_cycles[shot] = clock64() - cycle_start;
        }
        return;
      }
      const int next_chain = chain_count++;
      if (tid == 0) {
        const u64 child_hash1 = node.hash1 ^ error_hash1[error];
        const u64 child_hash2 = node.hash2 ^ error_hash2[error];
        chain_error[next_chain] = error;
        chain_min_detector[next_chain] = min_detector;
        chain_parent[next_chain] = node.chain;
        chain_row[next_chain] = (u32)row;
        chain_depth[next_chain] = node.depth + 1;
        chain_hash1[next_chain] = child_hash1;
        chain_hash2[next_chain] = child_hash2;
        CtaCompactNode child{s_next_cost, next_num_dets, next_chain};
        cta_compact_push(
            s_heap_length, child, heap_cost, heap_num_dets, heap_chain,
            s_heap_cost, s_heap_num_dets, s_heap_chain);
      }
      __syncthreads();
      ++num_pushed;
    }
    // Keep the mirror equal to the parent counts between expanded nodes.  The
    // next state transition can then update both tables by only the changed
    // detector rows instead of copying the full E-byte table per pop.
    if (previous_child_error != -1) {
      if (!adjust_child_counts_cta(
              (u32)previous_child_error, false, state_bits, child_counts,
              edet_offsets, edets, d2e_offsets, d2e, &s_bad)) {
        if (tid == 0) out_status[shot] = 3;
        return;
      }
    }
    __syncthreads();
    for (u64 row = branch_begin + tid;
         row < branch_end; row += CTA_THREADS) {
      clear_prefix_position(base_stamp, d2e[row]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    out_status[shot] = 2;
    out_pushes[shot] = num_pushed;
    out_pops[shot] = pop_count;
    out_cycles[shot] = clock64() - cycle_start;
  }
}
"""


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


class ExactCtaDecoder:
    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.heap_cache = int(heap_cache)
        self.visited_capacity = _next_power_of_two(2 * self.workspace_capacity)
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.workspace_capacity >= np.iinfo(np.uint16).max:
            raise ValueError(
                "packed blocker counts require workspace_capacity < 65535")
        if self.max_output >= np.iinfo(np.uint16).max:
            raise ValueError("packed state requires max_output < 65535")
        if self.cta_threads not in (32, 64, 128, 256, 512):
            raise ValueError("cta_threads must be one of 32, 64, 128, 256, 512")
        if self.heap_cache not in (128, 256, 512, 1024):
            raise ValueError("heap_cache must be one of 128, 256, 512, 1024")
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
        self.d2e_ranked32, self.rank32_layout = repack_d2e_ranked32(model)
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
        options.extend((
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
            "weights", "error_to_dem",
            "detector_order", "error_hash1", "error_hash2", "count_values",
            "count_ranks16", "shot_offsets", "shot_detections",
            "shot_bits", "shot_hash1", "shot_hash2",
        ):
            self.device[name] = cp.asarray(getattr(model, name))
        self.device["d2e_ranked32"] = cp.asarray(self.d2e_ranked32)
        self.workspace = self._allocate_workspace()

    def _allocate_workspace(self) -> dict[str, cp.ndarray]:
        m = self.model.metadata
        shots = int(m["shots"])
        detectors = int(m["num_detectors"])
        errors = int(m["num_errors"])
        words = (detectors + 63) // 64
        n = shots * self.workspace_capacity
        v = shots * self.visited_capacity
        return {
            "heap_cost": cp.empty(n, cp.float64),
            "heap_num_dets": cp.empty(n, cp.uint32),
            "heap_depth": cp.empty(n, cp.uint32),
            "heap_chain": cp.empty(n, cp.int32),
            "heap_hash1": cp.empty(n, cp.uint64),
            "heap_hash2": cp.empty(n, cp.uint64),
            "chain_error": cp.empty(n, cp.uint32),
            "chain_min_detector": cp.empty(n, cp.uint32),
            "chain_parent": cp.empty(n, cp.int32),
            "chain_row": cp.empty(n, cp.uint32),
            "chain_depth": cp.empty(n, cp.uint32),
            "path_stack": cp.empty(n, cp.int32),
            "parent_counts": cp.empty(shots * errors, cp.uint8),
            "child_counts": cp.empty(shots * errors, cp.uint8),
            "counted_bits": cp.empty(shots * words, cp.uint64),
            "base_stamp": cp.empty(shots * errors, cp.uint32),
            # Packed into base_stamp's low half.  The kernel retains this
            # one-word placeholder only to keep the experimental ABI stable.
            "prefix_stamp": cp.empty(1, cp.uint32),
            "parent_cost_cache": cp.empty(shots * detectors, cp.float64),
            "parent_cost_stamp": cp.empty(shots * detectors, cp.uint32),
            "state_bits": cp.empty(shots * words, cp.uint64),
            "child_bits": cp.empty(shots * words, cp.uint64),
            "compare_bits": cp.empty(shots * words, cp.uint64),
            "visited_hash1": cp.empty(v, cp.uint64),
            "visited_hash2": cp.empty(v, cp.uint64),
            "visited_chain": cp.empty(v, cp.uint32),
            "out_status": cp.empty(shots, cp.uint32),
            "out_length": cp.empty(shots, cp.uint32),
            "out_errors": cp.empty(shots * self.max_output, cp.uint32),
            "out_cost_bits": cp.empty(shots, cp.uint64),
            "out_pushes": cp.empty(shots, cp.uint32),
            "out_pops": cp.empty(shots, cp.uint32),
            "out_cycles": cp.empty(shots, cp.uint64),
            "out_debug": cp.empty(shots * 8, cp.uint64),
        }

    def decode(self) -> tuple[list[dict], dict]:
        m = self.model.metadata
        shots = int(m["shots"])
        detectors = int(m["num_detectors"])
        errors = int(m["num_errors"])
        words = (detectors + 63) // 64
        d = self.device
        w = self.workspace
        for name in (
            "base_stamp", "prefix_stamp", "parent_cost_stamp",
            "parent_counts", "visited_chain",
        ):
            w[name].fill(0)
        args = (
            d["d2e_offsets"], d["d2e"], d["edet_offsets"], d["edets"],
            d["eneighbor_word_offsets"], d["eneighbor_words"],
            d["eneighbor_masks"], d["weights"],
            d["error_to_dem"], d["detector_order"], d["error_hash1"],
            d["error_hash2"], d["count_values"], np.uint32(m["rank_stride"]),
            d["d2e_ranked32"], d["count_ranks16"], d["shot_offsets"],
            d["shot_detections"], d["shot_bits"], d["shot_hash1"], d["shot_hash2"],
            np.int32(shots), np.int32(detectors), np.int32(errors), np.int32(words),
            np.uint32(m["beam"]), np.uint32(m["pqlimit"]),
            np.int32(self.workspace_capacity), np.int32(self.visited_capacity),
            np.int32(self.max_output), w["heap_cost"], w["heap_num_dets"],
            w["heap_depth"], w["heap_chain"], w["heap_hash1"], w["heap_hash2"],
            w["chain_error"], w["chain_min_detector"], w["chain_parent"],
            w["chain_row"], w["chain_depth"], w["path_stack"],
            w["parent_counts"], w["child_counts"],
            w["counted_bits"], w["base_stamp"], w["prefix_stamp"],
            w["parent_cost_cache"], w["parent_cost_stamp"], w["state_bits"],
            w["child_bits"], w["compare_bits"], w["visited_hash1"],
            w["visited_hash2"], w["visited_chain"], w["out_status"],
            w["out_length"], w["out_errors"], w["out_cost_bits"],
            w["out_pushes"], w["out_pops"], w["out_cycles"], w["out_debug"],
        )
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        self.kernel((shots,), (self.cta_threads,), args)
        end.record()
        end.synchronize()
        kernel_ms = float(cp.cuda.get_elapsed_time(start, end))

        status = cp.asnumpy(w["out_status"])
        lengths = cp.asnumpy(w["out_length"])
        errors_out = cp.asnumpy(w["out_errors"]).reshape(shots, self.max_output)
        cost_bits = cp.asnumpy(w["out_cost_bits"])
        pushes = cp.asnumpy(w["out_pushes"])
        pops = cp.asnumpy(w["out_pops"])
        cycles = cp.asnumpy(w["out_cycles"])
        debug = cp.asnumpy(w["out_debug"]).reshape(shots, 8)
        decisions = []
        for shot in range(shots):
            predicted_errors = [int(x) for x in errors_out[shot, : int(lengths[shot])]]
            decisions.append({
                "status": int(status[shot]),
                "low_confidence": int(status[shot]) == 2,
                "predicted_errors_populated": True,
                "total_cost_bits": int(cost_bits[shot]),
                "predicted_errors": predicted_errors,
                "predictions": self.model.predictions_for_dem_errors(predicted_errors),
            })
        return decisions, {
            "kernel_ms": kernel_ms,
            "workspace_capacity": self.workspace_capacity,
            "cta_threads": self.cta_threads,
            "heap_cache": self.heap_cache,
            "compact_state": "blocker16_prefix16",
            "ranked_incidence_layout": {
                "storage_bits": 32,
                "used_bits": self.rank32_layout.total_bits,
                "error_bits": self.rank32_layout.error_bits,
                "group_bits": self.rank32_layout.group_bits,
                "lower_bits": self.rank32_layout.lower_bits,
                "bytes": int(self.d2e_ranked32.nbytes),
            },
            "workspace_bytes": sum(int(value.nbytes) for value in w.values()),
            "status_counts": {
                int(value): int(np.count_nonzero(status == value))
                for value in np.unique(status)
            },
            "max_pushes": int(pushes.max(initial=0)),
            "max_pops": int(pops.max(initial=0)),
            "median_pushes": float(np.median(pushes)),
            "median_pops": float(np.median(pops)),
            "shot_telemetry": [
                {
                    "shot": int(shot),
                    "status": int(status[shot]),
                    "pushes": int(pushes[shot]),
                    "pops": int(pops[shot]),
                    "cycles": int(cycles[shot]),
                }
                for shot in range(shots)
            ],
            "cycle_quantiles": {
                str(q): float(np.quantile(cycles, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            },
            "debug_totals": {
                name: int(debug[:, index].sum())
                for index, name in enumerate((
                    "expanded", "branch_rows", "base_pass", "beam_pass",
                    "visited_pass", "affected", "parent_queries", "child_queries",
                ))
            },
        }


def _pin_one_cpu() -> None:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        kernel32.SetProcessAffinityMask.restype = ctypes.c_int
        if not kernel32.SetProcessAffinityMask(kernel32.GetCurrentProcess(), 1):
            raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask failed")
    elif hasattr(os, "sched_getaffinity"):
        available = os.sched_getaffinity(0)
        os.sched_setaffinity(0, {min(available)})


def main() -> None:
    import argparse
    _pin_one_cpu()
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




