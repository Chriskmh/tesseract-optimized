"""Exact-search CUDA prototype for immutable, one-order Tesseract batches.

The model comes from ``export_model``.  The device implementation deliberately
uses one CUDA thread per shot first: this is the correctness reference used
before adding CTA cooperation.  It keeps the graph resident, stores search
states as error chains plus exact syndrome hashes, and asks the caller to replay
a shot on CPU if a finite device workspace fills.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault(
    "CUPY_CACHE_DIR", str(Path(__file__).resolve().parents[3] / "scratch" / "cupy_cache")
)

import cupy as cp
import numpy as np


KERNEL = r"""
typedef unsigned int u32;
typedef unsigned long long u64;
typedef unsigned short u16;
typedef unsigned char u8;
typedef signed char i8;

struct NodeValue {
  double cost;
  u32 num_dets;
  u32 depth;
  int chain;
  u64 hash1;
  u64 hash2;
};

__device__ __forceinline__ double inf_value() {
  return __longlong_as_double(0x7ff0000000000000ULL);
}

__device__ __forceinline__ bool node_greater(const NodeValue &a, const NodeValue &b) {
  return a.cost > b.cost || (a.cost == b.cost && a.num_dets < b.num_dets);
}

__device__ __forceinline__ NodeValue heap_load(
    int i, const double *cost, const u32 *num_dets, const u32 *depth,
    const int *chain, const u64 *hash1, const u64 *hash2) {
  NodeValue value;
  value.cost = cost[i];
  value.num_dets = num_dets[i];
  value.depth = depth[i];
  value.chain = chain[i];
  value.hash1 = hash1[i];
  value.hash2 = hash2[i];
  return value;
}

__device__ __forceinline__ void heap_store(
    int i, const NodeValue &value, double *cost, u32 *num_dets, u32 *depth,
    int *chain, u64 *hash1, u64 *hash2) {
  cost[i] = value.cost;
  num_dets[i] = value.num_dets;
  depth[i] = value.depth;
  chain[i] = value.chain;
  hash1[i] = value.hash1;
  hash2[i] = value.hash2;
}

// This is the libstdc++ push_heap path used by the WSL CPU oracle.  In
// particular, comparator-equivalent nodes do not move upward.
__device__ __noinline__ void heap_push(
    int &length, const NodeValue &input, double *cost, u32 *num_dets, u32 *depth,
    int *chain, u64 *hash1, u64 *hash2) {
  int hole = length++;
  const int top = 0;
  NodeValue value = input;
  int parent = (hole - 1) / 2;
  while (hole > top) {
    NodeValue parent_value = heap_load(parent, cost, num_dets, depth, chain, hash1, hash2);
    if (!node_greater(parent_value, value)) break;
    heap_store(hole, parent_value, cost, num_dets, depth, chain, hash1, hash2);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  heap_store(hole, value, cost, num_dets, depth, chain, hash1, hash2);
}

// This follows libstdc++ __pop_heap/__adjust_heap, including choosing the right
// child when the two children are comparator-equivalent.
__device__ __noinline__ NodeValue heap_pop(
    int &length, double *cost, u32 *num_dets, u32 *depth, int *chain,
    u64 *hash1, u64 *hash2) {
  NodeValue result = heap_load(0, cost, num_dets, depth, chain, hash1, hash2);
  NodeValue value = heap_load(length - 1, cost, num_dets, depth, chain, hash1, hash2);
  const int new_length = --length;
  if (new_length == 0) return result;

  const int top = 0;
  int hole = 0;
  int second_child = hole;
  while (second_child < (new_length - 1) / 2) {
    second_child = 2 * (second_child + 1);
    NodeValue right = heap_load(second_child, cost, num_dets, depth, chain, hash1, hash2);
    NodeValue left = heap_load(second_child - 1, cost, num_dets, depth, chain, hash1, hash2);
    if (node_greater(right, left)) --second_child;
    NodeValue selected = heap_load(second_child, cost, num_dets, depth, chain, hash1, hash2);
    heap_store(hole, selected, cost, num_dets, depth, chain, hash1, hash2);
    hole = second_child;
  }
  if ((new_length & 1) == 0 && second_child == (new_length - 2) / 2) {
    second_child = 2 * (second_child + 1);
    NodeValue left = heap_load(second_child - 1, cost, num_dets, depth, chain, hash1, hash2);
    heap_store(hole, left, cost, num_dets, depth, chain, hash1, hash2);
    hole = second_child - 1;
  }
  int parent = (hole - 1) / 2;
  while (hole > top) {
    NodeValue parent_value = heap_load(parent, cost, num_dets, depth, chain, hash1, hash2);
    if (!node_greater(parent_value, value)) break;
    heap_store(hole, parent_value, cost, num_dets, depth, chain, hash1, hash2);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  heap_store(hole, value, cost, num_dets, depth, chain, hash1, hash2);
  return result;
}

__device__ __forceinline__ bool bit_is_set(const u64 *bits, u32 detector) {
  return ((bits[detector >> 6] >> (detector & 63)) & 1ULL) != 0;
}

__device__ __forceinline__ void toggle_bit(u64 *bits, u32 detector) {
  bits[detector >> 6] ^= 1ULL << (detector & 63);
}

__device__ __forceinline__ u32 error_count(
    u32 error, const u64 *bits, const u64 *edet_offsets, const u32 *edets) {
  u32 count = 0;
  const u64 begin = edet_offsets[error];
  const u64 end = edet_offsets[error + 1];
  for (u64 i = begin; i < end; ++i) count += bit_is_set(bits, edets[i]);
  return count;
}

__device__ __noinline__ double detector_cost(
    u32 detector, const u64 *bits,
    const u32 *base_stamp, u32 base_generation,
    const u32 *prefix_stamp, u32 prefix_generation,
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets,
    const double *weights, double penalty,
    const u32 *rank_group, const u32 *rank_lower,
    const u32 *count_ranks, const double *count_values, u32 rank_stride) {
  (void)weights;
  (void)penalty;
  u32 best_rank = 0xffffffffU;
  u32 best_type = 0xffffffffU;
  for (u64 j = d2e_offsets[detector]; j < d2e_offsets[detector + 1]; ++j) {
    const u32 error = d2e[j];
    if (rank_lower[error] >= best_rank) break;
    const bool blocked = base_stamp[error] == base_generation ||
        (prefix_generation != 0 && prefix_stamp[error] == prefix_generation);
    if (!blocked) {
      const u32 count = error_count(error, bits, edet_offsets, edets);
      const u32 type = rank_group[error] * rank_stride + count;
      const u32 rank = count_ranks[type];
      if (rank < best_rank) {
        best_rank = rank;
        best_type = type;
      }
    }
  }
  return best_type == 0xffffffffU ? inf_value() : count_values[best_type];
}

__device__ __noinline__ void reconstruct_state(
    int chain_index, const u64 *initial_bits, u64 *bits, int num_words,
    u32 *base_stamp, u32 generation,
    const u32 *chain_error, const u32 *chain_min_detector, const int *chain_parent,
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets) {
  for (int word = 0; word < num_words; ++word) bits[word] = initial_bits[word];
  int index = chain_index;
  while (index != -1) {
    const u32 error = chain_error[index];
    const u32 min_detector = chain_min_detector[index];
    for (u64 j = d2e_offsets[min_detector]; j < d2e_offsets[min_detector + 1]; ++j) {
      const u32 blocked_error = d2e[j];
      base_stamp[blocked_error] = generation;
      if (blocked_error == error) break;
    }
    for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
      toggle_bit(bits, edets[j]);
    }
    index = chain_parent[index];
  }
}

__device__ __noinline__ bool state_equals_chain(
    const u64 *candidate, int existing_chain,
    const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u64 *edet_offsets, const u32 *edets) {
  for (int word = 0; word < num_words; ++word) scratch[word] = initial_bits[word];
  int index = existing_chain;
  while (index != -1) {
    const u32 error = chain_error[index];
    for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
      toggle_bit(scratch, edets[j]);
    }
    index = chain_parent[index];
  }
  for (int word = 0; word < num_words; ++word) {
    if (candidate[word] != scratch[word]) return false;
  }
  return true;
}

// Returns 1 if found, 0 if absent (empty_slot receives insertion position),
// and -1 if the table has no empty slot.
__device__ __noinline__ int visited_find(
    u64 hash1, u64 hash2, const u64 *candidate, int &empty_slot,
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
      if (state_equals_chain(candidate, existing_chain, initial_bits, scratch, num_words,
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

extern "C" __global__ void decode_exact(
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *eneighbor_offsets, const u32 *eneighbors,
    const double *weights, const u32 *error_to_dem,
    const u32 *detector_order, const u64 *error_hash1, const u64 *error_hash2,
    const u32 *rank_group, const u32 *rank_lower,
    const u32 *count_ranks, const double *count_values, u32 rank_stride,
    const u64 *d2e_ranked, const u16 *count_ranks16,
    const u64 *shot_offsets, const u32 *shot_detections,
    const u64 *initial_shot_bits, const u64 *shot_hash1, const u64 *shot_hash2,
    int num_shots, int num_detectors, int num_errors, int num_words,
    u32 detector_beam, u32 pqlimit, double penalty,
    int workspace_capacity, int visited_capacity, int max_output,
    double *heap_cost_all, u32 *heap_num_dets_all, u32 *heap_depth_all,
    int *heap_chain_all, u64 *heap_hash1_all, u64 *heap_hash2_all,
    u32 *chain_error_all, u32 *chain_min_detector_all, int *chain_parent_all,
    u32 *chain_row_all, int *path_stack_all,
    u8 *parent_counts_all, u8 *child_counts_all, u64 *counted_bits_all,
    u32 *base_stamp_all, u32 *prefix_stamp_all,
    double *parent_cost_cache_all, u32 *parent_cost_stamp_all,
    u64 *state_bits_all, u64 *child_bits_all, u64 *compare_bits_all,
    u64 *visited_hash1_all, u64 *visited_hash2_all, u32 *visited_chain_all,
    u32 *out_status, u32 *out_length, u32 *out_errors,
    u64 *out_cost_bits, u32 *out_pushes, u32 *out_pops, u64 *out_cycles) {
  const int shot = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (shot >= num_shots) return;

  double *heap_cost = heap_cost_all + (u64)shot * workspace_capacity;
  u32 *heap_num_dets = heap_num_dets_all + (u64)shot * workspace_capacity;
  u32 *heap_depth = heap_depth_all + (u64)shot * workspace_capacity;
  int *heap_chain = heap_chain_all + (u64)shot * workspace_capacity;
  u64 *heap_hash1 = heap_hash1_all + (u64)shot * workspace_capacity;
  u64 *heap_hash2 = heap_hash2_all + (u64)shot * workspace_capacity;
  u32 *chain_error = chain_error_all + (u64)shot * workspace_capacity;
  u32 *chain_min_detector = chain_min_detector_all + (u64)shot * workspace_capacity;
  int *chain_parent = chain_parent_all + (u64)shot * workspace_capacity;
  u32 *chain_row = chain_row_all + (u64)shot * workspace_capacity;
  int *path_stack = path_stack_all + (u64)shot * workspace_capacity;
  u8 *parent_counts = parent_counts_all + (u64)shot * num_errors;
  u8 *child_counts = child_counts_all + (u64)shot * num_errors;
  u64 *counted_bits = counted_bits_all + (u64)shot * num_words;
  u32 *base_stamp = base_stamp_all + (u64)shot * num_errors;
  u32 *prefix_stamp = prefix_stamp_all + (u64)shot * num_errors;
  double *parent_cost_cache = parent_cost_cache_all + (u64)shot * num_detectors;
  u32 *parent_cost_stamp = parent_cost_stamp_all + (u64)shot * num_detectors;
  u64 *state_bits = state_bits_all + (u64)shot * num_words;
  u64 *child_bits = child_bits_all + (u64)shot * num_words;
  u64 *compare_bits = compare_bits_all + (u64)shot * num_words;
  u64 *visited_hash1 = visited_hash1_all + (u64)shot * visited_capacity;
  u64 *visited_hash2 = visited_hash2_all + (u64)shot * visited_capacity;
  u32 *visited_chain = visited_chain_all + (u64)shot * visited_capacity;
  const u64 *initial_bits = initial_shot_bits + (u64)shot * num_words;

  out_status[shot] = 0;
  out_length[shot] = 0;
  out_cost_bits[shot] = 0;
  out_pushes[shot] = 0;
  out_pops[shot] = 0;

  const u64 shot_begin = shot_offsets[shot];
  const u64 shot_end = shot_offsets[shot + 1];
  const u32 initial_num_dets = (u32)(shot_end - shot_begin);
  if (initial_num_dets == 0) {
    out_status[shot] = 1;
    return;
  }

  // The host stream clears all generation and visited arrays with flat device
  // fills immediately before launch.

  const u32 initial_generation = 1;
  double initial_cost = 0.0;
  for (u64 i = shot_begin; i < shot_end; ++i) {
    const u32 detector = shot_detections[i];
    const double value = detector_cost(
        detector, initial_bits, base_stamp, initial_generation,
        prefix_stamp, 0, d2e_offsets, d2e, edet_offsets, edets, weights, penalty,
        rank_group, rank_lower, count_ranks, count_values, rank_stride);
    initial_cost = __dadd_rn(initial_cost, value);
  }
  if (initial_cost == inf_value()) {
    out_status[shot] = 2;
    return;
  }

  int heap_length = 0;
  int chain_count = 0;
  u32 num_pushed = 1;
  u32 pop_count = 0;
  u32 generation = 1;
  u32 min_num_dets = initial_num_dets;
  u32 max_num_dets = initial_num_dets + detector_beam;
  int current_chain = -1;
  u32 current_depth = 0;
  NodeValue root{initial_cost, initial_num_dets, 0, -1, shot_hash1[shot], shot_hash2[shot]};
  heap_push(heap_length, root, heap_cost, heap_num_dets, heap_depth,
            heap_chain, heap_hash1, heap_hash2);

  while (heap_length != 0) {
    const NodeValue node = heap_pop(heap_length, heap_cost, heap_num_dets, heap_depth,
                                    heap_chain, heap_hash1, heap_hash2);
    ++pop_count;
    if (node.num_dets > max_num_dets) continue;
    if (node.num_dets == 0) {
      if ((int)node.depth > max_output) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
        return;
      }
      int walker = node.chain;
      double total_cost = 0.0;
      for (u32 i = 0; i < node.depth; ++i) {
        const u32 error = chain_error[walker];
        const u32 output_position = node.depth - 1 - i;
        out_errors[(u64)shot * max_output + output_position] = error;
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
      out_pushes[shot] = num_pushed;
      out_pops[shot] = pop_count;
      return;
    }

    if (++generation == 0) {
      out_status[shot] = 3;
      out_pushes[shot] = num_pushed;
      out_pops[shot] = pop_count;
      return;
    }
    reconstruct_state(node.chain, initial_bits, state_bits, num_words,
                      base_stamp, generation, chain_error, chain_min_detector, chain_parent,
                      d2e_offsets, d2e, edet_offsets, edets);

    int empty_slot = -1;
    const int visited = visited_find(
        node.hash1, node.hash2, state_bits, empty_slot,
        visited_hash1, visited_hash2, visited_chain, visited_capacity,
        initial_bits, compare_bits, num_words, chain_error, chain_parent,
        edet_offsets, edets);
    if (visited < 0) {
      out_status[shot] = 3;
      out_pushes[shot] = num_pushed;
      out_pops[shot] = pop_count;
      return;
    }
    if (visited > 0) continue;
    visited_hash1[empty_slot] = node.hash1;
    visited_hash2[empty_slot] = node.hash2;
    visited_chain[empty_slot] = (u32)(node.chain + 2);

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      const u32 tightened = min_num_dets + detector_beam;
      if (tightened < max_num_dets) max_num_dets = tightened;
    }

    u32 min_detector = 0xffffffffU;
    for (int position = 0; position < num_detectors; ++position) {
      const u32 detector = detector_order[position];
      if (bit_is_set(state_bits, detector)) {
        min_detector = detector;
        break;
      }
    }
    if (min_detector == 0xffffffffU) {
      out_status[shot] = 3;
      out_pushes[shot] = num_pushed;
      out_pops[shot] = pop_count;
      return;
    }

    const u32 prefix_generation = generation;
    for (u64 row = d2e_offsets[min_detector]; row < d2e_offsets[min_detector + 1]; ++row) {
      const u32 error = d2e[row];
      if (base_stamp[error] == generation) continue;
      prefix_stamp[error] = prefix_generation;

      const u32 parent_overlap = error_count(error, state_bits, edet_offsets, edets);
      const u32 degree = (u32)(edet_offsets[error + 1] - edet_offsets[error]);
      const u32 next_num_dets = node.num_dets + degree - 2 * parent_overlap;
      if (next_num_dets > max_num_dets) continue;

      for (int word = 0; word < num_words; ++word) child_bits[word] = state_bits[word];
      for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
        toggle_bit(child_bits, edets[j]);
      }
      const u64 child_hash1 = node.hash1 ^ error_hash1[error];
      const u64 child_hash2 = node.hash2 ^ error_hash2[error];
      int child_empty = -1;
      const int child_visited = visited_find(
          child_hash1, child_hash2, child_bits, child_empty,
          visited_hash1, visited_hash2, visited_chain, visited_capacity,
          initial_bits, compare_bits, num_words, chain_error, chain_parent,
          edet_offsets, edets);
      if (child_visited < 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
        return;
      }
      if (child_visited > 0) continue;

      double next_cost = __dadd_rn(node.cost, weights[error]);
      for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
        const u32 detector = edets[j];
        if (bit_is_set(state_bits, detector)) {
          double parent_value;
          if (parent_cost_stamp[detector] == generation) {
            parent_value = parent_cost_cache[detector];
          } else {
            parent_value = detector_cost(
                detector, state_bits, base_stamp, generation, prefix_stamp, 0,
                d2e_offsets, d2e, edet_offsets, edets, weights, penalty,
                rank_group, rank_lower, count_ranks, count_values, rank_stride);
            parent_cost_cache[detector] = parent_value;
            parent_cost_stamp[detector] = generation;
          }
          next_cost = __dsub_rn(next_cost, parent_value);
        } else {
          const double child_value = detector_cost(
              detector, child_bits, base_stamp, generation,
              prefix_stamp, prefix_generation,
              d2e_offsets, d2e, edet_offsets, edets, weights, penalty,
              rank_group, rank_lower, count_ranks, count_values, rank_stride);
          next_cost = __dadd_rn(next_cost, child_value);
        }
      }
      for (u64 j = eneighbor_offsets[error]; j < eneighbor_offsets[error + 1]; ++j) {
        const u32 detector = eneighbors[j];
        if (!bit_is_set(state_bits, detector) || !bit_is_set(child_bits, detector)) continue;
        double parent_value;
        if (parent_cost_stamp[detector] == generation) {
          parent_value = parent_cost_cache[detector];
        } else {
          parent_value = detector_cost(
              detector, state_bits, base_stamp, generation, prefix_stamp, 0,
              d2e_offsets, d2e, edet_offsets, edets, weights, penalty,
              rank_group, rank_lower, count_ranks, count_values, rank_stride);
          parent_cost_cache[detector] = parent_value;
          parent_cost_stamp[detector] = generation;
        }
        next_cost = __dsub_rn(next_cost, parent_value);
        const double child_value = detector_cost(
            detector, child_bits, base_stamp, generation,
            prefix_stamp, prefix_generation,
            d2e_offsets, d2e, edet_offsets, edets, weights, penalty,
            rank_group, rank_lower, count_ranks, count_values, rank_stride);
        next_cost = __dadd_rn(next_cost, child_value);
      }
      if (next_cost == inf_value()) continue;

      // CPU pushes the child and then checks the budget.  The transient pushed
      // node is unobservable when the check fails, so avoid requiring a
      // pqlimit+1 workspace slot while preserving the exact low-confidence result.
      if ((u64)num_pushed + 1ULL > (u64)pqlimit) {
        out_status[shot] = 2;
        out_pushes[shot] = num_pushed + 1;
        out_pops[shot] = pop_count;
        return;
      }
      if (chain_count >= workspace_capacity || heap_length >= workspace_capacity) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
        return;
      }
      const int next_chain = chain_count++;
      chain_error[next_chain] = error;
      chain_min_detector[next_chain] = min_detector;
      chain_parent[next_chain] = node.chain;
      NodeValue child{next_cost, next_num_dets, node.depth + 1, next_chain,
                      child_hash1, child_hash2};
      heap_push(heap_length, child, heap_cost, heap_num_dets, heap_depth,
                heap_chain, heap_hash1, heap_hash2);
      ++num_pushed;
    }
  }

  out_status[shot] = 2;
  out_pushes[shot] = num_pushed;
  out_pops[shot] = pop_count;
}
"""


WARP_KERNEL = r"""
// One cooperative warp owns one shot.  Lane zero preserves every ordered
// search mutation while all lanes scan the expensive detector-cost rows.
#ifndef TESSERACT_MONOTONIC_RANK_ROWS
#define TESSERACT_MONOTONIC_RANK_ROWS 0
#endif
struct WarpCycleRecorder {
  u64 start;
  u64 *output;
  int shot;
  int lane;
  __device__ WarpCycleRecorder(u64 *output, int shot, int lane)
      : start(clock64()), output(output), shot(shot), lane(lane) {}
  __device__ ~WarpCycleRecorder() {
    if (lane == 0) output[shot] = clock64() - start;
  }
};

__device__ __noinline__ double detector_cost_warp(
    u32 detector, const u8 *error_counts,
    const u32 *blocker_counts,
    const u32 *prefix_stamp, u32 prefix_generation,
    const u64 *d2e_offsets, const u64 *d2e_ranked,
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
    if (lane == 0) stop = ((d2e_ranked[base] >> 40) & 65535U) >= best_rank;
    stop = __shfl_sync(full, stop, 0);
    if (stop) break;
#endif

    const u64 position = base + (u64)lane;
    u32 local_type = 0xffffffffU;
    u32 local_rank = 0xffffffffU;
    u32 local_lower = 0xffffffffU;
    u64 key = 0xffffffffffffffffULL;
    if (position < end) {
      const u64 packed = d2e_ranked[position];
      const u32 error = (u32)packed;
      const u32 group = (u32)((packed >> 32) & 255U);
      const u32 lower = (u32)((packed >> 40) & 65535U);
      local_lower = lower;
      if (lower < best_rank) {
        const bool blocked = blocker_counts[error] != 0 ||
            (prefix_generation != 0 && prefix_stamp[error] == prefix_generation);
        if (!blocked) {
          const int count = error_counts[error];
          if (count < 0 || count > 16) continue;
          local_type = group * rank_stride + count;
          const u32 rank = count_ranks16[local_type];
          if (rank != 65535U) {
            local_rank = rank;
            key = ((u64)rank << 32) | (u32)(position - begin);
          }
        }
      }
    }
#if TESSERACT_MONOTONIC_RANK_ROWS
    for (int offset = 16; offset != 0; offset >>= 1) {
      const u64 other = __shfl_down_sync(full, key, offset);
      if (other < key) key = other;
    }
    key = __shfl_sync(full, key, 0);
    if (key != 0xffffffffffffffffULL) {
      const u32 rank = (u32)(key >> 32);
      const u32 relative_position = (u32)key;
      const int winner_lane = (int)(relative_position & 31U);
      const u32 type = __shfl_sync(full, local_type, winner_lane);
      if (rank < best_rank) {
        best_rank = rank;
        best_type = type;
      }
    }
#else
    // Some CPU rows contain rounded-min-cost ties whose exact cross-product
    // lower ranks decrease.  Reproduce the original ordered scan, including
    // its first early-exit position and strict first-winner tie rule.  The
    // count/rank loads remain lane-parallel; every lane performs the same
    // 32-entry shuffle fold so the warp stays converged.
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

struct CompactNode {
  double cost;
  u32 num_dets;
  int chain;
};

#ifndef TESSERACT_HEAP_CACHE
#define TESSERACT_HEAP_CACHE 0
#endif
#ifndef TESSERACT_WARPS_PER_BLOCK
#define TESSERACT_WARPS_PER_BLOCK 1
#endif
constexpr int compact_heap_cache = TESSERACT_HEAP_CACHE;
constexpr int warps_per_block = TESSERACT_WARPS_PER_BLOCK;

__device__ __forceinline__ bool compact_greater(
    const CompactNode &a, const CompactNode &b) {
  return a.cost > b.cost || (a.cost == b.cost && a.num_dets < b.num_dets);
}

__device__ __forceinline__ CompactNode compact_load(
    int i, const double *cost, const u32 *num_dets, const int *chain,
    const double *shared_cost, const u32 *shared_num_dets, const int *shared_chain) {
  if (i < compact_heap_cache) {
    return CompactNode{shared_cost[i], shared_num_dets[i], shared_chain[i]};
  }
  return CompactNode{cost[i], num_dets[i], chain[i]};
}

__device__ __forceinline__ void compact_store(
    int i, const CompactNode &value, double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  if (i < compact_heap_cache) {
    shared_cost[i] = value.cost;
    shared_num_dets[i] = value.num_dets;
    shared_chain[i] = value.chain;
  } else {
    cost[i] = value.cost;
    num_dets[i] = value.num_dets;
    chain[i] = value.chain;
  }
}

__device__ __noinline__ void compact_push(
    int &length, const CompactNode &input,
    double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  int hole = length++;
  CompactNode value = input;
  int parent = (hole - 1) / 2;
  while (hole > 0) {
    const CompactNode parent_value = compact_load(
        parent, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    if (!compact_greater(parent_value, value)) break;
    compact_store(hole, parent_value, cost, num_dets, chain,
                  shared_cost, shared_num_dets, shared_chain);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  compact_store(hole, value, cost, num_dets, chain,
                shared_cost, shared_num_dets, shared_chain);
}

__device__ __noinline__ CompactNode compact_pop(
    int &length, double *cost, u32 *num_dets, int *chain,
    double *shared_cost, u32 *shared_num_dets, int *shared_chain) {
  const CompactNode result = compact_load(
      0, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
  CompactNode value = compact_load(
      length - 1, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
  const int new_length = --length;
  if (new_length == 0) return result;
  int hole = 0;
  int second_child = 0;
  while (second_child < (new_length - 1) / 2) {
    second_child = 2 * (second_child + 1);
    const CompactNode right = compact_load(
        second_child, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    const CompactNode left = compact_load(
        second_child - 1, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    if (compact_greater(right, left)) --second_child;
    compact_store(hole, compact_load(
                      second_child, cost, num_dets, chain,
                      shared_cost, shared_num_dets, shared_chain),
                  cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    hole = second_child;
  }
  if ((new_length & 1) == 0 && second_child == (new_length - 2) / 2) {
    second_child = 2 * (second_child + 1);
    compact_store(hole, compact_load(
                      second_child - 1, cost, num_dets, chain,
                      shared_cost, shared_num_dets, shared_chain),
                  cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    hole = second_child - 1;
  }
  int parent = (hole - 1) / 2;
  while (hole > 0) {
    const CompactNode parent_value = compact_load(
        parent, cost, num_dets, chain, shared_cost, shared_num_dets, shared_chain);
    if (!compact_greater(parent_value, value)) break;
    compact_store(hole, parent_value, cost, num_dets, chain,
                  shared_cost, shared_num_dets, shared_chain);
    hole = parent;
    parent = (hole - 1) / 2;
  }
  compact_store(hole, value, cost, num_dets, chain,
                shared_cost, shared_num_dets, shared_chain);
  return result;
}

__device__ __forceinline__ u32 first_active_detector_identity_warp(
    const u64 *bits, int num_words) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  for (int base = 0; base < num_words; base += 32) {
    const int word_index = base + lane;
    const u64 word = word_index < num_words ? bits[word_index] : 0;
    const u32 active_lanes = __ballot_sync(full, word != 0);
    if (active_lanes) {
      const int word_lane = __ffs(active_lanes) - 1;
      const u64 first_word = __shfl_sync(full, word, word_lane);
      return (u32)((base + word_lane) * 64 + __ffsll((long long)first_word) - 1);
    }
  }
  return 0xffffffffU;
}

__device__ __forceinline__ bool update_count_row_warp(
    u32 detector, int delta, u8 *counts,
    const u64 *d2e_offsets, const u32 *d2e) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  bool bad = false;
  for (u64 row = d2e_offsets[detector] + lane;
       row < d2e_offsets[detector + 1]; row += 32) {
    const u32 error = d2e[row];
    const int value = (int)counts[error] + delta;
    if (value < 0 || value > 16) {
      bad = true;
    } else {
      counts[error] = (u8)value;
    }
  }
  const bool any_bad = __any_sync(full, bad);
  __syncwarp(full);
  return !any_bad;
}

__device__ __forceinline__ bool transition_child_counts_warp(
    int previous_error, u32 current_error, const u64 *parent_bits, u8 *counts,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *d2e_offsets, const u32 *d2e) {
  u64 previous_position = 0;
  u64 previous_end = 0;
  if (previous_error >= 0) {
    previous_position = edet_offsets[(u32)previous_error];
    previous_end = edet_offsets[(u32)previous_error + 1];
  }
  u64 current_position = edet_offsets[current_error];
  const u64 current_end = edet_offsets[current_error + 1];
  while (previous_position < previous_end || current_position < current_end) {
    const u32 previous_detector = previous_position < previous_end
        ? edets[previous_position] : 0xffffffffU;
    const u32 current_detector = current_position < current_end
        ? edets[current_position] : 0xffffffffU;
    if (previous_detector == current_detector) {
      ++previous_position;
      ++current_position;
      continue;
    }
    const bool apply_current = current_detector < previous_detector;
    const u32 detector = apply_current ? current_detector : previous_detector;
    int delta = bit_is_set(parent_bits, detector) ? -1 : 1;
    if (!apply_current) delta = -delta;
    if (!update_count_row_warp(detector, delta, counts, d2e_offsets, d2e)) return false;
    if (apply_current) {
      ++current_position;
    } else {
      ++previous_position;
    }
  }
  return true;
}

__device__ __forceinline__ bool build_child_delta_warp(
    u32 error, const u64 *parent_bits, i8 *delta_values, u32 *delta_stamps,
    u32 generation, const u64 *edet_offsets, const u32 *edets,
    const u64 *d2e_offsets, const u32 *d2e) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
    const u32 detector = edets[j];
    const int change = bit_is_set(parent_bits, detector) ? -1 : 1;
    bool bad = false;
    for (u64 row = d2e_offsets[detector] + lane;
         row < d2e_offsets[detector + 1]; row += 32) {
      const u32 affected_error = d2e[row];
      int value = change;
      if (delta_stamps[affected_error] == generation) {
        value += delta_values[affected_error];
      }
      if (value < -16 || value > 16) {
        bad = true;
      } else {
        delta_values[affected_error] = (i8)value;
        delta_stamps[affected_error] = generation;
      }
    }
    const bool any_bad = __any_sync(full, bad);
    __syncwarp(full);
    if (any_bad) return false;
  }
  return true;
}

__device__ __forceinline__ bool update_parent_counts_warp(
    const u64 *state_bits, u64 *counted_bits, u64 *delta_bits, int num_words,
    u8 *counts, const u64 *d2e_offsets, const u32 *d2e) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  for (int word = lane; word < num_words; word += 32) {
    delta_bits[word] = state_bits[word] ^ counted_bits[word];
  }
  __syncwarp(full);
  while (true) {
    const u32 detector = first_active_detector_identity_warp(delta_bits, num_words);
    if (detector == 0xffffffffU) break;
    if (lane == 0) toggle_bit(delta_bits, detector);
    __syncwarp(full);
    const int delta = bit_is_set(state_bits, detector) ? 1 : -1;
    if (!update_count_row_warp(detector, delta, counts, d2e_offsets, d2e)) return false;
  }
  for (int word = lane; word < num_words; word += 32) counted_bits[word] = state_bits[word];
  __syncwarp(full);
  return true;
}

__device__ __forceinline__ bool apply_chain_edge_warp(
    int chain_index, bool undo, u64 *bits, u32 *blocker_counts,
    const u32 *chain_error, const u32 *chain_min_detector, const u32 *chain_row,
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets) {
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  const u32 error = chain_error[chain_index];
  const u32 detector = chain_min_detector[chain_index];
  const u64 begin = d2e_offsets[detector];
  const u64 selected = chain_row[chain_index];
  bool bad = selected < begin || selected >= d2e_offsets[detector + 1];
  if (!bad) {
    for (u64 row = begin + lane; row <= selected; row += 32) {
      const u32 blocked_error = d2e[row];
      const u32 value = blocker_counts[blocked_error];
      if (undo) {
        if (value == 0) bad = true;
        else blocker_counts[blocked_error] = value - 1;
      } else {
        if (value == 0xffffffffU) bad = true;
        else blocker_counts[blocked_error] = value + 1;
      }
    }
  }
  const bool any_bad = __any_sync(full, bad);
  __syncwarp(full);
  if (any_bad) return false;
  if (lane == 0) {
    for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
      toggle_bit(bits, edets[j]);
    }
  }
  __syncwarp(full);
  return true;
}

// Compare an existing visited state against parent_bits XOR support(error)
// without materializing the child syndrome for every sibling.  Hash matches
// are uncommon; only this exact-match path reconstructs the existing chain.
__device__ __noinline__ bool state_equals_virtual_child(
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
  // candidate == reconstructed iff parent == reconstructed XOR support(error).
  for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
    toggle_bit(scratch, edets[j]);
  }
  for (int word = 0; word < num_words; ++word) {
    if (parent_bits[word] != scratch[word]) return false;
  }
  return true;
}

__device__ __noinline__ int visited_find_virtual_child(
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
      if (state_equals_virtual_child(
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

extern "C" __global__ void decode_exact_warp(
    const u64 *d2e_offsets, const u32 *d2e,
    const u64 *edet_offsets, const u32 *edets,
    const u64 *eneighbor_offsets, const u32 *eneighbors,
    const double *weights, const u32 *error_to_dem,
    const u32 *detector_order, const u64 *error_hash1, const u64 *error_hash2,
    const u32 *rank_group, const u32 *rank_lower,
    const u32 *count_ranks, const double *count_values, u32 rank_stride,
    const u64 *d2e_ranked, const u16 *count_ranks16,
    const u64 *shot_offsets, const u32 *shot_detections,
    const u64 *initial_shot_bits, const u64 *shot_hash1, const u64 *shot_hash2,
    int num_shots, int num_detectors, int num_errors, int num_words,
    u32 detector_beam, u32 pqlimit, double penalty,
    int workspace_capacity, int visited_capacity, int max_output,
    double *heap_cost_all, u32 *heap_num_dets_all, u32 *heap_depth_all,
    int *heap_chain_all, u64 *heap_hash1_all, u64 *heap_hash2_all,
    u32 *chain_error_all, u32 *chain_min_detector_all, int *chain_parent_all,
    u32 *chain_row_all, int *path_stack_all,
    u8 *parent_counts_all, u8 *child_counts_all, u64 *counted_bits_all,
    u32 *base_stamp_all, u32 *prefix_stamp_all,
    double *parent_cost_cache_all, u32 *parent_cost_stamp_all,
    u64 *state_bits_all, u64 *child_bits_all, u64 *compare_bits_all,
    u64 *visited_hash1_all, u64 *visited_hash2_all, u32 *visited_chain_all,
    u32 *out_status, u32 *out_length, u32 *out_errors,
    u64 *out_cost_bits, u32 *out_pushes, u32 *out_pops, u64 *out_cycles) {
  (void)penalty;
  const u32 full = 0xffffffffU;
  const int lane = (int)(threadIdx.x & 31);
  const int warp_in_block = (int)(threadIdx.x >> 5);
  const int shot = (int)blockIdx.x * warps_per_block + warp_in_block;
  if (shot >= num_shots) return;
  WarpCycleRecorder cycle_recorder(out_cycles, shot, lane);
  constexpr int shared_heap_stride = compact_heap_cache ? compact_heap_cache : 1;
  __shared__ double shared_heap_cost_all[warps_per_block * shared_heap_stride];
  __shared__ u32 shared_heap_num_dets_all[warps_per_block * shared_heap_stride];
  __shared__ int shared_heap_chain_all[warps_per_block * shared_heap_stride];
  double *shared_heap_cost = shared_heap_cost_all + warp_in_block * shared_heap_stride;
  u32 *shared_heap_num_dets =
      shared_heap_num_dets_all + warp_in_block * shared_heap_stride;
  int *shared_heap_chain = shared_heap_chain_all + warp_in_block * shared_heap_stride;

  double *heap_cost = heap_cost_all + (u64)shot * workspace_capacity;
  u32 *heap_num_dets = heap_num_dets_all + (u64)shot * workspace_capacity;
  u32 *chain_depth = heap_depth_all + (u64)shot * workspace_capacity;
  int *heap_chain = heap_chain_all + (u64)shot * workspace_capacity;
  u64 *chain_hash1 = heap_hash1_all + (u64)shot * workspace_capacity;
  u64 *chain_hash2 = heap_hash2_all + (u64)shot * workspace_capacity;
  u32 *chain_error = chain_error_all + (u64)shot * workspace_capacity;
  u32 *chain_min_detector = chain_min_detector_all + (u64)shot * workspace_capacity;
  int *chain_parent = chain_parent_all + (u64)shot * workspace_capacity;
  u32 *chain_row = chain_row_all + (u64)shot * workspace_capacity;
  int *path_stack = path_stack_all + (u64)shot * workspace_capacity;
  u8 *parent_counts = parent_counts_all + (u64)shot * num_errors;
  u8 *child_counts = child_counts_all + (u64)shot * num_errors;
  u64 *counted_bits = counted_bits_all + (u64)shot * num_words;
  u32 *base_stamp = base_stamp_all + (u64)shot * num_errors;
  u32 *prefix_stamp = prefix_stamp_all + (u64)shot * num_errors;
  double *parent_cost_cache = parent_cost_cache_all + (u64)shot * num_detectors;
  u32 *parent_cost_stamp = parent_cost_stamp_all + (u64)shot * num_detectors;
  u64 *state_bits = state_bits_all + (u64)shot * num_words;
  u64 *child_bits = child_bits_all + (u64)shot * num_words;
  u64 *compare_bits = compare_bits_all + (u64)shot * num_words;
  u64 *visited_hash1 = visited_hash1_all + (u64)shot * visited_capacity;
  u64 *visited_hash2 = visited_hash2_all + (u64)shot * visited_capacity;
  u32 *visited_chain = visited_chain_all + (u64)shot * visited_capacity;
  const u64 *initial_bits = initial_shot_bits + (u64)shot * num_words;

  if (lane == 0) {
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
    if (lane == 0) out_status[shot] = 1;
    return;
  }

  // Flat device fills issued by the host precede this launch in the stream.
  for (int word = lane; word < num_words; word += 32) {
    state_bits[word] = initial_bits[word];
    counted_bits[word] = initial_bits[word];
  }
  __syncwarp(full);
  for (u64 i = shot_begin; i < shot_end; ++i) {
    if (!update_count_row_warp(
            shot_detections[i], 1, parent_counts, d2e_offsets, d2e)) {
      if (lane == 0) out_status[shot] = 3;
      return;
    }
  }

  const u32 initial_generation = 1;
  double initial_cost = 0.0;
  for (u64 i = shot_begin; i < shot_end; ++i) {
    const u32 detector = shot_detections[i];
    const double value = detector_cost_warp(
        detector, parent_counts, base_stamp,
        prefix_stamp, 0, d2e_offsets, d2e_ranked,
        count_ranks16, count_values, rank_stride);
    if (lane == 0) initial_cost = __dadd_rn(initial_cost, value);
  }
  initial_cost = __shfl_sync(full, initial_cost, 0);
  if (initial_cost == inf_value()) {
    if (lane == 0) out_status[shot] = 2;
    return;
  }

  int heap_length = 0;
  int chain_count = 0;
  u32 num_pushed = 1;
  u32 pop_count = 0;
  u32 generation = 1;
  u32 min_num_dets = initial_num_dets;
  u32 max_num_dets = initial_num_dets + detector_beam;
  int current_chain = -1;
  u32 current_depth = 0;
  if (lane == 0) {
    CompactNode root{initial_cost, initial_num_dets, -1};
    compact_push(heap_length, root, heap_cost, heap_num_dets, heap_chain,
                 shared_heap_cost, shared_heap_num_dets, shared_heap_chain);
  }

  while (__shfl_sync(full, heap_length != 0, 0)) {
    NodeValue node;
    if (lane == 0) {
      const CompactNode compact = compact_pop(
          heap_length, heap_cost, heap_num_dets, heap_chain,
          shared_heap_cost, shared_heap_num_dets, shared_heap_chain);
      node.cost = compact.cost;
      node.num_dets = compact.num_dets;
      node.chain = compact.chain;
      if (compact.chain == -1) {
        node.depth = 0;
        node.hash1 = shot_hash1[shot];
        node.hash2 = shot_hash2[shot];
      } else {
        node.depth = chain_depth[compact.chain];
        node.hash1 = chain_hash1[compact.chain];
        node.hash2 = chain_hash2[compact.chain];
      }
    }
    node.cost = __shfl_sync(full, node.cost, 0);
    node.num_dets = __shfl_sync(full, node.num_dets, 0);
    node.depth = __shfl_sync(full, node.depth, 0);
    node.chain = __shfl_sync(full, node.chain, 0);
    node.hash1 = __shfl_sync(full, node.hash1, 0);
    node.hash2 = __shfl_sync(full, node.hash2, 0);
    ++pop_count;
    if (node.num_dets > max_num_dets) continue;
    if (node.num_dets == 0) {
      if (lane == 0) {
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
      }
      return;
    }

    if (++generation == 0) {
      if (lane == 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
      }
      return;
    }
    int current_cursor = current_chain;
    int target_cursor = node.chain;
    u32 current_cursor_depth = current_depth;
    u32 target_cursor_depth = node.depth;
    int stack_length = 0;
    bool state_ok = true;
    while (current_cursor_depth > target_cursor_depth) {
      if (!apply_chain_edge_warp(
              current_cursor, true, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets)) {
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
      if (lane == 0) path_stack[stack_length] = target_cursor;
      ++stack_length;
      target_cursor = chain_parent[target_cursor];
      --target_cursor_depth;
    }
    while (state_ok && current_cursor != target_cursor) {
      if (current_cursor < 0 || target_cursor < 0 || stack_length >= workspace_capacity) {
        state_ok = false;
        break;
      }
      if (!apply_chain_edge_warp(
              current_cursor, true, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets)) {
        state_ok = false;
        break;
      }
      current_cursor = chain_parent[current_cursor];
      if (lane == 0) path_stack[stack_length] = target_cursor;
      ++stack_length;
      target_cursor = chain_parent[target_cursor];
    }
    __syncwarp(full);
    while (state_ok && stack_length != 0) {
      const int edge = path_stack[--stack_length];
      if (!apply_chain_edge_warp(
              edge, false, state_bits, base_stamp,
              chain_error, chain_min_detector, chain_row,
              d2e_offsets, d2e, edet_offsets, edets)) {
        state_ok = false;
      }
    }
    if (!state_ok) {
      if (lane == 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
      }
      return;
    }
    current_chain = node.chain;
    current_depth = node.depth;

    int empty_slot = -1;
    int visited = 0;
    if (lane == 0) {
      visited = visited_find(
          node.hash1, node.hash2, state_bits, empty_slot,
          visited_hash1, visited_hash2, visited_chain, visited_capacity,
          initial_bits, compare_bits, num_words, chain_error, chain_parent,
          edet_offsets, edets);
    }
    visited = __shfl_sync(full, visited, 0);
    empty_slot = __shfl_sync(full, empty_slot, 0);
    if (visited < 0) {
      if (lane == 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
      }
      return;
    }
    if (visited > 0) continue;
    if (lane == 0) {
      visited_hash1[empty_slot] = node.hash1;
      visited_hash2[empty_slot] = node.hash2;
      visited_chain[empty_slot] = (u32)(node.chain + 2);
    }

    if (node.num_dets < min_num_dets) {
      min_num_dets = node.num_dets;
      const u32 tightened = min_num_dets + detector_beam;
      if (tightened < max_num_dets) max_num_dets = tightened;
    }
    if (!update_parent_counts_warp(
            state_bits, counted_bits, child_bits, num_words,
            parent_counts, d2e_offsets, d2e)) {
      if (lane == 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
      }
      return;
    }
    for (int error_index = lane; error_index < num_errors; error_index += 32) {
      child_counts[error_index] = parent_counts[error_index];
    }
    __syncwarp(full);
    const u32 min_detector = first_active_detector_identity_warp(state_bits, num_words);
    if (min_detector == 0xffffffffU) {
      if (lane == 0) {
        out_status[shot] = 3;
        out_pushes[shot] = num_pushed;
        out_pops[shot] = pop_count;
      }
      return;
    }

    const u32 prefix_generation = generation;
    int previous_count_error = -1;
    for (u64 row = d2e_offsets[min_detector]; row < d2e_offsets[min_detector + 1]; ++row) {
      const u32 error = d2e[row];
      if (base_stamp[error] != 0) continue;
      if (lane == 0) prefix_stamp[error] = prefix_generation;
      __syncwarp(full);

      const u32 parent_overlap = parent_counts[error];
      const u32 degree = (u32)(edet_offsets[error + 1] - edet_offsets[error]);
      const u32 next_num_dets = node.num_dets + degree - 2 * parent_overlap;
      if (next_num_dets > max_num_dets) continue;

      const u64 child_hash1 = node.hash1 ^ error_hash1[error];
      const u64 child_hash2 = node.hash2 ^ error_hash2[error];
      int child_empty = -1;
      int child_visited = 0;
      if (lane == 0) {
        child_visited = visited_find_virtual_child(
            child_hash1, child_hash2, state_bits, error, child_empty,
            visited_hash1, visited_hash2, visited_chain, visited_capacity,
            initial_bits, compare_bits, num_words, chain_error, chain_parent,
            edet_offsets, edets);
      }
      child_visited = __shfl_sync(full, child_visited, 0);
      if (child_visited < 0) {
        if (lane == 0) {
          out_status[shot] = 3;
          out_pushes[shot] = num_pushed;
          out_pops[shot] = pop_count;
        }
        return;
      }
      if (child_visited > 0) continue;

      if (!transition_child_counts_warp(
              previous_count_error, error, state_bits, child_counts,
              edet_offsets, edets, d2e_offsets, d2e)) {
        if (lane == 0) {
          out_status[shot] = 3;
          out_pushes[shot] = num_pushed;
          out_pops[shot] = pop_count;
        }
        return;
      }
      previous_count_error = (int)error;
      double next_cost = 0.0;
      if (lane == 0) next_cost = __dadd_rn(node.cost, weights[error]);
      for (u64 j = edet_offsets[error]; j < edet_offsets[error + 1]; ++j) {
        const u32 detector = edets[j];
        if (bit_is_set(state_bits, detector)) {
          u32 cached = 0;
          if (lane == 0) cached = parent_cost_stamp[detector] == generation;
          cached = __shfl_sync(full, cached, 0);
          double parent_value = 0.0;
          if (cached) {
            if (lane == 0) parent_value = parent_cost_cache[detector];
          } else {
            parent_value = detector_cost_warp(
                detector, parent_counts, base_stamp, prefix_stamp, 0,
                d2e_offsets, d2e_ranked,
                count_ranks16, count_values, rank_stride);
            if (lane == 0) {
              parent_cost_cache[detector] = parent_value;
              parent_cost_stamp[detector] = generation;
            }
          }
          if (lane == 0) next_cost = __dsub_rn(next_cost, parent_value);
        } else {
          const double child_value = detector_cost_warp(
              detector, child_counts, base_stamp, prefix_stamp, prefix_generation,
              d2e_offsets, d2e_ranked,
              count_ranks16, count_values, rank_stride);
          if (lane == 0) next_cost = __dadd_rn(next_cost, child_value);
        }
      }
      const u64 neighbor_begin = eneighbor_offsets[error];
      const u64 neighbor_end = eneighbor_offsets[error + 1];
      for (u64 base = neighbor_begin; base < neighbor_end; base += 32) {
        const u64 position = base + (u64)lane;
        const u32 lane_detector = position < neighbor_end ? eneighbors[position] : 0;
        // eneighbors[error] excludes the support of error, so the parent and
        // virtual child bits are identical for every detector in this row.
        const bool active = position < neighbor_end &&
            bit_is_set(state_bits, lane_detector);
        u32 active_lanes = __ballot_sync(full, active);
        while (active_lanes != 0) {
          const int source_lane = __ffs(active_lanes) - 1;
          const u32 detector = __shfl_sync(full, lane_detector, source_lane);
          u32 cached = 0;
          if (lane == 0) cached = parent_cost_stamp[detector] == generation;
          cached = __shfl_sync(full, cached, 0);
          double parent_value = 0.0;
          if (cached) {
            if (lane == 0) parent_value = parent_cost_cache[detector];
          } else {
            parent_value = detector_cost_warp(
                detector, parent_counts, base_stamp, prefix_stamp, 0,
                d2e_offsets, d2e_ranked,
                count_ranks16, count_values, rank_stride);
            if (lane == 0) {
              parent_cost_cache[detector] = parent_value;
              parent_cost_stamp[detector] = generation;
            }
          }
          if (lane == 0) next_cost = __dsub_rn(next_cost, parent_value);
          const double child_value = detector_cost_warp(
              detector, child_counts, base_stamp, prefix_stamp, prefix_generation,
              d2e_offsets, d2e_ranked,
              count_ranks16, count_values, rank_stride);
          if (lane == 0) next_cost = __dadd_rn(next_cost, child_value);
          active_lanes &= active_lanes - 1;
        }
      }
      next_cost = __shfl_sync(full, next_cost, 0);
      if (next_cost == inf_value()) continue;

      if ((u64)num_pushed + 1ULL > (u64)pqlimit) {
        if (lane == 0) {
          out_status[shot] = 2;
          out_pushes[shot] = num_pushed + 1;
          out_pops[shot] = pop_count;
        }
        return;
      }
      const int leader_heap_length = __shfl_sync(full, heap_length, 0);
      if (chain_count >= workspace_capacity || leader_heap_length >= workspace_capacity) {
        if (lane == 0) {
          out_status[shot] = 3;
          out_pushes[shot] = num_pushed;
          out_pops[shot] = pop_count;
        }
        return;
      }
      const int next_chain = chain_count++;
      if (lane == 0) {
        chain_error[next_chain] = error;
        chain_min_detector[next_chain] = min_detector;
        chain_parent[next_chain] = node.chain;
        chain_row[next_chain] = (u32)row;
        chain_depth[next_chain] = node.depth + 1;
        chain_hash1[next_chain] = child_hash1;
        chain_hash2[next_chain] = child_hash2;
        CompactNode child{next_cost, next_num_dets, next_chain};
        compact_push(heap_length, child, heap_cost, heap_num_dets, heap_chain,
                     shared_heap_cost, shared_heap_num_dets, shared_heap_chain);
      }
      ++num_pushed;
    }
  }

  if (lane == 0) {
    out_status[shot] = 2;
    out_pushes[shot] = num_pushed;
    out_pops[shot] = pop_count;
  }
}
"""


def _read(path: Path, name: str, dtype: np.dtype) -> np.ndarray:
    return np.fromfile(path / name, dtype=dtype)


def _splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def _next_power_of_two(value: int) -> int:
    return 1 << max(0, (value - 1).bit_length())


def _validate_sorted_csr(
    offsets: np.ndarray,
    values: np.ndarray,
    row_count: int,
    value_limit: int,
    label: str,
) -> None:
    """Validate a sorted, unique CSR using bounded temporary storage."""
    if row_count < 0 or len(offsets) != row_count + 1:
        raise ValueError(f"corrupt {label} CSR dimensions")
    if (
        int(offsets[0]) != 0
        or int(offsets[-1]) != len(values)
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError(f"corrupt {label} CSR bounds")
    if len(values) and int(values.max()) >= value_limit:
        raise ValueError(f"out-of-range {label} value")

    # A single global comparison can allocate hundreds of megabytes for the
    # large BB graphs.  Chunk it and clear comparisons that cross a row start.
    chunk_size = 1 << 22
    for begin in range(1, len(values), chunk_size):
        end = min(len(values), begin + chunk_size)
        invalid = values[begin:end] <= values[begin - 1 : end - 1]
        first_row = int(np.searchsorted(offsets, begin, side="left"))
        last_row = int(np.searchsorted(offsets, end - 1, side="right"))
        boundaries = offsets[first_row:last_row]
        boundaries = boundaries[(boundaries >= begin) & (boundaries < end)]
        if len(boundaries):
            invalid[(boundaries - begin).astype(np.intp, copy=False)] = False
        if np.any(invalid):
            raise ValueError(f"{label} rows must be strictly sorted and unique")


def _validate_packed_eneighbor_words(
    offsets: np.ndarray,
    words: np.ndarray,
    masks: np.ndarray,
    num_errors: int,
    num_detectors: int,
) -> None:
    if len(words) != len(masks):
        raise ValueError("corrupt packed eneighbor value arrays")
    detector_words = (num_detectors + 63) // 64
    _validate_sorted_csr(
        offsets, words, num_errors, detector_words, "packed eneighbor word"
    )
    if np.any(masks == 0):
        raise ValueError("packed eneighbor masks must be nonzero")
    final_bits = num_detectors & 63
    if final_bits and len(words):
        final_word = detector_words - 1
        invalid_mask = np.uint64(~((1 << final_bits) - 1) & ((1 << 64) - 1))
        if np.any((words == final_word) & ((masks & invalid_mask) != 0)):
            raise ValueError("out-of-range bit in packed eneighbor mask")


def _pack_eneighbor_words(
    offsets: np.ndarray, neighbors: np.ndarray, num_detectors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack each sorted detector row into ordered (word, bit-mask) groups."""
    _validate_sorted_csr(
        offsets, neighbors, len(offsets) - 1, num_detectors, "eneighbor"
    )
    if len(neighbors) == 0:
        return (
            np.zeros_like(offsets, dtype=np.uint64),
            np.empty(0, dtype=np.uint32),
            np.empty(0, dtype=np.uint64),
        )
    row_starts = offsets[:-1]
    words = np.right_shift(neighbors, np.uint32(6))
    starts = np.empty(len(neighbors), dtype=np.bool_)
    starts[0] = True
    starts[1:] = words[1:] != words[:-1]
    nonempty = row_starts < offsets[1:]
    starts[row_starts[nonempty].astype(np.intp, copy=False)] = True
    group_starts = np.flatnonzero(starts)
    bit_values = np.left_shift(
        np.uint64(1),
        np.bitwise_and(neighbors, np.uint32(63)).astype(np.uint64, copy=False),
    )
    masks = np.bitwise_or.reduceat(bit_values, group_starts).astype(np.uint64, copy=False)
    group_words = words[group_starts].astype(np.uint32, copy=False)
    group_offsets = np.searchsorted(group_starts, offsets, side="left").astype(
        np.uint64, copy=False
    )
    return group_offsets, group_words, masks


def _load_eneighbor_words(
    path: Path,
    raw_offsets: np.ndarray,
    raw_neighbors: np.ndarray,
    num_errors: int,
    num_detectors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Prefer the streamed on-disk form, with legacy host packing fallback."""
    names = (
        "eneighbor_word_offsets.u64",
        "eneighbor_words.u32",
        "eneighbor_masks.u64",
    )
    present = tuple((path / name).is_file() for name in names)
    if any(present) and not all(present):
        raise ValueError("incomplete packed eneighbor files")
    if all(present):
        offsets = _read(path, names[0], np.uint64)
        words = _read(path, names[1], np.uint32)
        masks = _read(path, names[2], np.uint64)
        _validate_packed_eneighbor_words(
            offsets, words, masks, num_errors, num_detectors
        )
        return offsets, words, masks
    return _pack_eneighbor_words(raw_offsets, raw_neighbors, num_detectors)


def _pack_rank_tables(
    d2e: np.ndarray,
    rank_group: np.ndarray,
    rank_lower: np.ndarray,
    count_ranks: np.ndarray,
    rank_groups: int,
    rank_stride: int,
    rank_maximum: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Validate and pack exact heuristic ranks for GPU row scans.

    The on-disk export deliberately keeps uint32 values, including UINT32_MAX
    for an invalid (group, count) combination.  The device representation uses
    uint16 and reserves UINT16_MAX for that sentinel.  Error id, eight-bit
    weight group, and sixteen-bit degree lower bound fit in one uint64 d2e row
    entry.  ``legacy_ranks8`` remains available for old CTA implementations
    only when the model is genuinely byte-wide; wide-rank callers fail during
    upload instead of silently truncating an exact rank.
    """
    packed_no_rank = int(np.iinfo(np.uint16).max)
    exported_no_rank = np.uint32(np.iinfo(np.uint32).max)
    if not 0 <= rank_maximum < packed_no_rank:
        raise ValueError("prototype packed ranks require rank_maximum < 65535")
    if not 0 < rank_stride <= 0xFFFFFFFF or not 0 <= rank_groups <= 256:
        raise ValueError("corrupt packed rank dimensions")
    if len(rank_group) != len(rank_lower):
        raise ValueError("corrupt rank error arrays")
    if len(count_ranks) != rank_groups * rank_stride:
        raise ValueError("corrupt count rank table")
    if len(rank_group) and (
        np.any(rank_group >= rank_groups)
        or np.any(rank_group > np.iinfo(np.uint8).max)
        or np.any(rank_lower > rank_maximum)
    ):
        raise ValueError("out-of-range packed error rank")

    valid = count_ranks != exported_no_rank
    if np.any(count_ranks[valid] > rank_maximum):
        raise ValueError("out-of-range packed count rank")
    count_ranks16 = np.full(len(count_ranks), packed_no_rank, dtype=np.uint16)
    count_ranks16[valid] = count_ranks[valid].astype(np.uint16, copy=False)
    d2e_ranked = (
        d2e.astype(np.uint64, copy=False)
        | (rank_group[d2e].astype(np.uint64) << np.uint64(32))
        | (rank_lower[d2e].astype(np.uint64) << np.uint64(40))
    )
    legacy_ranks8 = (
        count_ranks16.astype(np.uint8) if rank_maximum < 255 else None
    )
    return d2e_ranked, count_ranks16, legacy_ranks8


def _rank_rows_are_monotonic(
    d2e_offsets: np.ndarray, d2e: np.ndarray, rank_lower: np.ndarray
) -> bool:
    """Return whether every detector row has nondecreasing exact lower ranks."""
    lowers = rank_lower[d2e]
    if len(lowers) < 2:
        return True
    decreases = lowers[1:] < lowers[:-1]
    starts = d2e_offsets[:-1]
    starts = starts[(starts > 0) & (starts < len(lowers))]
    if len(starts):
        decreases[starts.astype(np.intp, copy=False) - 1] = False
    return not bool(np.any(decreases))


@dataclass
class HostModel:
    path: Path
    metadata: dict
    reference: list[dict]
    d2e_offsets: np.ndarray
    d2e: np.ndarray
    edet_offsets: np.ndarray
    edets: np.ndarray
    eneighbor_offsets: np.ndarray
    eneighbors: np.ndarray
    eneighbor_word_offsets: np.ndarray
    eneighbor_words: np.ndarray
    eneighbor_masks: np.ndarray
    weights: np.ndarray
    error_to_dem: np.ndarray
    detector_order: np.ndarray
    observable_offsets: np.ndarray
    observables: np.ndarray
    shot_offsets: np.ndarray
    shot_detections: np.ndarray
    error_hash1: np.ndarray
    error_hash2: np.ndarray
    rank_group: np.ndarray
    rank_lower: np.ndarray
    count_ranks: np.ndarray
    count_values: np.ndarray
    d2e_ranked: np.ndarray
    count_ranks16: np.ndarray
    rank_rows_monotonic: bool
    # Compatibility for experimental CTA files that have not yet moved to
    # uint16.  It is None on a wide-rank model, preventing lossy GPU upload.
    count_ranks8: np.ndarray | None
    shot_bits: np.ndarray
    shot_hash1: np.ndarray
    shot_hash2: np.ndarray
    dem_to_error: dict[int, int]
    observable_masks: tuple[int, ...]

    def repeated_shots(self, factor: int) -> "HostModel":
        """Repeat an exported batch to measure steady-state GPU throughput."""
        if factor < 1:
            raise ValueError("shot repeat factor must be positive")
        if factor == 1:
            return self
        stride = int(self.shot_offsets[-1])
        offset_parts = [self.shot_offsets]
        for repeat_index in range(1, factor):
            offset_parts.append(self.shot_offsets[1:] + repeat_index * stride)
        metadata = dict(self.metadata)
        metadata["shots"] = int(metadata["shots"]) * factor
        metadata["source_shot_repeat"] = factor
        metadata["cpu_reference_seconds"] = (
            float(metadata["cpu_reference_seconds"]) * factor
        )
        return replace(
            self,
            metadata=metadata,
            reference=self.reference * factor,
            shot_offsets=np.concatenate(offset_parts),
            shot_detections=np.tile(self.shot_detections, factor),
            shot_bits=np.tile(self.shot_bits, (factor, 1)),
            shot_hash1=np.tile(self.shot_hash1, factor),
            shot_hash2=np.tile(self.shot_hash2, factor),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HostModel":
        path = Path(path)
        metadata = json.loads((path / "metadata.json").read_text())
        reference = json.loads((path / "reference.json").read_text())
        detectors = int(metadata["num_detectors"])
        errors = int(metadata["num_errors"])
        shots = int(metadata["shots"])
        int32_max = int(np.iinfo(np.int32).max)
        if not (
            0 <= detectors <= int32_max
            and 0 <= errors <= int32_max
            and 0 <= shots <= int32_max
        ):
            raise ValueError(
                "GPU prototype dimensions must fit signed 32-bit kernel arguments"
            )
        d2e_offsets = _read(path, "d2e_offsets.u64", np.uint64)
        d2e = _read(path, "d2e.u32", np.uint32)
        # CTA chain rows and prefix positions use zero as a sentinel and store
        # flattened d2e positions plus one in u32.  Reject an export that could
        # truncate those positions instead of silently changing search order.
        if len(d2e) > int(np.iinfo(np.uint32).max):
            raise ValueError("GPU prototype d2e positions must fit uint32")
        edet_offsets = _read(path, "edet_offsets.u64", np.uint64)
        edets = _read(path, "edets.u32", np.uint32)
        eneighbor_offsets = _read(path, "eneighbor_offsets.u64", np.uint64)
        eneighbors = _read(path, "eneighbors.u32", np.uint32)
        weights = _read(path, "weights.f64", np.float64)
        error_to_dem = _read(path, "error_to_dem.u32", np.uint32)
        detector_order = _read(path, "detector_order.u32", np.uint32)
        observable_offsets = _read(path, "observable_offsets.u64", np.uint64)
        observables = _read(path, "observables.u32", np.uint32)
        shot_offsets = _read(path, "shot_offsets.u64", np.uint64)
        shot_detections = _read(path, "shot_detections.u32", np.uint32)
        rank_group = _read(path, "rank_group.u32", np.uint32)
        rank_lower = _read(path, "rank_lower.u32", np.uint32)
        count_ranks = _read(path, "count_ranks.u32", np.uint32)
        count_values = _read(path, "count_values.f64", np.float64)

        _validate_sorted_csr(
            eneighbor_offsets, eneighbors, errors, detectors, "eneighbor"
        )
        eneighbor_word_offsets, eneighbor_words, eneighbor_masks = _load_eneighbor_words(
            path, eneighbor_offsets, eneighbors, errors, detectors
        )
        if "eneighbors" in metadata and int(metadata["eneighbors"]) != len(eneighbors):
            raise ValueError("metadata eneighbor count does not match file")
        if (
            "eneighbor_word_groups" in metadata
            and int(metadata["eneighbor_word_groups"]) != len(eneighbor_words)
        ):
            raise ValueError("metadata packed eneighbor count does not match file")
        words = (detectors + 63) // 64
        if len(d2e_offsets) != detectors + 1 or len(edet_offsets) != errors + 1:
            raise ValueError("corrupt CSR dimensions")
        if (int(d2e_offsets[0]) != 0 or int(d2e_offsets[-1]) != len(d2e) or
                np.any(d2e_offsets[1:] < d2e_offsets[:-1])):
            raise ValueError("corrupt d2e CSR")
        if (int(edet_offsets[0]) != 0 or int(edet_offsets[-1]) != len(edets) or
                np.any(edet_offsets[1:] < edet_offsets[:-1])):
            raise ValueError("corrupt edet CSR")
        if len(d2e) and int(d2e.max()) >= errors:
            raise ValueError("out-of-range d2e error")
        if len(edets) and int(edets.max()) >= detectors:
            raise ValueError("out-of-range edet detector")
        if len(eneighbor_offsets) != errors + 1 or len(observable_offsets) != errors + 1:
            raise ValueError("corrupt error CSR dimensions")
        if (
            int(observable_offsets[0]) != 0
            or int(observable_offsets[-1]) != len(observables)
            or np.any(observable_offsets[1:] < observable_offsets[:-1])
        ):
            raise ValueError("corrupt observable CSR")
        num_observables = int(metadata.get("num_observables", 0))
        if num_observables < 0:
            raise ValueError("num_observables must be nonnegative")
        if len(observables) and int(observables.max()) >= num_observables:
            raise ValueError("out-of-range observable index")
        if len(weights) != errors or len(error_to_dem) != errors:
            raise ValueError("corrupt error arrays")
        if len(detector_order) != detectors or len(shot_offsets) != shots + 1:
            raise ValueError("corrupt detector order or shot offsets")
        if not metadata.get("rank_safe", False):
            raise ValueError("prototype requires a CPU-validated exact rank table")
        if not metadata.get("no_revisit_dets", False):
            raise ValueError("prototype requires no_revisit_dets")
        if int(metadata.get("detector_orders", 0)) != 1:
            raise ValueError("prototype requires exactly one detector order")
        rank_stride = int(metadata["rank_stride"])
        rank_groups = int(metadata["rank_groups"])
        if len(rank_group) != errors or len(rank_lower) != errors:
            raise ValueError("corrupt rank error arrays")
        if len(count_ranks) != rank_groups * rank_stride:
            raise ValueError("corrupt count rank table")
        if len(count_values) != rank_groups * rank_stride:
            raise ValueError("corrupt count value table")
        rank_maximum = int(metadata.get("rank_maximum", 0xffffffff))
        if not np.all(np.isfinite(weights)) or not np.all(weights > 0):
            raise ValueError("prototype requires finite positive weights")
        degrees = np.diff(edet_offsets)
        if len(degrees) and int(degrees.max()) > 16:
            raise ValueError("prototype requires compact error degree <= 16")
        if len(edets) > 1:
            support_boundary = np.zeros(len(edets), dtype=np.bool_)
            support_starts = edet_offsets[:-1]
            support_boundary[
                support_starts[support_starts < len(edets)].astype(np.intp, copy=False)
            ] = True
            if np.any((edets[1:] <= edets[:-1]) & ~support_boundary[1:]):
                raise ValueError("prototype requires sorted, unique error supports")
        for shot in range(shots):
            row = shot_detections[shot_offsets[shot] : shot_offsets[shot + 1]]
            if len(row) and (np.any(row[1:] <= row[:-1]) or int(row[-1]) >= detectors):
                raise ValueError("prototype requires sorted, unique, in-range shot detectors")

        detector_hash1 = np.fromiter(
            (_splitmix64(i ^ 0x243F6A8885A308D3) for i in range(detectors)),
            dtype=np.uint64,
            count=detectors,
        )
        detector_hash2 = np.fromiter(
            (_splitmix64(i ^ 0x13198A2E03707344) for i in range(detectors)),
            dtype=np.uint64,
            count=detectors,
        )
        error_hash1 = np.zeros(errors, dtype=np.uint64)
        error_hash2 = np.zeros(errors, dtype=np.uint64)
        for error in range(errors):
            begin, end = int(edet_offsets[error]), int(edet_offsets[error + 1])
            for detector in edets[begin:end]:
                error_hash1[error] ^= detector_hash1[int(detector)]
                error_hash2[error] ^= detector_hash2[int(detector)]

        shot_bits = np.zeros((shots, words), dtype=np.uint64)
        shot_hash1 = np.zeros(shots, dtype=np.uint64)
        shot_hash2 = np.zeros(shots, dtype=np.uint64)
        for shot in range(shots):
            begin, end = int(shot_offsets[shot]), int(shot_offsets[shot + 1])
            for detector_value in shot_detections[begin:end]:
                detector = int(detector_value)
                shot_bits[shot, detector >> 6] |= np.uint64(1) << np.uint64(detector & 63)
                shot_hash1[shot] ^= detector_hash1[detector]
                shot_hash2[shot] ^= detector_hash2[detector]

        d2e_ranked, count_ranks16, count_ranks8 = _pack_rank_tables(
            d2e, rank_group, rank_lower, count_ranks,
            rank_groups, rank_stride, rank_maximum,
        )
        rank_rows_monotonic = _rank_rows_are_monotonic(
            d2e_offsets, d2e, rank_lower
        )
        if (
            "rank_rows_monotonic" in metadata
            and bool(metadata["rank_rows_monotonic"]) != rank_rows_monotonic
        ):
            raise ValueError("metadata rank row monotonicity does not match files")

        # Convert each retained error's observable parity row to one Python
        # integer once.  Decoder output reconstruction then XORs a few cached
        # integers instead of repeatedly slicing NumPy CSR arrays on the host.
        # Python integers keep this exact for any validated observable count.
        observable_masks_list: list[int] = []
        for error in range(errors):
            begin = int(observable_offsets[error])
            end = int(observable_offsets[error + 1])
            mask = 0
            for observable in observables[begin:end]:
                mask ^= 1 << int(observable)
            observable_masks_list.append(mask)

        return cls(
            path=path,
            metadata=metadata,
            reference=reference,
            d2e_offsets=d2e_offsets,
            d2e=d2e,
            edet_offsets=edet_offsets,
            edets=edets,
            eneighbor_offsets=eneighbor_offsets,
            eneighbors=eneighbors,
            eneighbor_word_offsets=eneighbor_word_offsets,
            eneighbor_words=eneighbor_words,
            eneighbor_masks=eneighbor_masks,
            weights=weights,
            error_to_dem=error_to_dem,
            detector_order=detector_order,
            observable_offsets=observable_offsets,
            observables=observables,
            shot_offsets=shot_offsets,
            shot_detections=shot_detections,
            error_hash1=error_hash1,
            error_hash2=error_hash2,
            rank_group=rank_group,
            rank_lower=rank_lower,
            count_ranks=count_ranks,
            count_values=count_values,
            d2e_ranked=d2e_ranked,
            count_ranks16=count_ranks16,
            rank_rows_monotonic=rank_rows_monotonic,
            count_ranks8=count_ranks8,
            shot_bits=shot_bits,
            shot_hash1=shot_hash1,
            shot_hash2=shot_hash2,
            dem_to_error={int(dem): error for error, dem in enumerate(error_to_dem)},
            observable_masks=tuple(observable_masks_list),
        )

    def predictions_for_dem_errors(self, dem_errors: list[int]) -> list[int]:
        # error_to_dem contains the first original DEM index for each retained
        # error.  It is injective over retained errors, so invert once here.
        mask = 0
        for dem_error in dem_errors:
            error = self.dem_to_error[dem_error]
            mask ^= self.observable_masks[error]
        predictions: list[int] = []
        while mask:
            low_bit = mask & -mask
            predictions.append(low_bit.bit_length() - 1)
            mask ^= low_bit
        return predictions


class ExactGpuDecoder:
    def __init__(
        self,
        model: HostModel,
        workspace_capacity: int = 8192,
        max_output: int = 1024,
        threads_per_block: int = 1,
        mode: str = "warp",
        max_registers: int = 0,
        heap_cache: int = 0,
        warps_per_block: int = 1,
    ):
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.threads_per_block = int(threads_per_block)
        self.mode = mode
        self.max_registers = int(max_registers)
        self.heap_cache = int(heap_cache)
        self.warps_per_block = int(warps_per_block)
        if self.workspace_capacity < 2:
            raise ValueError("workspace_capacity must be at least 2")
        if self.threads_per_block < 1 or self.threads_per_block > 1024:
            raise ValueError("threads_per_block must be in [1, 1024]")
        if self.mode not in ("serial", "warp"):
            raise ValueError("mode must be 'serial' or 'warp'")
        if self.mode == "warp" and not np.array_equal(
            model.detector_order, np.arange(len(model.detector_order), dtype=np.uint32)
        ):
            raise ValueError("warp prototype currently requires identity detector order")
        if self.max_registers and not 32 <= self.max_registers <= 255:
            raise ValueError("max_registers must be zero or in [32, 255]")
        if self.heap_cache < 0 or self.heap_cache > self.workspace_capacity:
            raise ValueError("heap_cache must be in [0, workspace_capacity]")
        if self.warps_per_block not in (1, 2, 4, 8):
            raise ValueError("warps_per_block must be one of 1, 2, 4, 8")
        self.visited_capacity = _next_power_of_two(2 * self.workspace_capacity)
        compile_options = [
            "--std=c++11", "--fmad=false", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"
        ]
        if os.environ.get("TESSERACT_GPU_DEVICE_DEBUG"):
            # NVRTC disables device optimization with -G unless --dopt=on is
            # also supplied.  This gives a fast compile/debug path while the
            # optimized artifact remains the performance path.
            compile_options.append("--device-debug")
        if self.max_registers:
            compile_options.append(f"--maxrregcount={self.max_registers}")
        if self.heap_cache:
            compile_options.append(f"-DTESSERACT_HEAP_CACHE={self.heap_cache}")
        compile_options.append(f"-DTESSERACT_WARPS_PER_BLOCK={self.warps_per_block}")
        compile_options.append(
            "-DTESSERACT_MONOTONIC_RANK_ROWS="
            + ("1" if model.rank_rows_monotonic else "0")
        )
        self.module = cp.RawModule(
            code=KERNEL + WARP_KERNEL,
            options=tuple(compile_options),
            name_expressions=("decode_exact", "decode_exact_warp"),
            backend="nvrtc",
        )
        self.kernel = self.module.get_function(
            "decode_exact_warp" if self.mode == "warp" else "decode_exact"
        )
        self.device = {}
        for name in (
            "d2e_offsets", "d2e", "edet_offsets", "edets", "eneighbor_offsets",
            "eneighbors", "weights", "error_to_dem", "detector_order", "error_hash1",
            "error_hash2", "rank_group", "rank_lower", "count_ranks", "count_values",
            "d2e_ranked", "count_ranks16",
            "shot_offsets", "shot_detections", "shot_bits", "shot_hash1", "shot_hash2",
        ):
            self.device[name] = cp.asarray(getattr(model, name))
        self._workspace = None

    def _allocate_workspace(self) -> dict[str, cp.ndarray]:
        m = self.model.metadata
        shots = int(m["shots"])
        detectors = int(m["num_detectors"])
        errors = int(m["num_errors"])
        words = (detectors + 63) // 64
        n = shots * self.workspace_capacity
        v = shots * self.visited_capacity
        workspace = {
            "heap_cost": cp.empty(n, dtype=cp.float64),
            "heap_num_dets": cp.empty(n, dtype=cp.uint32),
            "heap_depth": cp.empty(n, dtype=cp.uint32),
            "heap_chain": cp.empty(n, dtype=cp.int32),
            "heap_hash1": cp.empty(n, dtype=cp.uint64),
            "heap_hash2": cp.empty(n, dtype=cp.uint64),
            "chain_error": cp.empty(n, dtype=cp.uint32),
            "chain_min_detector": cp.empty(n, dtype=cp.uint32),
            "chain_parent": cp.empty(n, dtype=cp.int32),
            "chain_row": cp.empty(n, dtype=cp.uint32),
            "path_stack": cp.empty(n, dtype=cp.int32),
            "parent_counts": cp.empty(shots * errors, dtype=cp.uint8),
            "child_counts": cp.empty(shots * errors, dtype=cp.uint8),
            "counted_bits": cp.empty(shots * words, dtype=cp.uint64),
            "base_stamp": cp.empty(shots * errors, dtype=cp.uint32),
            "prefix_stamp": cp.empty(shots * errors, dtype=cp.uint32),
            "parent_cost_cache": cp.empty(shots * detectors, dtype=cp.float64),
            "parent_cost_stamp": cp.empty(shots * detectors, dtype=cp.uint32),
            "state_bits": cp.empty(shots * words, dtype=cp.uint64),
            "child_bits": cp.empty(shots * words, dtype=cp.uint64),
            "compare_bits": cp.empty(shots * words, dtype=cp.uint64),
            "visited_hash1": cp.empty(v, dtype=cp.uint64),
            "visited_hash2": cp.empty(v, dtype=cp.uint64),
            "visited_chain": cp.empty(v, dtype=cp.uint32),
            "out_status": cp.empty(shots, dtype=cp.uint32),
            "out_length": cp.empty(shots, dtype=cp.uint32),
            "out_errors": cp.empty(shots * self.max_output, dtype=cp.uint32),
            "out_cost_bits": cp.empty(shots, dtype=cp.uint64),
            "out_pushes": cp.empty(shots, dtype=cp.uint32),
            "out_pops": cp.empty(shots, dtype=cp.uint32),
            "out_cycles": cp.empty(shots, dtype=cp.uint64),
        }
        self._workspace = workspace
        return workspace

    def decode(self) -> tuple[list[dict], dict]:
        workspace = self._workspace or self._allocate_workspace()
        m = self.model.metadata
        shots = int(m["shots"])
        detectors = int(m["num_detectors"])
        errors = int(m["num_errors"])
        words = (detectors + 63) // 64
        d = self.device
        w = workspace
        args = (
            d["d2e_offsets"], d["d2e"], d["edet_offsets"], d["edets"],
            d["eneighbor_offsets"], d["eneighbors"], d["weights"], d["error_to_dem"],
            d["detector_order"], d["error_hash1"], d["error_hash2"],
            d["rank_group"], d["rank_lower"], d["count_ranks"], d["count_values"],
            np.uint32(m["rank_stride"]),
            d["d2e_ranked"], d["count_ranks16"],
            d["shot_offsets"], d["shot_detections"], d["shot_bits"],
            d["shot_hash1"], d["shot_hash2"],
            np.int32(shots), np.int32(detectors), np.int32(errors), np.int32(words),
            np.uint32(m["beam"]), np.uint32(m["pqlimit"]), np.float64(m["det_penalty"]),
            np.int32(self.workspace_capacity), np.int32(self.visited_capacity),
            np.int32(self.max_output),
            w["heap_cost"], w["heap_num_dets"], w["heap_depth"], w["heap_chain"],
            w["heap_hash1"], w["heap_hash2"], w["chain_error"],
            w["chain_min_detector"], w["chain_parent"], w["chain_row"], w["path_stack"],
            w["parent_counts"], w["child_counts"], w["counted_bits"],
            w["base_stamp"], w["prefix_stamp"],
            w["parent_cost_cache"], w["parent_cost_stamp"],
            w["state_bits"], w["child_bits"], w["compare_bits"],
            w["visited_hash1"], w["visited_hash2"],
            w["visited_chain"], w["out_status"], w["out_length"], w["out_errors"],
            w["out_cost_bits"], w["out_pushes"], w["out_pops"], w["out_cycles"],
        )
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        start.record()
        for name in (
            "base_stamp", "prefix_stamp", "parent_cost_stamp", "parent_counts",
            "visited_chain",
        ):
            w[name].fill(0)
        if self.mode == "warp":
            block = 32 * self.warps_per_block
            self.kernel(((shots + self.warps_per_block - 1) // self.warps_per_block,),
                        (block,), args)
        else:
            block = self.threads_per_block
            self.kernel(((shots + block - 1) // block,), (block,), args)
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
        decisions = []
        for shot in range(shots):
            predicted_errors = [int(x) for x in errors_out[shot, : int(lengths[shot])]]
            decisions.append(
                {
                    "status": int(status[shot]),
                    "low_confidence": int(status[shot]) == 2,
                    "predicted_errors_populated": True,
                    "total_cost_bits": int(cost_bits[shot]),
                    "predicted_errors": predicted_errors,
                    "predictions": self.model.predictions_for_dem_errors(predicted_errors),
                }
            )
        stats = {
            "kernel_ms": kernel_ms,
            "workspace_capacity": self.workspace_capacity,
            "visited_capacity": self.visited_capacity,
            "threads_per_block": self.threads_per_block,
            "mode": self.mode,
            "max_registers": self.max_registers,
            "heap_cache": self.heap_cache,
            "warps_per_block": self.warps_per_block,
            "workspace_bytes": sum(int(x.nbytes) for x in workspace.values()),
            "status_counts": {int(x): int(np.count_nonzero(status == x)) for x in np.unique(status)},
            "max_pushes": int(pushes.max(initial=0)),
            "max_pops": int(pops.max(initial=0)),
            "median_pushes": float(np.median(pushes)),
            "median_pops": float(np.median(pops)),
        }
        if self.mode == "warp":
            stats["cycle_quantiles"] = {
                str(q): float(np.quantile(cycles, q))
                for q in (0.0, 0.5, 0.9, 0.99, 1.0)
            }
        if os.environ.get("TESSERACT_GPU_VERBOSE_SHOT_STATS"):
            stats["pushes"] = pushes.tolist()
            stats["pops"] = pops.tolist()
        return decisions, stats


def compare_with_reference(
    model: HostModel, decisions: list[dict], *, skip_workspace_overflow: bool = False
) -> list[dict]:
    mismatches = []
    fields = (
        "low_confidence",
        "predicted_errors_populated",
        "total_cost_bits",
        "predicted_errors",
        "predictions",
    )
    for shot, (actual, expected) in enumerate(zip(decisions, model.reference, strict=True)):
        if skip_workspace_overflow and actual["status"] == 3:
            continue
        differing = [field for field in fields if actual[field] != expected[field]]
        if differing:
            mismatches.append(
                {
                    "shot": shot,
                    "fields": differing,
                    "status": actual["status"],
                    "actual": {field: actual[field] for field in differing},
                    "expected": {field: expected[field] for field in differing},
                }
            )
    return mismatches


def main() -> None:
    import argparse

    # Repository experiments are required to consume at most one host CPU.
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

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("--workspace", type=int, default=8192)
    parser.add_argument("--max-output", type=int, default=1024)
    parser.add_argument("--threads-per-block", type=int, default=1)
    parser.add_argument("--mode", choices=("serial", "warp"), default="warp")
    parser.add_argument("--max-registers", type=int, default=0)
    parser.add_argument("--heap-cache", type=int, default=0)
    parser.add_argument("--warps-per-block", type=int, default=1)
    parser.add_argument("--tile-shots", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--allow-overflow", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    model = HostModel.load(args.model).repeated_shots(args.tile_shots)
    decoder = ExactGpuDecoder(
        model,
        workspace_capacity=args.workspace,
        max_output=args.max_output,
        threads_per_block=args.threads_per_block,
        mode=args.mode,
        max_registers=args.max_registers,
        heap_cache=args.heap_cache,
        warps_per_block=args.warps_per_block,
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
