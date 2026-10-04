"""Exact BB candidate with one state hash and checked 32-bit CSR offsets.

The accepted CSR32 kernel carries two independent 64-bit Zobrist hashes for
each queued and visited state.  Both hashes are only filters: every matching
entry is reconstructed and compared bit for bit before it is treated as the
same syndrome.  This isolated candidate therefore keeps hash1 as the probe
filter and removes all runtime reads and writes of hash2.  The second-hash ABI
slots remain as unused positional placeholders so the inherited, well-tested
launcher can be reused; the host aliases those slots to hash1 and allocates no
second chain/visited arrays.

No dispatcher selects this module until strict per-shot A/B validation and a
real timing run show that it is worthwhile.
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

import cta_compact_fused_rank32_nodebug_csr32_decoder as base
from gpu_decoder import HostModel, compare_with_reference


def _replace_exact(source: str, old: str, new: str, expected: int = 1) -> str:
    observed = source.count(old)
    if observed != expected:
        raise RuntimeError(
            f"single-hash audit expected {expected} copies of {old!r}, "
            f"found {observed}"
        )
    return source.replace(old, new)


SINGLE_HASH_FIND = r"""__device__ __noinline__ int visited_find_single_hash_cta(
    u64 hash1, const u64 *candidate, int &empty_slot,
    const u64 *visited_hash1, const u32 *visited_chain,
    int table_capacity, const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u32 *edet_offsets, const u32 *edets) {
  int slot = (int)(hash1 & (u64)(table_capacity - 1));
  for (int probe = 0; probe < table_capacity; ++probe) {
    const u32 encoded = visited_chain[slot];
    if (encoded == 0) {
      empty_slot = slot;
      return 0;
    }
    // hash1 is only a rejection filter.  Equality is always decided by the
    // reconstructed syndrome, so a 64-bit collision cannot change an answer.
    if (visited_hash1[slot] == hash1) {
      const int existing_chain = (int)encoded - 2;
      if (state_equals_chain(candidate, existing_chain, initial_bits, scratch,
                             num_words, chain_error, chain_parent,
                             edet_offsets, edets)) {
        empty_slot = slot;
        return 1;
      }
    }
    slot = (slot + 1) & (table_capacity - 1);
  }
  empty_slot = -1;
  return -1;
}

"""


DUAL_VIRTUAL_FIND = r"""__device__ __noinline__ int visited_find_virtual_child_cta(
    u64 hash1, u64 hash2, const u64 *parent_bits, u32 error, int &empty_slot,
    const u64 *visited_hash1, const u64 *visited_hash2, const u32 *visited_chain,
    int table_capacity, const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u32 *edet_offsets, const u32 *edets) {
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
"""


SINGLE_VIRTUAL_FIND = r"""__device__ __noinline__ int visited_find_virtual_child_cta(
    u64 hash1, const u64 *parent_bits, u32 error, int &empty_slot,
    const u64 *visited_hash1, const u32 *visited_chain,
    int table_capacity, const u64 *initial_bits, u64 *scratch, int num_words,
    const u32 *chain_error, const int *chain_parent,
    const u32 *edet_offsets, const u32 *edets) {
  int slot = (int)(hash1 & (u64)(table_capacity - 1));
  for (int probe = 0; probe < table_capacity; ++probe) {
    const u32 encoded = visited_chain[slot];
    if (encoded == 0) {
      empty_slot = slot;
      return 0;
    }
    if (visited_hash1[slot] == hash1) {
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
"""


def _build_single_hash_kernel(source: str) -> str:
    """Remove runtime hash2 traffic while retaining positional kernel ABI."""
    transformed = _replace_exact(
        source, DUAL_VIRTUAL_FIND, SINGLE_HASH_FIND + SINGLE_VIRTUAL_FIND
    )
    replacements = (
        (
            "const u32 *detector_order, const u64 *error_hash1, const u64 *error_hash2,",
            "const u32 *detector_order, const u64 *error_hash1, "
            "const u64 *unused_error_hash2,",
        ),
        (
            "const u64 *initial_shot_bits, const u64 *shot_hash1, const u64 *shot_hash2,",
            "const u64 *initial_shot_bits, const u64 *shot_hash1, "
            "const u64 *unused_shot_hash2,",
        ),
        (
            "int *heap_chain_all, u64 *heap_hash1_all, u64 *heap_hash2_all,",
            "int *heap_chain_all, u64 *heap_hash1_all, u64 *unused_heap_hash2_all,",
        ),
        (
            "u64 *visited_hash1_all, u64 *visited_hash2_all, u32 *visited_chain_all,",
            "u64 *visited_hash1_all, u64 *unused_visited_hash2_all, "
            "u32 *visited_chain_all,",
        ),
        (
            "  u64 *chain_hash2 = heap_hash2_all + (u64)shot * workspace_capacity;\n",
            "",
        ),
        (
            "  u64 *visited_hash2 = visited_hash2_all + (u64)shot * visited_capacity;\n",
            "",
        ),
        ("        s_node.hash2 = shot_hash2[shot];\n", ""),
        ("        s_node.hash2 = chain_hash2[compact.chain];\n", ""),
        (
            "    const NodeValue node = s_node;\n",
            "    const NodeValue node{s_node.cost, s_node.num_dets, s_node.depth,\n"
            "                         s_node.chain, s_node.hash1, 0ULL};\n",
        ),
        (
            "      s_visited = visited_find(\n"
            "          node.hash1, node.hash2, state_bits, s_empty_slot,\n"
            "          visited_hash1, visited_hash2, visited_chain, visited_capacity,",
            "      s_visited = visited_find_single_hash_cta(\n"
            "          node.hash1, state_bits, s_empty_slot,\n"
            "          visited_hash1, visited_chain, visited_capacity,",
        ),
        ("      visited_hash2[s_empty_slot] = node.hash2;\n", ""),
        ("        const u64 child_hash2 = node.hash2 ^ error_hash2[error];\n", ""),
        (
            "            child_hash1, child_hash2, state_bits, error, child_empty,\n"
            "            visited_hash1, visited_hash2, visited_chain, visited_capacity,",
            "            child_hash1, state_bits, error, child_empty,\n"
            "            visited_hash1, visited_chain, visited_capacity,",
        ),
        ("        chain_hash2[next_chain] = child_hash2;\n", ""),
    )
    for old, new in replacements:
        expected = 2 if old.startswith("        const u64 child_hash2") else 1
        transformed = _replace_exact(transformed, old, new, expected)

    # Only the four explicitly named, unused ABI placeholders may mention the
    # second hash after the transformation.
    for forbidden in (
        "node.hash2", "child_hash2", "chain_hash2", "visited_hash2[",
        "error_hash2[", "shot_hash2[", "visited_find(\n",
    ):
        if forbidden in transformed:
            raise RuntimeError(f"second-hash runtime operation survived: {forbidden}")
    for required in (
        "visited_find_single_hash_cta", "visited_find_virtual_child_cta",
        "if (visited_hash1[slot] == hash1)",
        "state_equals_chain(candidate", "state_equals_virtual_child_cta(",
    ):
        if required not in transformed:
            raise RuntimeError(f"single-hash exactness guard missing: {required}")
    return transformed


CTA_KERNEL = _build_single_hash_kernel(base.CTA_KERNEL)
COMMON = base.COMMON
ALLOWED_CTA_THREADS = base.ALLOWED_CTA_THREADS
ALLOWED_HEAP_CACHE = base.ALLOWED_HEAP_CACHE


def nvrtc_compile_check(*, monotonic: bool = True,
                        cta_threads: int = 640,
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
        filename="cta_compact_fused_rank32_nodebug_csr32_singlehash_decoder.cu",
        name_expressions=("decode_exact_cta",), cache_in_memory=True,
    )


class ExactCtaDecoder(base.ExactCtaDecoder):
    """CSR32 winner with one exact visited-state hash filter."""

    def __init__(self, model: HostModel, workspace_capacity: int = 8192,
                 max_output: int = 1024, max_registers: int = 0,
                 cta_threads: int = 128, heap_cache: int = 128):
        self.csr32_offsets = base.csr32.pack_model_csr32_offsets(model)
        self.model = model
        self.workspace_capacity = int(workspace_capacity)
        self.max_output = int(max_output)
        self.cta_threads = int(cta_threads)
        self.heap_cache = int(heap_cache)
        self.visited_capacity = base.rank32._next_power_of_two(  # pylint: disable=protected-access
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
            base.rank32._validate_fused_state_model(model)  # pylint: disable=protected-access
        )
        if not np.array_equal(
            model.detector_order,
            np.arange(len(model.detector_order), dtype=np.uint32),
        ):
            raise ValueError("CTA prototype currently requires identity detector order")
        self.d2e_ranked32, self.rank32_layout = base.rank32.repack_d2e_ranked32(model)

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
            "count_values", "count_ranks16", "shot_offsets",
            "shot_detections", "shot_bits", "shot_hash1",
        ):
            source = packed_offsets.get(name, getattr(model, name))
            self.device[name] = cp.asarray(source)
        # Keep positional launcher compatibility without a second allocation.
        self.device["error_hash2"] = self.device["error_hash1"]
        self.device["shot_hash2"] = self.device["shot_hash1"]
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
        workspace = {
            "heap_cost": cp.empty(n, cp.float64),
            "heap_num_dets": cp.empty(n, cp.uint32),
            "heap_depth": cp.empty(n, cp.uint32),
            "heap_chain": cp.empty(n, cp.int32),
            "heap_hash1": cp.empty(n, cp.uint64),
            "chain_error": cp.empty(n, cp.uint32),
            "chain_min_detector": cp.empty(n, cp.uint32),
            "chain_parent": cp.empty(n, cp.int32),
            "chain_row": cp.empty(n, cp.uint32),
            "chain_depth": cp.empty(n, cp.uint32),
            "path_stack": cp.empty(n, cp.int32),
            "error_state": cp.empty(shots * errors, cp.uint32),
            "counted_bits": cp.empty(shots * words, cp.uint64),
            "parent_cost_cache": cp.empty(shots * detectors, cp.float64),
            "parent_cost_stamp": cp.empty(shots * detectors, cp.uint32),
            "state_bits": cp.empty(shots * words, cp.uint64),
            "child_bits": cp.empty(shots * words, cp.uint64),
            "compare_bits": cp.empty(shots * words, cp.uint64),
            "visited_hash1": cp.empty(v, cp.uint64),
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
        # These names are ABI placeholders in the inherited launcher.  They
        # alias hash1 and therefore consume no separate device storage.
        workspace["heap_hash2"] = workspace["heap_hash1"]
        workspace["visited_hash2"] = workspace["visited_hash1"]
        return workspace

    def decode(self) -> tuple[list[dict], dict]:
        decisions, stats = super().decode()
        stats["single_hash_exact_filter"] = True
        stats["second_hash_storage"] = False
        return decisions, stats


def main() -> None:
    base.rank32._pin_one_cpu()  # pylint: disable=protected-access
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
