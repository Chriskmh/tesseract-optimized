#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
BAZEL="${BAZEL:-bazel}"
core=$(python3 -c 'import os;print(min(os.sched_getaffinity(0)))')
taskset -c "$core" "$BAZEL" --host_jvm_args=-XX:ActiveProcessorCount=1 build --jobs=1 --loading_phase_threads=1 \
  //benchmarks/scripts:original_worker \
  //benchmarks/scripts:optimized_worker \
  //benchmarks/scripts:input_export
mkdir -p benchmarks/scripts/bin
for target in original_worker optimized_worker input_export; do
  cp "bazel-bin/benchmarks/scripts/$target" "benchmarks/scripts/bin/$target"
  chmod +x "benchmarks/scripts/bin/$target"
done
