# GPU configuration

GPU profiles control batch size, memory use and concurrency while preserving
search settings and exact decoding behavior.

## Choose a profile

| Profile | Recommended use |
| --- | --- |
| `auto` (default) | Adapt batch sizes to the available GPU resources. |
| `legacy` | Use the configuration from the published benchmarks. |
| `dedicated` | Run on a GPU allocated exclusively to this workload. |
| `shared` | Limit memory and concurrency when sharing a GPU. |

`auto` retains the existing GTX 1660 Ti settings. Device selection respects
`CUDA_VISIBLE_DEVICES`; profiles use the resources visible to CUDA, including
MIG and MPS allocations.

## Examples

```bash
# Default device-aware configuration.
python benchmarks/scripts/run_decoder.py --family bb --mode gpu --shots 128

# Published benchmark configuration.
python benchmarks/scripts/run_decoder.py --family bb --mode gpu --shots 128 \
  --gpu-profile legacy --output outputs/legacy

# Shared GPU with a 4 GiB array-memory ceiling and reduced concurrency.
python benchmarks/scripts/run_decoder.py --family color --mode gpu --shots 128 \
  --gpu-profile shared --gpu-memory-gib 4 --gpu-compute-share 0.5 \
  --output outputs/shared
```

## Common options

| Option | Purpose |
| --- | --- |
| `--gpu-device` | Select a CUDA-visible GPU index; default 0. |
| `--gpu-memory-gib` | Set an absolute limit for the backend's GPU array memory. |
| `--gpu-memory-share` | Set its maximum share of visible GPU memory. |
| `--gpu-compute-share` | Adjust shot concurrency relative to visible compute resources. |
| `--gpu-max-chunk` | Limit the number of resident shots per launch. |
| `--gpu-pause-ms` | Add a wait between batches. |

The shared profile defaults to a 25% array-memory share and a 0.5 concurrency
share. Memory allocation also accounts for currently free VRAM. CUDA runtime and
other library allocations are outside the array-memory limit. Compute shares
control concurrency; use MPS/MIG for strict resource isolation.

Use a new output directory when changing devices or resource settings. For all
options, run `python benchmarks/scripts/run_decoder.py --help`.

## Python API

```python
import sys
sys.path.insert(0, "gpu")
from exact_gpu_backend import ExactGpuBackend, GpuResources

backend = ExactGpuBackend(
    model,
    cpu_exact_callback=decode_one_on_cpu,
    resources=GpuResources(profile="shared", memory_limit_bytes=4 << 30,
                           compute_share=0.5),
)
decisions, stats = backend.decode()
```

The backend retains exact CPU replay for cases that cannot complete on the GPU.
Configuration and validation records are available in `setup.json` and
[the resource validation record](GPU_RESOURCE_VALIDATION.json).
