# Complete Short profile

The Short profile follows the Tesseract paper's beam-climbing schedule:
sixteen coordinate detector orderings, beams 0 through 15, queue limit 200,000,
no-revisit enabled, and no detection penalty. Every implementation uses the
same resolved orderings. Successful trials replace the incumbent only at a
strictly lower cost, so equal-cost ties retain the earliest trial.

CUDA supports this through `gpu/short_gpu_backend.py`. The graph and workspace
remain resident while each trial uploads its detector permutation and sets its
beam. Ordered CUDA kernels retain exact search decisions and floating-point
costs. Wide detector rows use a wider state representation. Device workspace
exhaustion requests a live CPU replay of the corresponding trial; replay time
is included in the GPU timing.

Build the workers using the normal benchmark build command, then run:

```bash
python benchmarks/scripts/run_short.py --family bb --mode original --shots 64
python benchmarks/scripts/run_short.py --family bb --mode optimized --shots 64
python benchmarks/scripts/run_short.py --family bb --mode gpu --shots 64
```

Use `--family color` for the Superdense Color circuit, or `--circuit PATH` to
supply another circuit. Each mode generates/reuses the same inputs. Setup and
eight separate warmup shots are excluded. Saved eight-shot blocks contain full
decisions, logical failures, elapsed wall time and CUDA/replay trial counts.

The published beam-5 results remain a separate benchmark. Short needs its own
measurements; changing this profile can change both decoding time and LER.
