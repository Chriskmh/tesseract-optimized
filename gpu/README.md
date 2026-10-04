# GPU runtime

This directory contains the selected exact GPU backends and their runtime dependencies. Several base modules provide kernel source fragments used by the final implementations. They are required dependencies, not a collection of benchmark experiments. Use `python benchmarks/scripts/run_decoder.py --family bb --mode gpu --shots 128` from the repository root. CuPy compiles CUDA source at runtime. Resource exhaustion uses live exact CPU replay.
