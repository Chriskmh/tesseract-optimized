"""CPU/GPU entry point on Linux or Windows with a WSL CPU worker.

GPU decisions use the resident exact backend and a live optimized CPU callback.
Input arrays are regenerated from the published circuits and recorded seeds.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/scripts'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['bb','color'],required=True)
    p.add_argument('--mode',choices=['original','optimized','gpu'],required=True)
    p.add_argument('--shots',type=int,default=128)
    p.add_argument('--output',type=Path,default=ROOT/'outputs')
    p.add_argument('--gpu-device', type=int, default=0, help='CUDA-visible device index')
    p.add_argument('--gpu-profile', choices=['auto','legacy','dedicated','shared'], default='auto')
    p.add_argument('--gpu-memory-gib', type=float, help='Maximum backend array-pool size, GiB')
    p.add_argument('--gpu-memory-share', type=float, help='Maximum fraction of visible total VRAM')
    p.add_argument('--gpu-compute-share', type=float, help='Advisory shot concurrency fraction; not a hard SM quota')
    p.add_argument('--gpu-max-chunk', type=int, help='Maximum resident shots per CUDA launch')
    p.add_argument('--gpu-pause-ms', type=float, help='Cooperative wait between CUDA chunks')
    p.add_argument('--gpu-reserve-mib', type=int, default=512, help='Leave this much currently free VRAM unallocated')
    p.add_argument('--gpu-free-memory-fraction', type=float, default=0.75, help='Use at most this fraction of currently free VRAM')
    args=p.parse_args()
    if args.shots<1: p.error('--shots must be positive')
    import math
    gpu_options = None
    gpu_requested = (args.gpu_device != 0 or args.gpu_profile != 'auto'
        or any(value is not None for value in (args.gpu_memory_gib, args.gpu_memory_share,
            args.gpu_compute_share, args.gpu_max_chunk, args.gpu_pause_ms))
        or args.gpu_reserve_mib != 512 or args.gpu_free_memory_fraction != 0.75)
    if args.mode != 'gpu' and gpu_requested:
        p.error('GPU resource options require --mode gpu')
    if args.gpu_device < 0 or args.gpu_reserve_mib < 0:
        p.error('GPU device and reserve must be nonnegative')
    if not math.isfinite(args.gpu_free_memory_fraction) or not 0 < args.gpu_free_memory_fraction <= 1:
        p.error('--gpu-free-memory-fraction must be in (0, 1]')
    if args.gpu_memory_gib is not None and (
        not math.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib <= 0):
        p.error('--gpu-memory-gib must be positive and finite')
    os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'.cupy_cache'))
    import campaign as c
    if args.mode == 'gpu':
        from resource_policy import GpuResources
        try:
            resources = GpuResources(profile=args.gpu_profile,
                memory_limit_bytes=None if args.gpu_memory_gib is None else int(args.gpu_memory_gib*(1<<30)),
                memory_share=args.gpu_memory_share, compute_share=args.gpu_compute_share,
                max_chunk_size=args.gpu_max_chunk, pause_ms=args.gpu_pause_ms)
        except ValueError as exc:
            p.error(str(exc))
        import cupy as cp
        cp.cuda.Device(args.gpu_device).use()
        if gpu_requested:
            gpu_options = dict(resources=resources, reserve_bytes=args.gpu_reserve_mib*(1<<20),
                               memory_fraction=args.gpu_free_memory_fraction)
    output=args.output.resolve();c.OUT=output;output.mkdir(parents=True,exist_ok=True)
    if os.name!='nt':
        core=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{core})
        c.linux=lambda path:str(Path(path).resolve())
        c.wsl_command=lambda binary, argv:['taskset','-c',str(core),str(c.HERE/'bin'/binary),*argv]
    else:
        import ctypes
        ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),1)
    for name in ['input_export','optimized_worker' if args.mode=='gpu' else args.mode+'_worker']:
        if not (c.HERE/'bin'/name).exists(): p.error('build the workers first: bash benchmarks/scripts/build_benchmark.sh')
    published=json.loads((ROOT/'benchmarks/results/2026-10/summary.json').read_text())[args.family]
    circuit=ROOT/published['circuit'];budget=published['pqlimit']
    seed=202610100 if args.family=='bb' else 202610101
    input_dir=output/'inputs'/f'{args.family}_{args.shots}'
    c.export(circuit,args.shots,seed,budget,input_dir)
    result=c.experiment(args.mode,circuit,input_dir,budget,output/args.family/args.mode,args.shots,
                        formal=args.shots>=1000,gpu_options=gpu_options)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
