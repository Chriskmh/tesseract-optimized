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
    args=p.parse_args()
    if args.shots<1: p.error('--shots must be positive')
    os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'.cupy_cache'))
    import campaign as c
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
    result=c.experiment(args.mode,circuit,input_dir,budget,output/args.family/args.mode,args.shots,formal=args.shots>=1000)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
