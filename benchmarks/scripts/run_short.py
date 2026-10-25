"""Run the complete paper Short profile on CPU or CUDA, with live exact replay."""
import argparse
import dataclasses
import os
from pathlib import Path
import sys
import time
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ROOT=Path(__file__).resolve().parents[2]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--family',choices=['bb','color'],required=True)
    p.add_argument('--mode',choices=['original','optimized','gpu'],required=True)
    p.add_argument('--circuit',type=Path,help='Use another circuit, e.g. a higher physical error rate')
    p.add_argument('--shots',type=int,default=64)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/short')
    args=p.parse_args()
    if args.shots<1:p.error('--shots must be positive')
    os.environ.setdefault('CUPY_CACHE_DIR',str(ROOT/'.cupy_cache'))
    import campaign as c
    if os.name!='nt':
        core=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{core})
        c.linux=lambda path:str(Path(path).resolve())
        c.wsl_command=lambda binary,argv:['taskset','-c',str(core),str(c.HERE/'bin'/binary),*argv]
    else:
        import ctypes
        ctypes.windll.kernel32.SetProcessAffinityMask(ctypes.windll.kernel32.GetCurrentProcess(),1)
    circuit=(args.circuit or ROOT/f'benchmarks/circuits/{args.family}.stim').resolve()
    if not circuit.is_file():p.error('circuit does not exist')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    identity=f'{args.family}_{c.sha(circuit)[:12]}_{args.shots}'
    inputs=output/'inputs'/identity
    c.export(circuit,args.shots,2026101900+(args.family=='color'),200000,inputs,'paper_short')
    c.verify_inputs(inputs)
    directory=output/identity/args.mode;directory.mkdir(parents=True,exist_ok=True)
    import json,numpy as np
    worker=c.Worker('optimized' if args.mode=='gpu' else args.mode,circuit,inputs,200000,directory/'worker.log','paper_short')
    try:
        backend=None
        if args.mode=='gpu':
            from gpu_decoder import HostModel
            from short_gpu_backend import ShortGpuBackend
            model=HostModel.load(inputs)
            orders=json.loads((inputs/'detector_orders.json').read_text())
            if worker.ready['detector_orders']!=orders:raise RuntimeError('CPU/CUDA detector orders differ')
            backend=ShortGpuBackend(model,orders,chunk_size_override=8)
        def decode(start,stop):
            if backend is None:return worker.decode(range(start,stop),full=True)['decisions'],{}
            backend.set_shots(c.view(model,start,stop))
            backend.cpu_trial_callback=lambda i,t:worker.decode([start+i],full=True,trial=t)['decisions'][0]
            return backend.decode()
        decode(0,8)
        c.atomic(directory/'setup.json',dict(profile='paper_short',beam=15,beam_climbing=True,orders=16,
            pqlimit=200000,warmup_shots=8,cpu=worker.ready,
            gpu_plan=dataclasses.asdict(backend.backend.plan) if backend else None,
            timing='Parent wall time, including all sixteen GPU trials, transfers, selection and live CPU replay; setup and warmup excluded.'))
        tptr=np.fromfile(inputs/'truth_offsets.u64',dtype='<u8')
        tobs=np.fromfile(inputs/'truth_observables.u32',dtype='<u4')
        for start in range(8,args.shots+8,8):
            stop=min(start+8,args.shots+8);begin=time.perf_counter()
            decisions,stats=decode(start,stop);seconds=time.perf_counter()-begin
            failures=[]
            for i,row in zip(range(start,stop),decisions):
                truth=set(map(int,tobs[int(tptr[i]):int(tptr[i+1])]))
                failures.append(int(set(row['predictions'])!=truth))
            c.atomic(directory/f'block_{start-8:05d}.json',dict(start=start-8,shots=stop-start,wall_seconds=seconds,
                decisions=decisions,failures=failures,gpu_accepted_trials=stats.get('gpu_accepted_trials',0),
                cpu_replayed_trials=stats.get('cpu_replayed_trials',0)))
            print(f'{args.mode} {stop-8}/{args.shots} shots, {seconds/(stop-start):.6f} s/shot',flush=True)
    finally:worker.close()

if __name__=='__main__':main()
