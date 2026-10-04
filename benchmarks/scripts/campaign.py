"""Serial, resumable benchmark with live exact CPU replay (Windows + WSL).

No cached reference decisions are used. All 50,000 formal shots, including
empty syndromes and low-confidence outputs, enter the LER denominator.
"""
import argparse
import copy
import ctypes
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'results_20261001'
GPU_DIR = ROOT / 'gpu'
WSL_ROOT = '/mnt/' + ROOT.drive[0].lower() + ROOT.as_posix()[2:] if os.name=='nt' else str(ROOT)
os.environ.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
sys.path.insert(0, str(GPU_DIR))
import numpy as np

def stamp(): return datetime.now(timezone.utc).isoformat()
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)
def state(stage, **fields):
    atomic(OUT / 'state.json', dict(stage=stage, updated=stamp(), pid=os.getpid(), **fields))
    print(stamp(), stage, fields, flush=True)
def linux(path):
    path = Path(path).resolve()
    if os.name!='nt': return path.as_posix()
    return '/mnt/' + path.drive[0].lower() + path.as_posix()[2:]
def wsl_command(binary, args):
    return ['wsl.exe', '-d', 'Ubuntu-24.04', '--cd', WSL_ROOT,
            'env', 'OMP_NUM_THREADS=1', 'OPENBLAS_NUM_THREADS=1',
            'taskset', '-c', '0', linux(HERE / 'bin' / binary), *args]

class Worker:
    def __init__(self, mode, circuit, model, budget, log):
        self.log = open(log, 'ab')
        self.p = subprocess.Popen(wsl_command(mode + '_worker', [linux(circuit), linux(model), '5', str(budget)]),
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log)
        self.ready = self.receive()
        if not self.ready.get('ready'): raise RuntimeError('CPU worker not ready')
    def receive(self):
        line = self.p.stdout.readline()
        if not line: raise RuntimeError(f'CPU worker exited {self.p.poll()}; inspect worker log')
        return json.loads(line)
    def decode(self, indices, full=False):
        self.p.stdin.write((json.dumps(dict(indices=list(indices), full=full))+'\n').encode()); self.p.stdin.flush()
        return self.receive()
    def close(self):
        try:
            if self.p.poll() is None:
                self.p.stdin.write(b'{"quit":true}\n'); self.p.stdin.flush(); self.p.wait(timeout=30)
        finally:
            if self.p.poll() is None: self.p.kill(); self.p.wait()
            self.log.close()

def export(circuit, shots, seed, budget, path):
    path.mkdir(parents=True, exist_ok=True)
    identity = dict(circuit_sha256=sha(circuit), shots=shots+64, seed=seed, beam=5, budget=budget,
                    exporter_sha256=sha(HERE/'bin/input_export'))
    marker = path / 'input_identity.json'
    if marker.exists():
        if json.loads(marker.read_text()) != identity: raise RuntimeError('input identity changed')
        return
    with open(path / 'export.log', 'ab') as log:
        subprocess.run(wsl_command('input_export', [linux(circuit), str(shots+64), str(seed), '5', str(budget), linux(path)]),
                       stdout=log, stderr=log, check=True)
    files = {f.name: sha(f) for f in path.iterdir() if f.suffix in ('.u64', '.u32', '.f64')}
    atomic(path / 'input_hashes.json', files)
    atomic(marker, identity)

def verify_inputs(model):
    for name, digest in json.loads((model/'input_hashes.json').read_text()).items():
        if sha(model/name) != digest: raise RuntimeError(f'input changed: {name}')

def view(model, start, stop):
    sliced = copy.copy(model); sliced.metadata = dict(model.metadata, shots=stop-start)
    a, b = map(int, model.shot_offsets[[start, stop]])
    sliced.shot_offsets = model.shot_offsets[start:stop+1] - a
    sliced.shot_detections = model.shot_detections[a:b]
    sliced.shot_bits = model.shot_bits[start:stop]
    sliced.shot_hash1 = model.shot_hash1[start:stop]
    sliced.shot_hash2 = model.shot_hash2[start:stop]
    sliced.reference = []
    return sliced

def gpu_block(backend, model, worker, start, stop):
    backend.model = view(model, start, stop)
    backend.selection = dataclasses.replace(backend.selection,
        features=dataclasses.replace(backend.selection.features, shots=stop-start))
    backend.runner.model = backend.model; backend.runner.source_shots = stop-start
    backend.cpu_exact_callback = lambda i: worker.decode([start+i], full=True)['decisions'][0]
    result, stats = backend.decode()
    if stats['gpu_backend_error']: raise RuntimeError('GPU backend failed: '+stats['gpu_backend_error'])
    return result, stats

def make_backend(model):
    import exact_gpu_backend as dispatch
    aligned_capacity=dispatch.max_chunk_detections
    def any_origin_capacity(source,chunk_size):
        # Formal blocks start after 64 held-out warmup shots and advance by
        # 1000. Their origins need not align with CUDA chunks. Bound every
        # contiguous window, including all possible block/chunk alignments.
        offsets=np.asarray(source.shot_offsets,dtype=np.uint64)
        n=len(offsets)-1
        if n<=chunk_size: return int(offsets[-1]-offsets[0])
        return int(np.max(offsets[chunk_size:]-offsets[:-chunk_size],initial=0))
    # Only the benchmark's allocation planner is overridden during construction.
    # Kernels, decoder state, input order and all original guards are unchanged.
    dispatch.max_chunk_detections=any_origin_capacity
    try:
        width=int(np.diff(model.d2e_offsets).max(initial=0))
        if width>255:
            from cta_compact_wide_decoder import ExactCtaDecoder
            return dispatch.ExactGpuBackend(model,backend_override='final',decoder_class=ExactCtaDecoder,
                                           cta_threads_override=256,heap_cache_override=128)
        return dispatch.ExactGpuBackend(model)
    finally:
        dispatch.max_chunk_detections=aligned_capacity

def summarize(directory, shots):
    rows = [json.loads(p.read_text()) for p in sorted(directory.glob('block_*.json'))]
    count = sum(r['count'] for r in rows)
    if count != shots: raise RuntimeError(f'incomplete experiment {count}/{shots}')
    errors = sum(sum(r['failures']) for r in rows)
    seconds = sum(r['wall_seconds'] for r in rows)
    flags = sum(sum(r['flags']) for r in rows)
    fallbacks = sum(r['fallbacks'] for r in rows)
    p = errors/count; z=1.959963984540054
    center=(p+z*z/(2*count))/(1+z*z/count)
    radius=z*((p*(1-p)/count+z*z/(4*count*count))**.5)/(1+z*z/count)
    summary = dict(shots=count, errors=errors, ler=p, ler_wilson95=[center-radius, center+radius],
                   flags=flags, fallbacks=fallbacks, wall_seconds=seconds, shots_per_second=count/seconds,
                   completed=stamp(), setup=json.loads((directory/'setup.json').read_text()))
    atomic(directory/'summary.json', summary)
    return summary

def paired_comparison(base_dir, candidate_dir):
    def load(d): return np.array([v for f in sorted(d.glob('block_*.json')) for v in json.loads(f.read_text())['failures']],dtype=np.int8)
    a,b=load(base_dir),load(candidate_dir)
    if len(a)!=50000 or len(b)!=50000: raise RuntimeError('comparison missing shots')
    plus=int(np.sum((a==0)&(b==1))); minus=int(np.sum((a==1)&(b==0)))
    n=len(a); z=2.241402727604947
    def wilson(k):
        p=k/n; c=(p+z*z/(2*n))/(1+z*z/n)
        h=z*((p*(1-p)/n+z*z/(4*n*n))**.5)/(1+z*z/n)
        return c-h,c+h
    lo_p,hi_p=wilson(plus); lo_m,hi_m=wilson(minus)
    interval=[lo_p-hi_m,hi_p-lo_m]; margin=.1*float(a.mean())
    return dict(delta_ler=float(b.mean()-a.mean()),difference_ci95_conservative=interval,
                discordant_failures=plus+minus,margin=margin,
                equivalent=bool(margin>0 and interval[0]>=-margin and interval[1]<=margin))

def experiment(mode, circuit, model_path, budget, directory, shots, formal=False):
    directory.mkdir(parents=True, exist_ok=True)
    if (directory/'summary.json').exists(): return json.loads((directory/'summary.json').read_text())
    verify_inputs(model_path)
    ready_start = time.perf_counter()
    worker = Worker('optimized' if mode=='gpu' else mode, circuit, model_path, budget, directory/'worker.log')
    backend = None
    try:
        truth_offsets=np.fromfile(model_path/'truth_offsets.u64',dtype=np.uint64)
        truth_obs=np.fromfile(model_path/'truth_observables.u32',dtype=np.uint32)
        truths=[{int(v) for v in truth_obs[int(truth_offsets[i]):int(truth_offsets[i+1])]} for i in range(shots+64)]
        gpu_plan = None
        if mode=='gpu':
            from gpu_decoder import HostModel
            model = HostModel.load(model_path)
            backend = make_backend(model)
            if backend.runner is None or backend.backend_error: raise RuntimeError(f'GPU initialization failed: {backend.backend_error}')
            gpu_plan = dataclasses.asdict(backend.plan)
            gpu_plan['implementation_module']=type(backend.decoder).__module__
            gpu_block(backend, model, worker, 0, 64)
        else: worker.decode(range(64))
        atomic(directory/'setup.json', dict(wall_seconds=time.perf_counter()-ready_start, cpu=worker.ready, gpu_plan=gpu_plan,
                                           warmup_shots=64, budget=budget, beam=5, mode=mode))
        block_size = 1000 if formal else 100
        for block, begin in enumerate(range(64, shots+64, block_size)):
            path=directory/f'block_{block:05d}.json'
            if path.exists(): continue
            end=min(begin+block_size,shots+64)
            start=time.perf_counter()
            if mode=='gpu':
                decisions, stats=gpu_block(backend,model,worker,begin,end)
                fallbacks=len(stats['cpu_fallback_shots'])
                accepted=len(stats['gpu_accepted_shots'])
            else:
                response=worker.decode(range(begin,end)); decisions=response['decisions']
                fallbacks=0; accepted=0
            elapsed=time.perf_counter()-start
            if len(decisions)!=end-begin: raise RuntimeError('missing decisions')
            failures=[int(set(map(int,d['predictions']))!=truths[i]) for i,d in zip(range(begin,end),decisions)]
            flags=[int(d['low_confidence']) for d in decisions]
            row=dict(begin=begin-64,count=end-begin,wall_seconds=elapsed, failures=failures, flags=flags,
                     prediction_masks=[sum(1<<int(j) for j in d['predictions']) for d in decisions],
                     fallbacks=fallbacks,gpu_accepted=accepted)
            atomic(path,row)
            state('formal_running' if formal else 'pilot_running', experiment=str(directory.relative_to(OUT)),
                  completed_shots=end-64, target=shots, first_block_verified=True)
        return summarize(directory,shots)
    finally:
        worker.close()
        if backend:
            del backend
            import cupy as cp
            cp.get_default_memory_pool().free_all_blocks()
