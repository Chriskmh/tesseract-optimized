"""Exact paper Short orchestration: ordered beam-0..15 trials and strict selection.

Each trial uses the same graph and floating-point operations as single-order
CUDA. Only the first active detector lookup follows the exported permutation.
Failed device attempts require live CPU replay of that exact trial. No stored
reference decisions are consumed by this backend.
"""
from dataclasses import replace
import math
import struct
import time
import numpy as np
import cupy as cp
import cta_compact_final_nodebug_restrict_hot_decoder as base
import cta_compact_wide_decoder as wide
from exact_gpu_backend import ExactGpuBackend

_IDENTITY_SCAN = '''      // Exported exact models use the identity detector order.  Scan the
      // packed syndrome directly instead of testing every detector bit.
      for (int word = 0; word < num_words; ++word) {
        const u64 active = state_bits[word];
        if (active != 0) {
          s_min_detector = (u32)(word * 64 + __ffsll((long long)active) - 1);
          break;
        }
      }'''
_WHOLE_SCAN = '    if (tid == 0) {\n      s_min_detector = 0xffffffffU;\n' + _IDENTITY_SCAN + '\n    }\n    __syncthreads();'
_ORDERED_SCAN = '''    if (tid == 0) s_min_detector = 0xffffffffU;
    __syncthreads();
    u32 ordered_position = 0xffffffffU;
    for (int position = tid; position < num_detectors; position += CTA_THREADS) {
        const u32 detector = detector_order[position];
        if (bit_is_set(state_bits, detector)) {
          ordered_position = (u32)position;
          break;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
      const u32 other = __shfl_down_sync(0xffffffffU, ordered_position, offset);
      ordered_position = min(ordered_position, other);
    }
    if ((tid & 31) == 0 && ordered_position != 0xffffffffU)
      atomicMin(&s_min_detector, ordered_position);
    __syncthreads();
    if (tid == 0 && s_min_detector != 0xffffffffU)
      s_min_detector = detector_order[s_min_detector];
    __syncthreads();'''
if base.CTA_KERNEL.count(_WHOLE_SCAN) != 1:
    raise RuntimeError("Ordered CTA source anchor changed")

class OrderedCtaDecoder(base.ExactCtaDecoder):
    supports_detector_orders = True
    kernel_source = base.CTA_KERNEL.replace(_WHOLE_SCAN, _ORDERED_SCAN)

    def __init__(self, model, *args, **kwargs):
        order = np.asarray(model.detector_order)
        n = int(model.metadata['num_detectors'])
        if order.shape != (n,) or order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order),np.arange(n)):
            raise ValueError('detector_order must be a detector permutation')
        super().__init__(model, *args, **kwargs)

if wide.CTA_KERNEL.count(_WHOLE_SCAN) != 1:
    raise RuntimeError('Ordered wide CTA source anchor changed')

class OrderedWideCtaDecoder(wide.ExactCtaDecoder):
    supports_detector_orders = True
    kernel_source = wide.CTA_KERNEL.replace(_WHOLE_SCAN, _ORDERED_SCAN)

    def __init__(self, model, *args, **kwargs):
        order = np.asarray(model.detector_order)
        n = int(model.metadata['num_detectors'])
        if order.shape != (n,) or order.dtype.kind not in 'iu' or not np.array_equal(np.sort(order),np.arange(n)):
            raise ValueError('detector_order must be a detector permutation')
        super().__init__(model, *args, **kwargs)

class ShortGpuBackend:
    """Resident graph/workspace reused across all sixteen ordered trials.

    cpu_trial_callback(shot_index, trial_index) must decode the requested trial
    live with the identical graph/order/beam and return a full decision.
    """
    def __init__(self, model, orders, *, cpu_trial_callback=None, **gpu_options):
        self.model = model
        self.orders = np.asarray(orders,dtype=np.uint32)
        n = int(model.metadata['num_detectors'])
        if self.orders.shape != (16,n) or any(not np.array_equal(np.sort(o),np.arange(n)) for o in self.orders):
            raise ValueError('Short requires sixteen complete detector permutations')
        if int(model.metadata['pqlimit']) != 200000:
            raise ValueError('Short requires pqlimit 200000')
        self.cpu_trial_callback = cpu_trial_callback
        gpu_options.setdefault('workspace_override',200000)
        trial_model = replace(model,metadata=dict(model.metadata,beam=0,detector_orders=1),detector_order=self.orders[0].copy())
        decoder_class = OrderedWideCtaDecoder if int(np.diff(model.d2e_offsets).max(initial=0)) > 255 else OrderedCtaDecoder
        self.backend = ExactGpuBackend(trial_model,backend_override='final',decoder_class=decoder_class,
            cta_threads_override=256,heap_cache_override=128,**gpu_options)
        if not self.backend.resident:
            raise RuntimeError(f'Short CUDA initialization failed: {self.backend.backend_error}')

    def decode(self, *, trial_observer=None):
        begin = time.perf_counter()
        count = int(self.model.metadata['shots'])
        empty = dict(status=2,low_confidence=True,predicted_errors_populated=True,
                     total_cost_bits=0,predicted_errors=[],predictions=[])
        winners = [dict(empty) for _ in range(count)]
        # Match numeric_limits<double>::max(), strict <, earliest trial wins ties.
        costs = [float.fromhex('0x1.fffffffffffffp+1023')]*count
        trials = []
        for trial in range(16):
            resident = self.backend.runner
            resident.chunk_model.metadata['beam'] = trial
            resident.decoder.device['detector_order'].set(self.orders[trial])
            callback = None if self.cpu_trial_callback is None else lambda i,t=trial:self.cpu_trial_callback(i,t)
            decisions,stats = self.backend.decode(cpu_exact_callback=callback)
            if trial_observer is not None:
                trial_observer(trial, decisions, stats)
            for i,row in enumerate(decisions):
                cost = struct.unpack('<d',struct.pack('<Q',int(row['total_cost_bits'])))[0]
                if not row['low_confidence']:
                    if not math.isfinite(cost) or cost < 0:
                        raise RuntimeError('Invalid successful Short trial cost')
                    if cost < costs[i]:
                        costs[i] = cost
                        winners[i] = dict(row,status=1)
            trials.append(stats)
        # The original immediately returns a successful empty correction for
        # an empty syndrome, before running any ensemble trials.
        for i in range(count):
            if self.model.shot_offsets[i] == self.model.shot_offsets[i+1]:
                winners[i] = dict(empty,status=1,low_confidence=False)
        return winners,dict(profile='paper_short',trials=trials,trial_count=16,
            wall_seconds=time.perf_counter()-begin,
            gpu_accepted_trials=sum(len(s['gpu_accepted_shots']) for s in trials),
            cpu_replayed_trials=sum(len(s['cpu_fallback_shots']) for s in trials))

    def set_shots(self, model):
        """Select another shot view of the same immutable graph.

        The resident allocation must already cover its maximum chunk payload.
        All graph arrays must be shared, not replaced with another DEM.
        """
        for name in ('weights','d2e','edets','error_to_dem'):
            if getattr(model,name) is not getattr(self.model,name):
                raise ValueError('set_shots cannot change the decoding graph')
        self.model = model
        self.backend.model = model
        self.backend.selection = replace(self.backend.selection,
            features=replace(self.backend.selection.features,shots=int(model.metadata['shots'])))
        self.backend.runner.model = model
        self.backend.runner.source_shots = int(model.metadata['shots'])
