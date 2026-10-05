"""Host-only resource admission, topology and exact-replay regression tests."""
import dataclasses
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from exact_gpu_backend import ExactGpuBackend, plan_workspace, select_backend
from resident_chunk_runner import SHOT_BUFFER_NAMES
from resource_policy import DeviceInfo, GpuResources, resolve_resources


def model(shots=1024):
    return SimpleNamespace(
        metadata=dict(shots=shots, num_detectors=64, num_errors=8, rank_rows_monotonic=True),
        d2e_offsets=np.arange(65, dtype=np.uint64),
        shot_offsets=np.arange(shots + 1, dtype=np.uint64),
        shot_detections=np.arange(shots, dtype=np.uint32) % 64,
        shot_bits=np.ones((shots, 1), dtype=np.uint64),
        shot_hash1=np.arange(shots, dtype=np.uint64),
        shot_hash2=np.arange(shots, dtype=np.uint64),
    )


def decision(index, status=1):
    return dict(status=status, low_confidence=status == 2, predicted_errors_populated=True,
                total_cost_bits=index, predicted_errors=[index], predictions=[index % 2])


class FakeDecoder:
    """Encode source-shot identity to detect chunk padding/scatter mistakes."""
    def __init__(self, source, **kwargs):
        self.model = source
        self.device = {name: np.copy(getattr(source, name)) for name in SHOT_BUFFER_NAMES}

    def decode(self):
        rows = [decision(int(i), 3 if int(i) == 5 else 1) for i in self.device['shot_hash1']]
        return rows, dict(kernel_ms=0.0)


class ResourceTests(unittest.TestCase):
    small = DeviceInfo(0, 'GTX 1660 Ti', 24, 7, 5)
    large = DeviceInfo(0, 'synthetic Hopper', 132, 9, 0)

    def plan(self, resources, device=None, memory=(6 << 30, 6 << 30), **kwargs):
        source = model()
        return plan_workspace(source, select_backend(source, 'bb'), resources=resources,
                              device_info=device or self.small, memory_info=memory, **kwargs)

    def test_small_auto_preserves_entire_legacy_plan(self):
        legacy = dataclasses.asdict(self.plan(GpuResources(profile='legacy')))
        auto = dataclasses.asdict(self.plan(GpuResources()))
        legacy.pop('resource_policy'); auto.pop('resource_policy')
        self.assertEqual(legacy, auto)
        self.assertEqual(auto['chunk_size'], 64)
        self.assertEqual((auto['workspace_capacity'], auto['cta_threads'], auto['heap_cache']),
                         (100000, 640, 1024))

    def test_color_and_surface_large_source_keep_published_chunk_defaults(self):
        source = model()
        for backend in ('color', 'surface'):
            selection = select_backend(source, backend)
            legacy = plan_workspace(source, selection, memory_info=(6 << 30, 6 << 30),
                                    resources=GpuResources(profile='legacy'))
            auto = plan_workspace(source, selection, memory_info=(6 << 30, 6 << 30),
                                  device_info=self.small)
            left, right = dataclasses.asdict(legacy), dataclasses.asdict(auto)
            left.pop('resource_policy'); right.pop('resource_policy')
            self.assertEqual(left, right)
            self.assertEqual(auto.chunk_size, 256)

    def test_large_topology_scales_only_shots(self):
        result = self.plan(GpuResources(), self.large, (140 << 30, 141 << 30))
        self.assertEqual(result.chunk_size, 512)
        self.assertEqual((result.workspace_capacity, result.cta_threads, result.heap_cache),
                         (100000, 640, 1024))
        self.assertIn('unbenchmarked', result.resource_policy['reason'])

    def test_mig_or_mps_visible_slice_does_not_scale_parent_gpu(self):
        sliced = DeviceInfo(0, 'large GPU visible slice', 16, 9, 0)
        self.assertEqual(self.plan(GpuResources(), sliced).chunk_size, 64)
        self.assertEqual(self.plan(GpuResources(profile='shared'), sliced).chunk_size, 8)

    def test_shared_budget_intersects_total_share_free_headroom_and_cap(self):
        result = self.plan(GpuResources(profile='shared', memory_limit_bytes=200 << 20),
                           memory=(1 << 30, 6 << 30))
        self.assertEqual(result.usable_bytes, 200 << 20)
        self.assertLessEqual(result.estimated_device_bytes, result.usable_bytes)
        self.assertLessEqual(result.estimated_device_bytes,
                            result.usable_bytes-result.resource_policy['array_pool_headroom_bytes'])
        self.assertLessEqual(result.chunk_size, 12)
        self.assertEqual(result.workspace_capacity, 100000)
        result = self.plan(GpuResources(profile='shared'), memory=(6 << 30, 6 << 30))
        self.assertEqual(result.usable_bytes, int((6 << 30) * 0.25))

    def test_low_free_memory_shrinks_chunk_before_search_arena(self):
        result = self.plan(GpuResources(memory_limit_bytes=40 << 20))
        self.assertLess(result.chunk_size, 64)
        self.assertEqual(result.workspace_capacity, 100000)
        self.assertLessEqual(result.estimated_device_bytes, 40 << 20)

    def test_max_chunk_and_override_contract(self):
        self.assertEqual(self.plan(GpuResources(max_chunk_size=17), self.large).chunk_size, 17)
        with self.assertRaises(ValueError):
            self.plan(GpuResources(max_chunk_size=17), chunk_size_override=18)

    def test_explicit_share_caps_large_device_and_dedicated_mode(self):
        for profile in ('auto', 'dedicated', 'shared', 'legacy'):
            result = self.plan(GpuResources(profile=profile, compute_share=0.1), self.large)
            self.assertLessEqual(result.chunk_size, 14)

    def test_missing_topology_and_memory_are_conservative(self):
        def unavailable():
            raise RuntimeError('unavailable')
        resolved = resolve_resources(GpuResources(), 64, device_info=unavailable)
        self.assertEqual(resolved['preferred_chunk_size'], 64)
        self.assertIsNotNone(resolved['device_info_error'])
        self.assertEqual(resolve_resources(GpuResources(profile='shared'), 64,
                                          device_info=unavailable)['preferred_chunk_size'], 1)
        with self.assertRaises(MemoryError):
            self.plan(GpuResources(profile='shared'), memory=unavailable)

    def test_invalid_resource_limits(self):
        for options in (dict(profile='bad'), dict(memory_share=0), dict(compute_share=1.1),
                        dict(pause_ms=float('nan')), dict(max_chunk_size=0),
                        dict(memory_limit_bytes=-1)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                GpuResources(**options)

    def test_nondivisible_chunks_replay_in_order_without_losing_outputs(self):
        source = model(23)
        callbacks = []
        def cpu(index):
            callbacks.append(index)
            return decision(index)
        backend = ExactGpuBackend(source, decoder_class=FakeDecoder, backend_override='bb',
            memory_info=(6 << 30, 6 << 30), device_info=self.small,
            resources=GpuResources(max_chunk_size=7, pause_ms=2), cpu_exact_callback=cpu)
        with patch('resident_chunk_runner.time.sleep') as sleeper:
            rows, stats = backend.decode()
        self.assertEqual(rows, [decision(i) for i in range(23)])
        self.assertEqual(callbacks, [5])
        self.assertEqual(stats['cpu_fallback_shots'], [5])
        self.assertEqual(sleeper.call_count, 3)
        self.assertEqual(stats['gpu_stats']['chunks'][-1]['padding_shots'], 5)

    def test_no_memory_admission_replays_all_and_never_constructs_gpu(self):
        source = model(3)
        with patch.object(FakeDecoder, '__init__', side_effect=AssertionError('must not allocate')):
            backend = ExactGpuBackend(source, decoder_class=FakeDecoder,
                memory_info=(6 << 30, 6 << 30), device_info=self.small,
                resources=GpuResources(memory_limit_bytes=1), cpu_exact_callback=decision)
            rows, stats = backend.decode()
        self.assertEqual(rows, [decision(i) for i in range(3)])
        self.assertEqual(stats['cpu_fallback_shots'], [0, 1, 2])
        self.assertIsNone(backend.runner)


class ResumeTests(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'benchmarks/scripts'))
        import campaign
        self.campaign = campaign
        self.device = DeviceInfo(0, 'test device', 24, 7, 5)
        self.configuration = dict(resources=dataclasses.asdict(GpuResources()),
                                  reserve_bytes=512 << 20, memory_fraction=0.75,
                                  device=dataclasses.asdict(self.device))

    def run_existing(self, configuration, options=None, device=None):
        directory = Path('unused-test-output')
        records = {'setup.json': dict(gpu_configuration=configuration),
                   'summary.json': dict(shots=128)}
        with patch.object(Path, 'mkdir'), patch.object(Path, 'exists', return_value=True), \
             patch.object(Path, 'read_text', autospec=True,
                          side_effect=lambda path: json.dumps(records[path.name])), \
             patch('resource_policy.cuda_device_info', return_value=device or self.device):
            return self.campaign.experiment('gpu', None, None, 100000, directory, 128,
                                            gpu_options=options)

    def test_same_configuration_can_resume_without_decoding(self):
        self.assertEqual(self.run_existing(self.configuration), dict(shots=128))

    def test_changed_resources_or_device_cannot_mix_benchmark_results(self):
        with self.assertRaisesRegex(RuntimeError, 'configuration/device changed'):
            self.run_existing(self.configuration, dict(resources=GpuResources(profile='shared')))
        with self.assertRaisesRegex(RuntimeError, 'configuration/device changed'):
            self.run_existing(self.configuration, device=DeviceInfo(1, 'other device', 132, 9, 0))

    def test_legacy_records_reject_new_explicit_resources(self):
        with self.assertRaisesRegex(RuntimeError, 'Old run'):
            self.run_existing(None, dict(resources=GpuResources(profile='shared')))


if __name__ == '__main__':
    unittest.main()
