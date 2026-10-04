"""Verify the published outcomes and block timings without decoding."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
results=ROOT/'benchmarks/results/2026-10'
summary=json.loads((results/'summary.json').read_text(encoding='utf-8'))
for family,entry in summary.items():
    arrays=np.load(results/f'{family}_shot_decisions.npz')
    blocks=json.loads((results/f'{family}_blocks.json').read_text(encoding='utf-8'))
    for mode,s in entry['modes'].items():
        rows=blocks[mode]
        assert [r['begin'] for r in rows]==list(range(0,50000,1000))
        assert sum(r['count'] for r in rows)==s['shots']==50000
        assert abs(sum(r['wall_seconds'] for r in rows)-s['wall_seconds'])<1e-6
        for field in ('failures','flags','prediction_masks'):
            assert arrays[f'{mode}_{field}'].shape==(50000,)
        assert int(arrays[f'{mode}_failures'].sum())==s['errors']==sum(r['failure_count'] for r in rows)
        assert int(arrays[f'{mode}_flags'].sum())==s['flags']==sum(r['flag_count'] for r in rows)
        assert np.array_equal(arrays['original_failures'],arrays[f'{mode}_failures'])
        for row in rows:
            start=row['begin']; stop=start+row['count']
            assert int(arrays[f'{mode}_failures'][start:stop].sum())==row['failure_count']
            assert int(arrays[f'{mode}_flags'][start:stop].sum())==row['flag_count']
    print(f'{family}: all three 50000-shot outcomes and block timings verified')
for circuit in json.loads((ROOT/'benchmarks/circuits/manifest.json').read_text(encoding='utf-8'))['circuits']:
    assert hashlib.sha256((ROOT/circuit['circuit']).read_bytes()).hexdigest()==circuit['sha256']
print('Circuit identities verified; no decoding performed')
