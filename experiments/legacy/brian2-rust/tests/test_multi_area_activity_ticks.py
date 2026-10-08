"""The NEST physical observation window must use the prescribed grid map."""
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('activity_ticks', ROOT/'tools/analyze_multi_area_activity.py')
activity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(activity)


@pytest.mark.parametrize('end_tick',[25000,105000])
@pytest.mark.parametrize('offset,expected_bins,expected_times', [
    (0, (1, 2), [.5000, 2.4998, 2.4999]),
    (1, (2, 1), [.5000, .5001, 2.4999]),
])
def test_physical_window_boundaries_and_raster(monkeypatch, tmp_path, offset, expected_bins, expected_times,end_tick):
    model = {'definition': {'populations': [dict(name='mam_V1_23E', count=1,
             dt=struct.pack('>d', .0001).hex(), steps=end_tick, monitor={'window_steps': end_tick})]}}
    path = tmp_path/'model.json';path.write_text(json.dumps(model))
    results = tmp_path/'results';results.mkdir()
    for name in ['results.bin', 'events.bin']:
        (results/name).write_bytes(b'fixture')
    population = dict(spike_ticks=np.array([4998, 4999, 5000, end_tick-2, end_tick-1], dtype=np.int64),
                      indices=np.zeros(5, dtype=np.int64))
    monkeypatch.setattr(activity, 'load_results', lambda *args, **kwargs: {'populations': [population]})
    plots = []
    monkeypatch.setattr(activity, 'plot', lambda rows, areas, groups, rates, raster, window, output: plots.append((raster, window)))
    output = tmp_path/'activity'
    activity.analyze(path, results, output, spike_tick_offset=offset,end_tick=end_tick)
    with np.load(output/'activity-arrays.npz') as arrays:
        histogram = arrays['population_counts'][0]
        assert (histogram[0], histogram[-1]) == expected_bins
        assert histogram.sum() == 3
    report = json.loads((output/'activity.json').read_text())
    assert report['observed_spikes'] == 3 and report['mean_rate_hz'] == 3/((end_tick-5000)*.0001)
    assert report['window']['raw_start_tick'] == 5000-offset
    assert report['window']['raw_end_tick'] == end_tick-offset
    assert report['window']['spike_tick_offset'] == offset
    if end_tick==105000:
        expected_times=np.array([5000,end_tick-2,end_tick-1] if offset==0 else [5000,5001,end_tick-1])*.0001
    np.testing.assert_allclose(plots[0][0][0]['times'], expected_times, rtol=0, atol=1e-15)


@pytest.mark.parametrize('offset', [-1, 2, True, .5])
def test_invalid_grid_map_is_rejected_before_io(tmp_path, offset):
    with pytest.raises(ValueError, match='spike tick offset'):
        activity.analyze(tmp_path/'missing', tmp_path/'missing', tmp_path/'output', spike_tick_offset=offset)
