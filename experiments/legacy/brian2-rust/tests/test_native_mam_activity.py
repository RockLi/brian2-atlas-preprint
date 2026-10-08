"""Native physical boundaries and bounded, population-local extraction."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('native_activity',ROOT/'tools/analyze_native_mam_activity.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def test_native_ticks_are_filtered_without_a_brian_offset(tmp_path):
    p=tmp_path/'rank.bin'
    np.array([(24999,4),(5000,0),(25000,1),(4999,2),(5001,3)],dtype=m.DTYPE).tofile(p)
    paths,totals=m.bucket_events([dict(count=2),dict(count=3)],[p],tmp_path/'split')
    assert totals.tolist()==[1,2]
    assert np.fromfile(paths[0],dtype=m.DTYPE).tolist()==[(5000,0)]
    assert sorted(np.fromfile(paths[1],dtype=m.DTYPE).tolist())==[(5001,1),(24999,2)]


def test_budget_rejected_before_output_and_invalid_ids_rejected(tmp_path):
    p=tmp_path/'rank.bin';np.array([(5000,0)],dtype=m.DTYPE).tofile(p)
    with pytest.raises(ValueError,match='input budget'):
        m.bucket_events([dict(count=1)],[p],tmp_path/'too_large',max_bytes=7)
    assert not (tmp_path/'too_large').exists()
    np.array([(4999,1)],dtype=m.DTYPE).tofile(p)
    with pytest.raises(ValueError,match='outside model'):
        m.bucket_events([dict(count=1)],[p],tmp_path/'bad_id')


def test_partial_record_rejected(tmp_path):
    p=tmp_path/'rank.bin';p.write_bytes(b'abc')
    with pytest.raises(ValueError,match='partial native'):
        m.bucket_events([dict(count=1)],[p],tmp_path/'partial')


def test_analysis_order_preserves_native_id_mapping(tmp_path):
    populations = [dict(name='V1-23E', area='V1', population='23E', count=2),
                   dict(name='46-23E', area='46', population='23E', count=3)]
    raw = tmp_path/'rank.bin'
    np.array([(5000,0),(5001,4)],dtype=m.DTYPE).tofile(raw)
    paths, totals = m.bucket_events(populations,[raw],tmp_path/'split')
    order = m.canonical_population_indices(populations)
    assert order == [1,0]
    assert [m.canonical_population_name(populations[i]) for i in order] == ['mam_46_23E','mam_V1_23E']
    assert [np.fromfile(paths[i],dtype=m.DTYPE).tolist() for i in order] == [[(5001,2)],[(5000,0)]]
    assert totals[order].tolist() == [1,1]
    with pytest.raises(ValueError,match='duplicate'):
        m.canonical_population_indices([populations[0],populations[0]])


def test_mixed_realizations_and_duplicate_ranks_are_rejected():
    reports = [dict(rank=i, seed=1730, ranks=48, threads=4,
                    duration_ms=2500, dt_ms=.1, nest_version='3.10.0')
               for i in range(48)]
    assert m.simulation_identity(reports)['seed'] == 1730
    reports[-1]['seed'] = 1729
    with pytest.raises(ValueError, match='mixed'):
        m.simulation_identity(reports)
    reports[-1]['seed'] = 1730
    reports[-1]['rank'] = 0
    with pytest.raises(ValueError, match='distinct'):
        m.simulation_identity(reports)


def test_long_native_identity_requires_explicit_duration():
    reports=[dict(rank=i,seed=1729,ranks=48,threads=4,duration_ms=10500,dt_ms=.1,nest_version='3.10.0') for i in range(48)]
    assert m.simulation_identity(reports,duration_ms=10500)['duration_ms']==10500
    with pytest.raises(ValueError,match='unexpected'):m.simulation_identity(reports)


@pytest.mark.parametrize('end_tick',[25000,105000])
def test_native_analysis_keeps_physical_edges_and_normalizes_halves(tmp_path,monkeypatch,end_tick):
    import gzip,hashlib,json
    raw=gzip.decompress((ROOT/'mpi-evidence/region/full-parameters.json.gz').read_bytes())
    params=json.loads(raw);assert params['total_neurons']==4129924
    audit=tmp_path/'audit';parameter=audit/'node25/fixture/parameters.json'
    parameter.parent.mkdir(parents=True);parameter.write_bytes(raw)
    folder=audit/'node25/runs/fixture';folder.mkdir(parents=True)
    middle=(5000+end_tick)//2
    events=np.array([(t,0) for t in [4999,5000,middle-1,middle,end_tick-1,end_tick]],dtype=m.DTYPE)
    for rank in range(48):
        file=folder/f'rank{rank}.events.bin'
        (events if rank==0 else np.empty(0,dtype=m.DTYPE)).tofile(file)
        report=dict(rank=rank,seed=1729,ranks=48,threads=4,duration_ms=end_tick//10,dt_ms=.1,nest_version='3.10.0',
                    parameters_sha256=hashlib.sha256(raw).hexdigest(),event_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),event_bytes=file.stat().st_size)
        (folder/f'rank{rank}.json').write_text(json.dumps(report))
    bins=np.bincount(events['tick'][events['tick']<end_tick]//500,minlength=end_tick//500)
    (audit/'summary.json').write_text(json.dumps(dict(passed=True,parameters_sha256=hashlib.sha256(raw).hexdigest(),physical_50ms_bin_counts=bins.tolist())))
    windows=[]
    monkeypatch.setattr(m,'plot',lambda rows,areas,groups,rates,raster,window,output:windows.append(window))
    out=tmp_path/'analysis';m.analyze(audit,out,end_tick=end_tick)
    result=json.loads((out/'activity.json').read_text());duration=(end_tick-5000)*.0001
    assert result['observed_spikes']==4 and result['mean_rate_hz']==4/4129924/duration
    assert windows[0]['end_tick']==end_tick and windows[0]['seconds']==duration
    area=next(r for r in result['areas'] if r['area']==params['populations'][0]['area'])
    assert area['first_half_rate_hz']==area['second_half_rate_hz']==2/area['neurons']/(duration/2)
    with np.load(out/'activity-arrays.npz') as arrays:
        assert arrays['population_counts'].shape==(254,(end_tick-5000)//10)
        assert arrays['population_counts'].sum()==4
    assert not (out/'population-events').exists()
    np.testing.assert_array_equal(np.fromfile(folder/'rank0.events.bin',dtype=m.DTYPE),events)
