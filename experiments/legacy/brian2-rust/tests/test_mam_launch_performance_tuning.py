import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
import mam_launch_performance_tuning as launcher

PHASE = Path(__file__).resolve().parents[1]/'mpi-evidence/performance-protocol-v1'


@pytest.mark.parametrize('case_id,ranks,threads', [('nest48x4-tuning',48,4), ('nest96x2-tuning',96,2)])
def test_fixed_full_model_commands_and_collector_admission(case_id, ranks, threads, tmp_path):
    protocol = json.loads((PHASE/'protocol.json').read_text())
    case, label, options = launcher.options_for(protocol, case_id, tmp_path/'logs')
    app = options['application']
    assert options['ranks_per_node'] == ranks//6
    assert options['timeout'] == 1800
    assert options['guard_memory_mib'] == 262144
    assert options['guard_cpu_count'] == 32
    assert options['guard_min_free_gib'] == 128
    for flag, value in {'--ranks':str(ranks), '--threads':str(threads), '--duration-ms':'2500',
                        '--seed':'1729', '--max-neurons':'4200000', '--max-edges':'25000000000',
                        '--max-spikes-per-rank':'8388608'}.items():
        assert app[app.index(flag)+1] == value
    assert launcher.old.LABEL not in ' '.join(app)
    assert 'mam_nest_reference.py' not in ' '.join(app)
    assert app[app.index('--output')+1] == launcher.old.BASE+'/runs/'+label
    layout = json.loads((PHASE/(case['layout']+'-layout.json')).read_text())
    admission = launcher.admission_for(protocol, case, label, layout, options, {})
    assert len(admission['placements']) == ranks
    assert len(admission['guards']) == 7
    assert set(admission['source_catalog']) == set(launcher.SOURCES)
    from mam_collect_benchmark_terminal import host_code
    for i in range(6):
        assert label in host_code(admission, i, 'b2mpi-123456789abc')


def test_target_is_not_an_implicit_next_run(tmp_path):
    with pytest.raises(ValueError, match='only two'):
        launcher.options_for({}, 'nest-selected-target', tmp_path)


def test_receipt_never_overwrites(tmp_path):
    path = tmp_path/'receipt.json'
    launcher.write(path, {'first': True})
    with pytest.raises(FileExistsError): launcher.write(path, {'first': False})
    assert json.loads(path.read_text()) == {'first': True}
