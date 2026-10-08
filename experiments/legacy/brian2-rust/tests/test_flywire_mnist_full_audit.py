from pathlib import Path
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'), str(ROOT/'experiments'), str(ROOT/'python')]
import flywire_mnist_full_audit as audit
from flywire_mnist.protocol import atomic_json, file_sha


def phase_fixture(path):
    p = dict(preset='smoke', sha256='p', smoke_ids=dict(train=[3,17], test=[4,18]))
    ids = np.array([3,17], dtype=np.int64)
    x = np.arange(24, dtype=np.uint16).reshape(2,4,3)
    v = np.arange(16, dtype=np.uint16).reshape(2,4,2)
    (path/'shards').mkdir(); (path/'cache').mkdir()
    atomic_json(path/'cache/manifest.json', dict(protocol='p'))
    shard = path/'shards/train-000000.npz'
    np.savez_compressed(shard, ids=ids, features=x, projected=v, case_id='p',
                        payload_sha256=audit.full.r.payload_hash(ids,x,v), reused=0)
    for name, value in [('ids',ids), ('neural',x), ('projected',v)]:
        np.save(path/f'train-{name}.npy', value)
    manifest = dict(protocol='p', phase='train', count=2, shards={shard.name:file_sha(shard)},
                    files={f'train-{n}.npy':file_sha(path/f'train-{n}.npy') for n in ['ids','neural','projected']},
                    cache_manifest_sha256=file_sha(path/'cache/manifest.json'))
    atomic_json(path/'train-manifest.json', manifest)
    return p, manifest


def test_detects_assembled_content_corruption_even_with_updated_hash(tmp_path):
    p, manifest = phase_fixture(tmp_path)
    audit.audit_phase(tmp_path,p,'train',3,2)
    path = tmp_path/'train-neural.npy'
    x = np.load(path); x[1,2,0] += 1; np.save(path,x)
    manifest['files'][path.name] = file_sha(path)
    atomic_json(tmp_path/'train-manifest.json',manifest)
    with pytest.raises(ValueError,match='assembled rows differ'):
        audit.audit_phase(tmp_path,p,'train',3,2)


def test_rejects_missing_shard_coverage_and_permuted_ids(tmp_path):
    p, manifest = phase_fixture(tmp_path)
    original = manifest['shards']; manifest['shards'] = {}
    atomic_json(tmp_path/'train-manifest.json',manifest)
    with pytest.raises(ValueError,match='manifest shard coverage'):
        audit.audit_phase(tmp_path,p,'train',3,2)
    manifest['shards'] = original
    np.save(tmp_path/'train-ids.npy',np.array([17,3],dtype=np.int64))
    manifest['files']['train-ids.npy'] = file_sha(tmp_path/'train-ids.npy')
    atomic_json(tmp_path/'train-manifest.json',manifest)
    with pytest.raises(ValueError,match='canonical sample IDs'):
        audit.audit_phase(tmp_path,p,'train',3,2)


def test_rejects_landmark_data_and_parameter_mutations():
    raw = np.random.default_rng(4).integers(0,8,size=(20,4,5))
    labels = np.arange(20)%10; ids = np.arange(100,120)
    model,_ = audit.full.nystrom.fit(raw,labels,np.arange(20),landmarks=10,gamma=.25,alpha=.5,seed=783)
    model['landmark_ids'] = ids[model['landmark_rows']]
    choice = dict(gamma=.25, alpha_per_sample=.025, representation='sqrt_sum')
    p = dict(landmark_seed=783, landmarks=10)
    audit.audit_model(model,raw,ids,choice,p)
    model['reference'][0,0] += 1
    with pytest.raises(ValueError,match='landmark features'):
        audit.audit_model(model,raw,ids,choice,p)
    model['gamma'] = np.array(.5)
    with pytest.raises(ValueError,match='gamma changed'):
        audit.audit_model(model,raw,ids,choice,p)


def test_input_audit_matches_events_and_detects_window_swap():
    from scipy.sparse import csr_matrix
    from flywire_mnist.config import Config
    cfg = Config(seed=1783)
    images = np.random.default_rng(12).integers(0,256,size=(5,28,28),dtype=np.uint8)
    images[0] = 0; images[1] = 255
    matrix = csr_matrix(np.eye(784))
    projected = np.zeros((5,4,784),dtype=np.uint16)
    for row,image in enumerate(images):
        channels,ticks = audit.full.timing.regular_encode(image,row,cfg)
        for window,(left,right) in enumerate(zip(cfg.edges,cfg.edges[1:])):
            projected[row,window] = np.bincount(channels[(ticks>=left)&(ticks<right)],minlength=784)
    audit.audit_input(images,projected,cfg,matrix,batch_size=2)
    # Preserve total charge but move one actual pulse into a different window.
    row,window,channel = np.argwhere(projected>0)[0]
    projected[row,window,channel] -= 1
    projected[row,(window+1)%4,channel] += 1
    with pytest.raises(ValueError,match='regular pulses'):
        audit.audit_input(images,projected,cfg,matrix)


def test_full_training_equations_reject_fit_on_subset_and_changed_coefficients():
    raw = np.random.default_rng(18).integers(0,12,size=(40,4,5))
    labels = np.arange(40)%10
    model,_ = audit.full.nystrom.fit(raw,labels,np.arange(40),landmarks=20,gamma=.5,alpha=.1)
    result = audit.audit_fit_stationarity(model,raw,labels,batch_size=7)
    assert result['relative_kernel_gradient_norm'] < 1e-10
    subset,_ = audit.full.nystrom.fit(raw,labels,np.arange(30),landmarks=20,gamma=.5,alpha=.1)
    with pytest.raises(ValueError,match='normal equations'):
        audit.audit_fit_stationarity(subset,raw,labels)
    model['coefficients'][0,0] += .01
    with pytest.raises(ValueError,match='normal equations'):
        audit.audit_fit_stationarity(model,raw,labels)
