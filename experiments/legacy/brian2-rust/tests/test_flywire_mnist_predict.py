from pathlib import Path
import json
import sys
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'validation'),str(ROOT/'experiments'),str(ROOT/'python')]
import flywire_mnist_predict as prediction
from flywire_mnist.config import Config, digest


def test_image_contract_has_no_implicit_scaling_or_dtype_conversion():
    image = np.arange(784,dtype=np.uint16).reshape(28,28).astype(np.uint8)
    result = prediction.validate_image(image)
    np.testing.assert_array_equal(result,image)
    result[0,0] = 255
    assert image[0,0] == 0
    for invalid in [image/255, image.astype(np.uint16), image.reshape(784), np.zeros((32,32),np.uint8)]:
        with pytest.raises(ValueError,match='uint8 28x28'):
            prediction.validate_image(invalid)


def test_unapproved_smoke_and_incomplete_artifacts_are_rejected(tmp_path):
    artifact = tmp_path/'artifact'; artifact.mkdir()
    (artifact/'protocol.json').write_text(json.dumps(dict(preset='smoke')))
    with pytest.raises(ValueError,match='explicit opt-in'):
        prediction.Predictor(tmp_path,artifact)
    p = dict(preset='standard'); p['sha256'] = digest(p)
    (artifact/'protocol.json').write_text(json.dumps(p))
    (artifact/'report.json').write_text(json.dumps(dict(status='running',protocol=p['sha256'])))
    with pytest.raises(ValueError,match='completed evaluation'):
        prediction.Predictor(tmp_path,artifact)


def test_failed_request_cleans_only_its_own_runtime(tmp_path,monkeypatch):
    runtime = tmp_path/'runtime'; runtime.mkdir()
    keep = runtime/'base.bin'; keep.write_bytes(b'frozen')
    class FailedRunner:
        directory = runtime
        def run(self,indices,ticks,key):
            (runtime/key).mkdir()
            (runtime/key/'partial').write_bytes(b'partial')
            (runtime/f'{key}.spikes').write_bytes(b'events')
            raise RuntimeError('simulated native failure')
    predictor = object.__new__(prediction.Predictor)
    predictor.closed = False; predictor.runner = FailedRunner()
    predictor.artifact = tmp_path; predictor.p = {}; predictor.cfg = Config()
    monkeypatch.setattr(prediction.full,'require_lock',lambda *args:None)
    with pytest.raises(RuntimeError,match='simulated native failure'):
        predictor.predict(np.zeros((28,28),np.uint8))
    assert list(runtime.iterdir()) == [keep]
    assert keep.read_bytes() == b'frozen'
    predictor.closed = True
    with pytest.raises(RuntimeError,match='closed'):
        predictor.predict(np.zeros((28,28),np.uint8))


@pytest.mark.parametrize('threads',[0,-1,257,True,1.5,'8'])
def test_invalid_thread_configuration_is_rejected_before_loading(tmp_path,threads):
    with pytest.raises(ValueError,match='threads must be an integer'):
        prediction.Predictor(tmp_path,tmp_path,threads=threads)
