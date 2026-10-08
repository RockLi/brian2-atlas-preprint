"""Primary raw files stay unopened until the terminal resource gate passes."""
import importlib
import json
from pathlib import Path

import pytest


def test_live_launch_rejected_before_any_raw_read(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'tools'))
    module = importlib.import_module('mam_primary_output_audit')
    cache = importlib.import_module('mam_correlation_stream')
    def forbidden(*a, **k):
        raise AssertionError('file hashing began before terminal gate')
    monkeypatch.setattr(cache, 'file_sha', forbidden)
    values = dict(admission=json.loads((root/'mpi-evidence/primary-run/admission.json').read_text()),
                  launch=dict(error=None, returncodes={'controller': None,
                    **{'proxy-'+str(r): None for r in range(4)}}), collected=[], runtime={})
    for name, value in values.items():
        (tmp_path/(name+'.json')).write_text(json.dumps(value))
    with pytest.raises(ValueError, match='not terminal'):
        module.run(tmp_path, tmp_path/'report.json')
    assert not (tmp_path/'report.json').exists()
