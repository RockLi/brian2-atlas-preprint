"""Extend only the pinned native runner deadline; preserve transport API fixtures."""
from functools import wraps
import os
from pathlib import Path
from brian2_rust.training import NativeLIFTrainer
original = NativeLIFTrainer.__init__
@wraps(original)
def bounded_validation(self, *args, **kwargs):
    pinned = os.environ.get('B2_TRAIN_RUNNER')
    selected = kwargs.get('runner') or pinned
    if (pinned and selected and Path(selected).resolve() == Path(pinned).resolve()
            and 'request_timeout' not in kwargs):
        kwargs['request_timeout'] = 300
    return original(self, *args, **kwargs)
NativeLIFTrainer.__init__ = bounded_validation
