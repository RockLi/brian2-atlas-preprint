"""Verified held-speed study using the shared recorded-trial explorer."""
from .speed_replay import SpeedReplay
from .generalization_study import MODELS,CONDITIONS
from .generalization_data import generalization_movie
from .motion_refinement import read,sha
from pathlib import Path


class GeneralizationReplay(SpeedReplay):
    schema='flywire-speed-generalization-v1'
    model_names=MODELS
    condition_names=CONDITIONS
    movie_factory=staticmethod(generalization_movie)

    def __init__(self,directory):
        super().__init__(directory)
        root=Path(directory);self.input_audit=None
        if (root/'input-audit.json').exists():
            verified=read(root/'input-audit-verification.json');audit=read(root/'input-audit.json')
            if not verified['all_passed'] or sha(root/'input-audit.json')!=verified['audit_sha256'] or audit['features_sha256']!=self.report['features_sha256']:
                raise ValueError('input audit not verified')
            self.input_audit=audit
        self.totals_audit=None
        if (root/'totals-audit.json').exists():
            verified=read(root/'totals-audit-verification.json');audit=read(root/'totals-audit.json')
            if not verified['all_passed'] or sha(root/'totals-audit.json')!=verified['audit_sha256'] or audit['features_sha256']!=self.report['features_sha256'] or audit['report_sha256']!=sha(root/'report.json') or audit['model_sha256']!=sha(root/'totals-audit-model.npz'):
                raise ValueError('totals audit not verified')
            self.totals_audit=audit

    def index(self):
        return {**super().index(),'input_audit':self.input_audit,'totals_audit':self.totals_audit}
