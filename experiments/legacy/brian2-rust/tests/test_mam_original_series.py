from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from audit_mam_original_series import series_contract, psd_comparison
from mam_paper_spectrum import spectrum


class OriginalSeriesTests(unittest.TestCase):
    def test_resolution_conflict_is_reported_not_repaired(self):
        metadata=dict(t_min=500.,t_max=10500.,resolution=5.)
        result=series_contract(np.zeros(10000),metadata)
        self.assertFalse(result['resolution_metadata_consistent'])
        self.assertEqual(result['implied_resolution_ms_from_span_and_length'],1.)
        self.assertEqual(metadata['resolution'],5.)

    def test_invalid_series_rejected(self):
        for a in [np.zeros((2,4)),np.array([float('nan')]),np.zeros(100001)]:
            with self.assertRaises(ValueError):series_contract(a,dict(t_min=500.,t_max=10500.,resolution=1.))

    def test_different_psd_remains_explicitly_unresolved(self):
        rate=np.sin(np.arange(10000)*.13)+2.
        f,p=spectrum(rate)
        self.assertTrue(psd_comparison(rate,f,p)['reconstruction_within_arithmetic_tolerance'])
        self.assertFalse(psd_comparison(rate,f,p*1.1)['reconstruction_within_arithmetic_tolerance'])
        self.assertFalse(psd_comparison(rate,f*2,p)['frequency_grid_exact'])


if __name__=='__main__':unittest.main()
