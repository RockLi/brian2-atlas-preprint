from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from audit_mam_original_statistics import adapt


class OriginalStatisticsTests(unittest.TestCase):
    def setUp(self):
        areas = ['A'+str(i) for i in range(31)] + ['TH']
        self.norm = {'populations': []}
        self.rates = {}; self.lvr = {}; self.corr = {}
        for area in areas:
            pops = ['23E', '23I', '5E', '5I', '6E', '6I']
            if area != 'TH':
                pops += ['4E', '4I']
            self.rates[area] = {'total': 2.}
            self.lvr[area] = {}; self.corr[area] = {}
            for pop in pops:
                self.norm['populations'].append(dict(name='mam_'+area+'_'+pop, official_normalization_neurons=10.5))
                self.rates[area][pop] = [2., float('nan')]
                self.lvr[area][pop] = 1.
                self.corr[area][pop] = -.01
        self.rates['Parameters'] = dict(t_min=500., t_max=10500., compute_stat=False, areas=areas)
        self.params = dict(T=10500., cc_weights_factor=1., cc_weights_I_factor=1.)

    def run_case(self):
        return adapt('ground10', self.rates, self.lvr, self.corr, self.params, self.norm)

    def test_unused_nan_and_negative_correlation_are_preserved_correctly(self):
        result = self.run_case()
        self.assertEqual(result['summary']['neuron_weighted_hz_using_modern_N'], 2.)
        self.assertTrue(all(r['unused_rate_statistic'] is None and r['correlation'] == -.01 for r in result['populations']))
        self.assertIsNone(result['metadata_by_metric']['lvr'])

    def test_absent_layer_slots_are_not_real_zero_rate_populations(self):
        self.rates['TH']['4E'] = [0., 0.]
        result = self.run_case()
        self.assertEqual(len(result['populations']), 254)
        self.assertEqual(len(result['excluded_absent_anatomy_placeholders']), 1)
        self.rates['TH']['4E'] = [1., 0.]
        with self.assertRaises(ValueError): self.run_case()

    def test_area_total_discrepancy_is_reported_without_overwriting(self):
        self.rates['TH']['total'] = 1.
        result = self.run_case()
        self.assertEqual(result['summary']['areas_with_consistency_error_above_1e_minus_10'], ['TH'])
        self.assertEqual(next(r for r in result['areas'] if r['area'] == 'TH')['published_rate_hz'], 1.)

    def test_nan_rate_and_modern_scalar_are_rejected(self):
        for value in [[float('nan'), 0.], 2.]:
            self.rates['TH']['23E'] = value
            with self.assertRaises(ValueError): self.run_case()

    def test_conflicting_window_and_unknown_slots_are_rejected(self):
        self.corr['Parameters'] = dict(t_min=500., t_max=2500.)
        with self.assertRaises(ValueError): self.run_case()
        del self.corr['Parameters']
        self.lvr['TH']['unknown'] = 0.
        with self.assertRaises(ValueError): self.run_case()


if __name__ == '__main__':
    unittest.main()
