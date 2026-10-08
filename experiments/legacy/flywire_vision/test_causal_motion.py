import unittest
from collections import Counter
import hashlib
import numpy as np
from scipy.sparse import eye
from .causal_motion import motion_features, paired_interval, TRAVEL
from .motion_challenge import iter_orbit


class CausalMotionChecks(unittest.TestCase):
    def test_full_cycle_closes_and_each_static_input_has_all_labels(self):
        for seed in (1000000,1000001):
            counts={k:Counter() for k in ('right','left','up','down')}
            for row, frames in iter_orbit(seed,TRAVEL):
                np.testing.assert_array_equal(frames[0],frames[-1])
                counts[row['kind']][hashlib.sha256(frames[0].tobytes()).hexdigest()]+=1
            self.assertTrue(all(v==counts['right'] for v in counts.values()))
            self.assertEqual(sum(counts['right'].values()),4)

    def test_external_motion_feature_is_translation_invariant_and_reverses(self):
        data=np.zeros((8,12,12))
        for i in range(8):data[i,4,(2+i)%12]=1
        projection=eye(144,format='csr')
        zero=np.zeros_like(data).ravel()
        forward=motion_features(data.ravel(),zero,projection)
        reversed_=motion_features(data[::-1].ravel(),zero,projection)
        translated=motion_features(np.roll(data,(3,5),axis=(1,2)).ravel(),zero,projection)
        self.assertGreater(forward[0],0)
        np.testing.assert_allclose(forward,-reversed_,atol=1e-12)
        np.testing.assert_allclose(forward,translated,atol=1e-12)
        static=np.repeat(data[:1],8,axis=0)
        np.testing.assert_allclose(motion_features(static.ravel(),zero,projection),0,atol=1e-12)

    def test_paired_statistics_weight_whole_orbits(self):
        rows=[{'group':1}]*16+[{'group':2}]*16
        result=paired_interval([1]*16+[0]*16,[0]*32,rows)
        self.assertEqual(result['difference'],.5)
        self.assertEqual(result['group_bootstrap_95_interval'],[0.,1.])


if __name__=='__main__':unittest.main()
