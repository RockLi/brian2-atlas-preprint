import unittest
from collections import Counter
import numpy as np
from .generalization_data import generalization_movie,digest
from .generalization_readout import transform,candidates
from .multispeed_readout import bank_features


class GeneralizationChecks(unittest.TestCase):
    def test_held_speed_closed_balanced_and_reversed(self):
        for speed in (3,5):
            hashes={k:Counter() for k in ('right','left','up','down')}
            for i in range(2):
                for j in range(2):
                    movies={k:generalization_movie(k,783,(i,j),speed) for k in hashes}
                    for k,m in movies.items():
                        np.testing.assert_array_equal(m[0],m[-1]);hashes[k][digest(m[0])]+=1
                    np.testing.assert_array_equal(movies['right'][::-1],movies['left'])
                    np.testing.assert_array_equal(movies['up'][::-1],movies['down'])
            self.assertTrue(all(v==hashes['right'] for v in hashes.values()))

    def test_reference_exact_and_all_bands_reverse(self):
        grids=np.random.default_rng(901).normal(size=(2,2,40,12,12))
        np.testing.assert_allclose(transform(grids,{'band':'p8','pool':False}),bank_features(grids,{'mode':'sum','speeds':[1,2]}),atol=1e-12)
        for recipe in candidates():
            x=transform(grids,recipe)
            np.testing.assert_allclose(transform(grids[:,:,::-1],recipe),-x,atol=1e-12)
            np.testing.assert_array_equal(transform(np.zeros_like(grids),recipe),np.zeros_like(x))
            self.assertEqual(x.shape[1],8 if recipe['pool'] else 24)


if __name__=='__main__':unittest.main()
