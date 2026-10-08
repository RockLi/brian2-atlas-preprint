import unittest
import numpy as np
from .motion_readout import binned_residual, cell_weights, correlations, spectral_energy
from .motion_resolution import temporal_counts
from .motion_refinement import coarse, shuffle_readout_time


class MotionReadoutChecks(unittest.TestCase):
    def test_fine_bins_preserve_tick_boundaries_and_tail(self):
        result = temporal_counts([1,1,1,1,1,2], [999,1000,4999,5000,5999,6000], [1], 3)
        self.assertEqual(result.shape,(60,1))
        self.assertEqual(result[[9,10,49,50,59],0].tolist(),[1]*5)
        self.assertEqual(int(result.sum()),5)

    def test_residual_is_signed_and_window_has_no_warmup(self):
        raw = np.zeros((2,60,3),dtype=np.uint16)
        blank = np.ones((60,3),dtype=np.uint16)
        raw[:, :10] = 100
        result = binned_residual(raw,blank,dict(start=10,stop=60,width=5))
        np.testing.assert_array_equal(result,-np.ones((2,10,3))*5)

    def test_noise_weight_uses_only_the_supplied_fit_rows(self):
        raw = np.zeros((2,60,2));raw[1,:,0] = 2
        recipe=dict(start=10,stop=50,width=1,weight='std')
        np.testing.assert_allclose(cell_weights(raw,np.zeros((60,2)),recipe),[1,5])

    def test_coarse_unsigned_counts_allow_negative_baseline_residual(self):
        counts=np.zeros((2,60,3),dtype=np.uint16)
        blank=np.ones((60,3),dtype=np.uint16)
        np.testing.assert_array_equal(coarse(counts)-coarse(blank),-np.ones((2,24))*5)

    def test_shuffle_reorders_blank_with_counts_without_inventing_signal(self):
        blank=np.random.default_rng(12).integers(0,5,size=(60,3),dtype=np.uint16)
        arrays=dict(blank=blank,weights=np.ones(3),xy=np.array([[0.,0.],[.2,.2],[-.2,.3]]),valid=np.ones(3,dtype=bool),family=np.array([0,1,0]))
        recipe=dict(start=10,stop=50,width=1,separation='family',transform='signed',feature='spectral')
        np.testing.assert_array_equal(shuffle_readout_time(blank,arrays,recipe,1,[0,0]),0)

    def test_spectral_energy_prefers_forward_wave_and_reverses(self):
        t=np.arange(40)[:,None,None]
        x=np.arange(12)[None,None,:]
        wave=np.broadcast_to(1+np.cos(2*np.pi*(x/12-t/40)),(40,12,12))
        grid=np.stack([wave,wave])[None]
        recipe=dict(width=1,transform='both')
        forward=spectral_energy(grid,recipe)
        reverse=spectral_energy(grid[:,:,::-1],recipe)
        # The sign convention is learned from fit labels; energy must distinguish it.
        self.assertGreater(abs(forward[0,0]),.99)
        np.testing.assert_allclose(forward,-reverse,atol=1e-10)
        np.testing.assert_allclose(forward,spectral_energy(np.roll(grid,3,axis=4),recipe),atol=1e-10)

    def test_correlations_reverse_and_translate_without_metadata(self):
        grids = np.zeros((1,2,40,12,12))
        for t in range(40):grids[0,0,t,3,(t//3)%12] = 1
        for width in (1,2,5):
            for mode in ('both','signed'):
                recipe=dict(width=width,transform=mode)
                forward = correlations(grids,recipe)
                backward = correlations(grids[:,:,::-1],recipe)
                moved = correlations(np.roll(grids,(2,4),axis=(3,4)),recipe)
                np.testing.assert_allclose(forward,-backward,atol=1e-12)
                np.testing.assert_allclose(forward,moved,atol=1e-12)
                stationary = np.repeat(grids[:,:,:1],40,axis=2)
                np.testing.assert_allclose(correlations(stationary,recipe),0,atol=1e-12)


if __name__=='__main__':unittest.main()
