"""Sampling eligibility, reproducibility and physical-window invariants."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from analyze_mam_v1_subsample import SubsampleCounts,END

SAMPLES=[34,9,50,12,15,3,14,3]


def fixture():
    # 64 cells per population; ID0 is exactly at .56 Hz, all others above it.
    cells=np.repeat(np.arange(512),57)
    ticks=np.tile(np.r_[5001,5004,5005,np.arange(5050,5103),END],512)
    keep=~((cells%64==0)&(ticks==END))
    return ticks[keep],cells[keep]


def test_sampling_cutoff_and_binning_match_independent_histogram():
    ticks,cells=fixture();counter=SubsampleCounts([64]*8,SAMPLES,[64.2]*8)
    counter.first(ticks,cells);counter.select();counter.second(ticks,cells);counter.finish()
    for row in counter.selection:
        assert row['exact_cutoff_cells']==1
        assert row['wrapper_first_eligible']['population_local_ids']==list(range(1,row['samples']+1))
        assert row['wrapper_first_eligible']['eligible_cells']==63
        assert row['paper_random_eligible']['eligible_cells']==64
    for sampler in range(2):
        for population in range(8):
            keep=counter.selected[sampler,cells]&(cells//64==population)
            # Physical integer ticks: histogram includes5005, excludes5001/5004,
            # includes the terminal tick, and agrees without reusing bin indices.
            expected=np.histogram(ticks[keep],bins=100000,range=(5005,END+5))[0]
            np.testing.assert_array_equal(counter.bins[sampler,population],expected)
    other=SubsampleCounts([64]*8,SAMPLES,[64.2]*8)
    for i in range(0,len(cells),123):other.first(ticks[i:i+123],cells[i:i+123])
    other.select();np.testing.assert_array_equal(other.selected,counter.selected)


def test_missing_eligible_cells_refuses_to_reduce_sample():
    counter=SubsampleCounts([64]*8,SAMPLES,[64.2]*8)
    with pytest.raises(ValueError,match='insufficient eligible'):counter.select()


def test_changed_second_pass_cannot_publish():
    ticks,cells=fixture();counter=SubsampleCounts([64]*8,SAMPLES,[64.2]*8)
    counter.first(ticks,cells);counter.select();counter.second(ticks[:-1],cells[:-1])
    with pytest.raises(ValueError,match='changed between passes'):counter.finish()


def test_boundary_5000_is_frozen_but_never_eligibility_count():
    counter=SubsampleCounts([64]*8,SAMPLES,[64.2]*8)
    counter.first(np.array([5000,5001,END]),np.array([0,0,0]))
    assert counter.counts[0]==2 and counter.frozen.sum()==2
    with pytest.raises(ValueError):counter.first(np.array([END+1]),np.array([0]))
