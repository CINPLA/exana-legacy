import numpy as np
from scipy import stats

def test_spatial_coherence():
    from exana.statistics.tools import spatial_coherence
    rm = np.array([1, 2, 3, 4])
    rm_target = np.array([2, 3])
    assert np.arctanh(spatial_coherence(rm, n_neighbours)) == np.arctanh(1)
