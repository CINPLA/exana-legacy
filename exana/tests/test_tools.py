import numpy as np
import quantities as pq


def test_rescale_linear_track_2d_to_1d():
    from exana.tracking.tools import rescale_linear_track_2d_to_1d
    x = np.array([2.0, 2.5, 3.])*pq.m
    y = np.array([2.0, 2.5, 3.])*pq.m
    e0 = np.array([2.0, 2.0])*pq.m
    e1 = np.array([4., 4.])*pq.m
    x_rot = rescale_linear_track_2d_to_1d(x, y, e0, e1)
    assert np.round(x_rot[0], 4) == np.round(0.*pq.m, 4)
    assert np.round(x_rot[1], 4) == np.round(np.sqrt(2*0.5**2.)*pq.m, 4)
    assert np.round(x_rot[2], 4) == np.round(np.sqrt(2*1**2.)*pq.m, 4)
