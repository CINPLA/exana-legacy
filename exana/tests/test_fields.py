import numpy as np
import quantities as pq
import neo


def test_spatial_rate_map():
    from exana.tracking.fields import spatial_rate_map
    t = np.arange(10) * pq.s
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 0.1 * pq.m
    y = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * 0.1 * pq.m
    spike_times = np.linspace(0., 2, 5)*pq.s
    sptr = neo.SpikeTrain(times=spike_times, t_stop=10*pq.s)
    binsize = 1/2. * pq.m
    box_len = 1. * pq.m
    rate_map = spatial_rate_map(
        x,y, t, sptr,
        binsize=binsize,
        box_xlen=box_len,
        box_ylen=box_len,
        mask_unvisited=False,
        convolve=False)
    assert np.array_equal(rate_map, np.array([[1., 0],[0, 0]]))

    t = np.arange(3) * pq.s
    x = np.array([0.25,0.5,0.75]) * pq.m
    y = np.array([0.75,0.5,0.25]) * pq.m
    spike_times = np.arange(3) * pq.s
    sptr = neo.SpikeTrain(times=spike_times, t_stop=3*pq.s)
    binsize = 1./5 * pq.m
    box_len = 1. * pq.m
    rate_map = spatial_rate_map(
        x,y, t, sptr,
        binsize=binsize,
        box_xlen=box_len,
        box_ylen=box_len,
        mask_unvisited=False,
        convolve_spikes_and_time = True,
        smoothing = 0.1,
        convolve=True)
    # Do we like this behaviour?
    assert np.array_equal(rate_map, np.ones_like(rate_map))


def test_spatial_rate_map_1d():
    from exana.tracking.fields import spatial_rate_map_1d
    track_len = 10. * pq.m
    t = np.arange(10) * pq.s
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) * pq.m
    sptr = neo.SpikeTrain(times=np.linspace(0., 2, 5)*pq.s,
                          t_stop=10*pq.s)
    binsize = 5 * pq.m
    rate, bins = spatial_rate_map_1d(
        x, t, sptr,
        binsize=binsize,
        track_len=track_len,
        mask_unvisited=True,
        convolve=False,
        return_bins=True,
        smoothing=0.02)
    assert np.array_equal(rate, [1., 0])

if __name__ == "__main__":
    test_spatial_rate_map()
    test_spatial_rate_map_1d()
