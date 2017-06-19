from __future__ import absolute_import, division, print_function

import numpy as np
import quantities as pq


def _inf_rate(rate_map, px):
    '''
    A helper function for information rate.

    Originally from https://github.com/MattNolanLab/gridcells
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    return (np.nansum(np.ravel(tmp_rate_map * np.log2(tmp_rate_map/avg_rate) *
            px)), avg_rate)


def sparsity(rate_map, px):
    '''
    Compute sparsity of a rate map, The sparsity  measure is an adaptation
    to space. The adaptation measures the fraction of the environment  in which
    a cell is  active. A sparsity of, 0.1 means that the place field of the
    cell occupies 1/10 of the area the subject traverses [2]_

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        sparsity

    References
    ----------
    .. [2] Skaggs, W. E., McNaughton, B. L., Wilson, M., & Barnes, C. (1996).
       Theta phase precession in hippocampal neuronal populations and the
       compression of temporal sequences. Hippocampus, 6, 149-172.
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    avg_sqr_rate = np.sum(np.ravel(tmp_rate_map**2 * px))
    return avg_rate**2 / avg_sqr_rate


def selectivity(rate_map, px):
    '''
    "The selectivity measure max(rate)/mean(rate)  of the cell. The more
    tightly concentrated  the cell's activity, the higher the selectivity.
    A cell with no spatial tuning at all will  have a  selectivity of 1" [2]_.

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.

    Returns
    -------
    out : float
        selectivity
    '''
    tmp_rate_map = rate_map.copy()
    tmp_rate_map[np.isnan(tmp_rate_map)] = 0
    avg_rate = np.sum(np.ravel(tmp_rate_map * px))
    max_rate = np.max(np.ravel(tmp_rate_map))
    return max_rate / avg_rate


def information_rate(rate_map, px):
    '''
    Compute information rate of a cell given variable x.
    A simple algorithm devised by [1]_. This computes the spatial information
    rate of cell spikes given variable x (e.g. position, head direction) in
    bits/second. This function is copied from Lucas Solanka, Matt Nolans lab

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions. If units are in Hz, then
        the information rate will be in bits/s.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information rate.

    Notes
    -----
    Quote from [1]_:
    "To get the basic idea, imagine we are recording the activity of a neuron
    in the brain of a rat, while the rat is wandering around randomly on a
    circular platform. Suppose we observe that the cell fires only when the
    rat is on the left half of the platform, and that it fires at a constant
    rate everywhere on the left half; and suppose that on the whole the rat
    spends half of its time on the left half of the platform. In this case, if
    we are prevented from seeing where the rat is, but are informed that the
    neuron has just this very moment fired a spike, we obtain thereby one bit
    of information about the current location of the rat. Suppose we have a
    second cell, which fires only in the southwest quarter of the platform; in
    this case a spike would give us two bits of information. If there were in
    addition a small amount of background firing, the information would be
    slightly less than two bits. And so on.". Invalid positions are masked with
    ``np.nan`` and replaced with 0.

    References
    ----------
    .. [1] Skaggs, W.E. et al., 1993. An Information-Theoretic Approach to
       Deciphering the Hippocampal Code. In Advances in Neural Information
       Processing Systems 5. pp. 1030-1037.
    '''
    return _inf_rate(rate_map, px)[0] * pq.bit/pq.s


def information_specificity(rate_map, px):
    '''
    Compute the 'specificity' of the cell firing rate to a variable X.
    Compute :func:`information_rate` for this cell and divide by the average
    firing rate of the cell. See [1]_ for more information.

    Originally from https://github.com/MattNolanLab/gridcells

    Parameters
    ----------
    rate_map : numpy.ndarray
        A firing rate map, any number of dimensions.
    px : numpy.ndarray
        Probability density function for variable ``x``. ``px.shape`` must be
        equal ``rate_maps.shape``

    Returns
    -------
    I : float
        Information in bits/spike.
    '''
    I, avg_rate = _inf_rate(rate_map, px)
    return I / avg_rate


def prob_dist(x, y, bins):
    '''
    Calculate a probability distribution for animal positions in an arena.

    Parameters
    ----------
    x : quantities.Quantity array in m
    y : quantities.Quantity array in m
    bins : quantities.Quantity array in m

    Returns
    -------
    dist : numpy.ndarray
        Probability distribution for the positional data. The first dimension
        is the y axis, the second dimension is the x axis.
    '''

    H, _, _ = np.histogram2d(x, y, bins=bins, normed=False)
    return (H / len(x)).T


def occupancy_map(x, y, t,
                  binsize=0.01*pq.m,
                  box_xlen=1*pq.m,
                  box_ylen=1*pq.m,
                  mask_unvisited=True,
                  convolve=True,
                  return_bins=False,
                  smoothing=0.02):
    '''Divide a 2D space in bins of size binsize**2, count the time spent
    in each bin. The map can  be convolved with a gaussian kernel of size
    csize determined by the smoothing factor, binsize and box_xlen.

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    binsize : float
        spatial binsize
    box_xlen : quantities scalar in m
        side length of quadratic box
    mask_unvisited: bool
        mask bins which has not been visited by nans
    convolve : bool
        convolve the rate map with a 2D Gaussian kernel

    
    Returns
    -------
    occupancy_map : numpy.ndarray
    if return_bins = True
    out : occupancy_map, xbins, ybins
    '''

    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in [
            x, y, t] for var2 in [x, y, t]]):
        raise ValueError('x, y, t must have same number of elements')
    if box_xlen < x.max() or box_ylen < y.max():
        raise ValueError(
            'box length must be larger or equal to max path length')
    from decimal import Decimal as dec
    decimals = 1e10
    remainderx = dec(float(box_xlen)*decimals) % dec(float(binsize)*decimals)
    remaindery = dec(float(box_ylen)*decimals) % dec(float(binsize)*decimals)
    if remainderx != 0 or remaindery != 0:
        raise ValueError('the remainder should be zero i.e. the ' +
                         'box length should be an exact multiple ' +
                         'of the binsize')
    is_quantities([x, y, t], 'vector')
    is_quantities(binsize, 'scalar')
    t = t.rescale('s')
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    binsize = binsize.rescale('m').magnitude
    x = x.rescale('m').magnitude
    y = y.rescale('m').magnitude

    # interpolate one extra timepoint
    t_ = np.array(t.tolist() + [t.max() + np.median(np.diff(t))]) * pq.s
    time_in_bin = np.diff(t_.magnitude)
    xbins = np.arange(0, box_xlen + binsize, binsize)
    ybins = np.arange(0, box_ylen + binsize, binsize)
    ix = np.digitize(x, xbins, right=True)
    iy = np.digitize(y, ybins, right=True)
    time_pos = np.zeros((xbins.size, ybins.size))
    for n in range(len(x) - 1):
        time_pos[ix[n], iy[n]] += time_in_bin[n]
    # correct for shifting of map since digitize returns values at right edges
    time_pos = time_pos[1:, 1:]
    if convolve:
        rate[np.isnan(rate)] = 0.  # for convolution
        from astropy.convolution import Gaussian2DKernel, convolve_fft
        csize = (box_xlen / binsize) * smoothing
        kernel = Gaussian2DKernel(csize)
        rate = convolve_fft(rate, kernel)  # TODO edge correction
    if mask_unvisited:
        was_in_bin = np.asarray(time_pos, dtype=bool)
        rate[np.invert(was_in_bin)] = np.nan
    if return_bins:
        return rate.T, xbins, ybins
    else:
        return rate.T

    H, _, _ = np.histogram2d(x, y, bins=bins, normed=False)
    return (H / len(x)).T
