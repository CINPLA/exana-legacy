# -*- coding: utf-8 -*-
"""
Ahthor Chris Rodger

Originally from OpenElectrophy, with the following licence information:

    OpenElectrophy is released under CeCill-B.

    This licence is compatible BSD and compatible with the french copyright laws.

    For short, you can use it and modify it as you want.
"""

import numpy as np

def correlogram(t1, t2=None, bin_width=.001, limit=.02, auto=False,
                density=False):
    """Return crosscorrelogram of two spike trains.

    Essentially, this algorithm subtracts each spike time in `t1`
    from all of `t2` and bins the results with np.histogram, though
    several tweaks were made for efficiency.

    Parameters
    ---------
        t1 : np.array
            First spiketrain, raw spike times in seconds.
        t2 : np.array
            Second spiketrain, raw spike times in seconds.
        bin_width : float
            Width of each bar in histogram in seconds.
        limit : float
            Positive and negative extent of histogram, in seconds.
        auto : bool
            If True, then returns autocorrelogram of `t1` and in
            this case `t2` can be None. Default is False.
        density : bool
            If True, then returns the probability density function.

    See also
    --------
    numpy.histogram : The histogram function in use.

    Returns
    -------
        (count, bins) : tuple
            a tuple containing the bin edges (in seconds) and the
            count of spikes in each bin.

    Note
    ----
        `bins` are relative to `t1`. That is, if `t1` leads `t2`, then
        `count` will peak in a positive time bin.

    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> limit = 1
    >>> bin_width = .2
    >>> counts, bins = correlogram(t1=t1, t2=t2, bin_width=bin_width,
    ...                            limit=limit, auto=False)
    >>> counts
    array([ 0,  0,  0,  3,  3, 13,  5,  1,  0,  0])

    The correlogram algorithm is identical to, but computationally faster than
    the histogram of differences of each timepoint, i.e.
    >>> diff = [t2 - t for t in t1]
    >>> counts2, bins = np.histogram(diff, bins=bins)
    >>> np.array_equal(counts2, counts)
    True

    """
    # For auto-CCGs, make sure we use the same exact values
    # Otherwise numerical issues may arise when we compensate for zeros later
    if auto: t2 = t1

    # For efficiency, `t1` should be no longer than `t2`
    swap_args = False
    if len(t1) > len(t2):
        swap_args = True
        t1, t2 = t2, t1

    # Sort both arguments (this takes negligible time)
    t1 = np.sort(t1)
    t2 = np.sort(t2)

    # Determine the bin edges for the histogram
    # Later we will rely on the symmetry of `bins` for undoing `swap_args`
    limit = float(limit)
    bins = np.linspace(-limit, limit, num=int(2 * limit/bin_width + 1))

    # The numpy.arange method overshoots slightly the edges i.e. bin_width + epsilon
    # which leads to inclusion of spikes falling on edges.
    # bins = np.arange(-limit, limit + bin_width, bin_width)

    # Determine the indexes into `t2` that are relevant for each spike in `t1`
    ii2 = np.searchsorted(t2, t1 - limit)
    jj2 = np.searchsorted(t2, t1 + limit)

    # Concatenate the recentered spike times into a big array
    # We have excluded spikes outside of the histogram range to limit
    # memory use here.
    big = np.concatenate([t2[i:j] - t for t, i, j in zip(t1, ii2, jj2)])

    # Actually do the histogram. Note that calls to np.histogram are
    # expensive because it does not assume sorted data.
    count, bins = np.histogram(big, bins=bins, density=density)

    if auto:
        # Compensate for the peak at time zero that results in autocorrelations
        # by subtracting the total number of spikes from that bin. Note
        # possible numerical issue here because 0.0 may fall at a bin edge.
        c_temp, bins_temp = np.histogram([0.], bins=bins)
        bin_containing_zero = np.nonzero(c_temp)[0][0]
        count[bin_containing_zero] -= len(t1)

    # Finally compensate for the swapping of t1 and t2
    if swap_args:
        # Here we rely on being able to simply reverse `counts`. This is only
        # possible because of the way `bins` was defined (bins = -bins[::-1])
        count = count[::-1]

    return count, bins
