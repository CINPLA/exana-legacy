import neo
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
import os


def concatenate_spiketrains(spike_trains):
    '''
    Concatenate multiple spiketrains

    Parameters
    ----------
    spike_trains : list, tuple
        A list containing neo.SpikeTrain to be concatenated.

    Returns
    -------
    neo.SpikeTrain
        A new spike train concatenated.

    Example
    -------
    >>> spiketrain1 = neo.SpikeTrain(times=np.arange(5), t_stop=10, units='s')
    >>> spiketrain2 = neo.SpikeTrain(times=np.arange(5, 10), t_stop=20,
    ...                              units='s')
    >>> spiketrain = concatenate_spiketrains([spiketrain1, spiketrain2])
    >>> spiketrain.times
    array([0., 1., 2., 3., 4., 5., 6., 7., 8., 9.]) * s
    '''
    if not isinstance(spike_trains, (list, tuple)):
        raise TypeError('Expected "list, tuple", got "' + str(type(spike_trains)) + '"')
    if not all(isinstance(sptr, neo.SpikeTrain) for sptr in spike_trains):
        raise TypeError('Expected "list" containing "neo.SpikeTrain", got "' +
                        str([type(sptr) for sptr in spike_trains]) + '"')
    if len(spike_trains) == 1:
        return spike_trains[0]
    elif len(spike_trains) == 0:
        raise ValueError('Recieved an empty list')
    ts = spike_trains[0].times.magnitude
    ts_units = spike_trains[0].times.units
    wf = spike_trains[0].waveforms
    if wf is not None:
        wf_units = wf.units
        wf = wf.magnitude
    t_stop = [spike_trains[0].t_stop]
    t_start = [spike_trains[0].t_start]
    sampling_rate = spike_trains[0].sampling_rate
    left_sweep = spike_trains[0].left_sweep
    for sptr in spike_trains[1:]:
        if not ts_units == sptr.times.units:
            raise ValueError('Times have different physical units.')
        ts = np.concatenate((ts, sptr.times.magnitude))
        if wf is not None:
            if not wf_units == sptr.waveforms.units:
                raise ValueError('Waveforms have different physical units.')
            wf = np.concatenate((wf, sptr.waveforms.magnitude))
        t_stop.append(sptr.t_stop)
        t_start.append(sptr.t_start)
        assert sampling_rate == sptr.sampling_rate
        assert left_sweep == sptr.left_sweep
    idxs = np.argsort(ts)
    ts = ts[idxs] * ts_units
    if wf is not None:
        wf = wf[idxs] * wf_units
    if not all(t1 == t2 for t1 in t_stop for t2 in t_stop):
        import warnings
        warnings.warn('Spiketrains have different stop times, we choose the' +
                      ' largest')
    if not all(t1 == t2 for t1 in t_start for t2 in t_start):
        import warnings
        warnings.warn('Spiketrains have different start times, we choose the' +
                      ' smallest')
    return neo.SpikeTrain(times=ts,
                          waveforms=wf,
                          t_stop=max(t_stop),
                          t_start=min(t_start),
                          left_sweep=sptr.left_sweep,
                          sampling_rate=sptr.sampling_rate)


def moving_average(vector, N):
    """
    Circular moving average (box car) using convolution with square function of
    unity. By circular we connect the start with the end before averaging.

    Parameters
    ----------
    vector : np.array
        Vector of desired averaging.
    N : int
        Length of square function (box car).

    Returns
    -------
    np.array

    Note
    ----
    Equal, but faster than:
    vector = np.concatenate((vector[-N:], vector, vector[:N]))
    return [sum(vector[i:i + N]) / N for i in range(N - 2, len(vector) - N - 2)]

    Examples
    -------
    >>> a = np.ones((10, ))
    >>> moving_average(a, 5)
    array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1.])

    >>> a = np.concatenate((np.arange(5), np.arange(5)[::-1]))
    >>> print(a)
    [0 1 2 3 4 4 3 2 1 0]
    >>> moving_average(a, 5)
    array([0.8, 1.2, 2. , 2.8, 3.2, 3.2, 2.8, 2. , 1.2, 0.8])

    >>> a = np.arange(10)
    >>> moving_average(a, 5)
    array([4., 3., 2., 3., 4., 5., 6., 7., 6., 5.])
    """
    if N * 2 > len(vector):
        raise ValueError('Window must be at least half of "len(vector)"')
    vector = np.concatenate((vector[-N:], vector, vector[:N]))
    return np.convolve(vector, np.ones((N,)) / N, mode='same')[N:-N]


def nested_dict2pandas_df(dictionary, depth=3, fcn=None):
    import pandas as pd
    if fcn is None:
        def fcn(inp):
            return pd.Series(inp)
    if depth == 3:
        reform = {(outerKey, midKey, innerKey): fcn(values)
                  for outerKey, midDict in dictionary.iteritems()
                  for midKey, innerDict in midDict.iteritems()
                  for innerKey, values in innerDict.iteritems()}
    elif depth == 2:
        reform = {(outerKey, innerKey): fcn(values)
                  for outerKey, innerDict in dictionary.iteritems()
                  for innerKey, values in innerDict.iteritems()}
    else:
        raise NotImplementedError
    return pd.DataFrame(reform)


def is_quantities(data, dtype='scalar'):
    """
    Test if data is of isinstance Quantity and if it is scalar or 1D vector.

    Parameters
    ----------
    data : list
        data to test
    dtype : str
        Default = "scalar", or "vector"
    """
    if not isinstance(data, list):
        data = [data]
    for d in data:
        if dtype == 'scalar':
            try:
                assert isinstance(d.units, pq.Quantity)
                assert d.shape in ((), (1, ))
            except:
                raise ValueError('data must be a scalar quantities value')
        if dtype == 'vector':
            try:
                assert isinstance(d.units, pq.Quantity)
                assert len(d.shape) == 1
            except:
                raise ValueError('data must be a 1d quantities.Quantity array')


def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)

    Obtained from http://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
    """
    from scipy.ndimage.filters import maximum_filter
    from scipy.ndimage.morphology import (generate_binary_structure,
                                          binary_erosion)
    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask

    # we create the mask of the background
    background = (image == 0)

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood,
                                       border_value=1)

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask
    detected_peaks = local_max - eroded_background

    return detected_peaks


def normalize(x, mode='minmax'):
    '''
    Normalizes x ignoring nan with given mode

    Parameters
    ----------
    x : np.ndarray
    mode : str
        Default='minmax' or 'zscore'

    Returns
    -------
    minmax : x in [0,1]
    zscore : mean(x) = 0, std(x) in [0,1]

    Example
    -------
    >>> a = np.arange(3, 10)
    >>> normalize(a)
    array([0.        , 0.16666667, 0.33333333, 0.5       , 0.66666667,
           0.83333333, 1.        ])

    >>> a = np.arange(3, 10)
    >>> normalize(a, mode='zscore')
    array([-1.5, -1. , -0.5,  0. ,  0.5,  1. ,  1.5])
    '''
    x = np.array(x)
    if mode == 'minmax':
        xp = (x - np.nanmin(x))
        x = xp / np.nanmax(xp)
    elif mode == 'zscore':
        x = (x - np.nanmean(x)) / np.nanstd(x)
    return x


def find_max_peak(vector):
    """
    Find the maximum peak of a 1D vector, using scipy argrelmax function.

    Parameters
    ----------
    vector : np.array
        1D array

    Returns
    -------
    peak, index : float, np.array
        The maximum peak and its corresponding index in the input vector.

    Examples
    --------
    # no peaks returns nan
    >>> a = np.concatenate((np.arange(5), np.arange(5)[::-1]))
    >>> find_max_peak(a)
    (nan, nan)

    >>> a = np.concatenate((np.arange(5), np.arange(6)[::-1]))
    >>> print(a)
    [0 1 2 3 4 5 4 3 2 1 0]
    >>> find_max_peak(a)
    (5, array([5]))

    >>> a = np.concatenate((np.arange(5), np.arange(6)[::-1], [5, 0]))
    >>> print(a)
    [0 1 2 3 4 5 4 3 2 1 0 5 0]
    >>> find_max_peak(a)
    (5, array([ 5, 11]))
    """
    from scipy import signal
    assert np.ndim(vector) == 1
    pksind, = signal.argrelmax(vector)
    if len(pksind) == 0:
        return np.nan, np.nan
    pk = vector[pksind].max()
    inds, = np.where(vector == pk)
    return pk, inds


def find_first_peak(vector):
    """
    Find the first peak of a 1D vector, using scipy argrelmax function.

    Parameters
    ----------
    vector : np.array
        1D array

    Returns
    -------
    peak, index : float, np.array
        The maximum peak and its corresponding index in the input vector.

    Examples
    --------
    # no peaks returns nan
    >>> a = np.concatenate((np.arange(5), np.arange(5)[::-1]))
    >>> find_first_peak(a)
    (nan, nan)

    >>> a = np.concatenate((np.arange(5), np.arange(6)[::-1], [5, 0]))
    >>> print(a)
    [0 1 2 3 4 5 4 3 2 1 0 5 0]
    >>> find_first_peak(a)
    (5, 5)
    """
    from scipy import signal
    assert np.ndim(vector) == 1
    pksind, = signal.argrelmax(vector)
    if len(pksind) == 0:
        return np.nan, np.nan
    ind = pksind.min()
    pk = vector[ind]
    return pk, ind


def masked_corrcoef2d(arr1, arr2):
    """
    Correlation coefficient of two 2 dimensional masked arrays.

    Parameters
    ----------
    arr1 : np.array
        2D array.
    arr2 : np.array
        2D array.

    See also
    --------
    numpy.corrcoef : NumPy corrcoef function.
    numpy.ma : NumPy mask module.

    Returns
    -------
    corr : np.array
        correlation coefficient from np.corrcoef.

    Example
    --------
    >>> import numpy.ma as ma
    >>> a = np.reshape(np.arange(10), (2,5))
    >>> v = np.reshape(np.arange(10), (2,5))
    >>> mask = np.zeros((2, 5), dtype=bool)
    >>> mask[1:, 3:] = True
    >>> v = ma.masked_array(v, mask=mask)
    >>> print(v)
    [[0 1 2 3 4]
     [5 6 7 -- --]]
    >>> masked_corrcoef2d(a, v)
    masked_array(
      data=[[1.0, 1.0],
            [1.0, 1.0]],
      mask=[[False, False],
            [False, False]],
      fill_value=1e+20)
            """
    import numpy.ma as ma
    a_ = np.reshape(arr1, (1, arr1.size))
    v_ = np.reshape(arr2, (1, arr2.size))
    corr = ma.corrcoef(a_, v_)
    return corr


def corrcoef2d(arr1, arr2):
    """
    Correlation coefficient of two 2 dimensional arrays.

    Parameters
    ----------
    arr1 : np.array
        2D array.
    arr2 : np.array
        2D array.

    See also
    --------
    numpy.corrcoef : NumPy corrcoef function.

    Returns
    -------
    corr : np.array
        correlation coefficient from np.corrcoef.

    Example
    --------
    >>> a = np.reshape(np.arange(10), (2,5))
    >>> v = np.reshape(np.arange(10), (2,5))
    >>> corrcoef2d(a, v)
    array([[1., 1.],
           [1., 1.]])
    """
    a_ = np.reshape(arr1, (1, arr1.size))
    v_ = np.reshape(arr2, (1, arr2.size))
    corr = np.corrcoef(a_, v_)
    return corr


def fftcorrelate2d(arr1, arr2, mode='full', normalize=False):
    """
    Cross correlation of two 2 dimensional arrays using fftconvolve from scipy.

    Parameters
    ----------
    arr1 : np.array
        2D array
    arr2 : np.array
        2D array
    mode : str
        Sent directly to numpe.fftconvolve
    normalize : bool
        Normalize arrays before convolution or not. Default is False.

    See also
    --------
    scipy.signal.fftconvolve : SciPy convolve function using fft.

    Returns
    -------
    corr : np.array
        Cross correlation

    Example
    --------
    >>> a = np.reshape(np.arange(4), (2,2))
    >>> acorr = fftcorrelate2d(a, a)
    """
    from scipy.signal import fftconvolve
    if normalize:
        a_ = np.reshape(arr1, (1, arr1.size))
        v_ = np.reshape(arr2, (1, arr2.size))
        arr1 = (arr1 - np.mean(a_)) / (np.std(a_) * len(a_))
        arr2 = (arr2 - np.mean(v_)) / np.std(v_)
    corr = fftconvolve(arr1, np.fliplr(np.flipud(arr2)), mode=mode)
    return corr
