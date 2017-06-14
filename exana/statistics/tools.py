import numpy as np
import quantities as pq
import neo


def theta_mod_idx(sptr, **kwargs):
    '''Theta modulation index as defined in [1]_

    References:
    -----------
    .. [1] Cacucci, F., Lever, C., Wills, T. J., Burgess, N., & O'Keefe, J. (2004).
       Theta-modulated place-by-direction cells in the hippocampal formation in the rat.
       The Journal of Neuroscience, 24(38), 8265-8277.
    '''
    par = {'corr_bin_width': 0.01*pq.s,
           'corr_limit': 1.*pq.s}
    if kwargs:
        par.update(kwargs)
    from .correlogram import correlogram
    bin_width = par['corr_bin_width'].rescale('s').magnitude
    limit = par['corr_limit'].rescale('s').magnitude
    count, bins = correlogram(t1=sptr.times.magnitude, t2=None,
                              bin_width=bin_width, limit=limit,  auto=True)
    th = count[(bins[:-1] >= .05) & (bins[:-1] <= .07)].mean()
    pk = count[(bins[:-1] >= .1) & (bins[:-1] <= .14)].mean()
    return (pk - th)/(pk + th)


def fano_factor(trials, bins=1, return_mean_var=False, return_bins=False):
    """
    Calculate binned fano factor over several trials.

    Parameters
    ----------
    trials : list
        a list with np.arrays or neo.Spiketrains of spike times
    bins : np.ndarray or int
        bins of where to calculate fano factor. Default is 1
    return_mean_var : bool
        return mean count rate of trials and variance

    Returns
    -------
    out : float, or optional tuple
        fano factor, or optional (mean, var) if return_mean_var, or optional
        (fano factor, bins).

    Note
    ----
    This is a similar method as in [1]_, however there a sliding window was
    used.

    To do
    -----
    Sliding window calculation of the Fano factor
    window = 50 * pq.ms
    step_size = 10 * pq.ms
    t_stop = 1 * pq.s
    bins = []; i = 0
    while i * step_size + window <= t_stop:
        bins.extend([i * step_size, i * step_size + window])
        i += 1

    Examples
    --------
    >>> t1 = np.arange(0, .5, .1)
    >>> t2 = np.arange(0.1, .6, .1)
    >>> fano_factor([t1, t2], bins=3)
    array([ 0.,  0.,  0.])

    If you want to further work with the means and vars
    >>> fano_factor([t1, t2], bins=3, return_mean_var=True)
    (array([ 2.,  1.,  2.]), array([ 0.,  0.,  0.]))

    The fano factor is 1 for Poisson processes
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> np.random.seed(12345)
    >>> t1 = [homogeneous_poisson_process(
    ...     10 * pq.Hz, t_start=0.0 * pq.s, t_stop=1 * pq.s) for _ in range(100)]
    >>> fano_factor(t1)
    array([ 0.95394394])

    The Fano factor computed in bins along time can be acheived with including
    `bins` which can be `int`
    >>> ff, bins = fano_factor(t1, bins=4, return_bins=True)
    >>> ff
    array([ 0.78505226,  1.16330097,  1.00901961,  0.80781457])

    >>> bins
    array([ 0.06424358,  0.29186518,  0.51948679,  0.74710839,  0.97472999])

    To specify bins
    >>> bins = np.arange(0, 1, .2)
    >>> fano_factor(t1, bins=bins)
    array([ 0.95941748,  1.09      ,  1.05650485,  0.72886256])

    References
    ----------
    .. [1] Churchland, M. M., Byron, M. Y., Cunningham, J. P., Sugrue, L. P.,
       Cohen, M. R., Corrado, G. S., ... & Bradley, D. C. (2010). Stimulus onset
       quenches neural variability: a widespread cortical phenomenon. Nature
       neuroscience, 13(3), 369-378.
    """
    assert len(trials) > 0, 'trials cannot be empty'
    if isinstance(trials[0], neo.SpikeTrain):
        trials = [trial.times for trial in trials]
    if isinstance(bins, int):
        nbins = bins
    else:
        nbins = len(bins) - 1
    hists = np.zeros((len(trials), nbins))
    for trial_num, trial in enumerate(trials):
        hist, _bins = np.histogram(trial, bins)
        hists[trial_num, :] = hist
    if len(trials) == 1:  # calculate fano over one trial
        axis = 1  # cols
    else:
        axis = 0  # rows
    mean = np.mean(hists, axis=axis)
    var = np.var(hists, axis=axis)
    if return_mean_var:
        if return_bins:
            return mean, var, bins
        else:
            return mean, var

    else:
        fano = var / mean
        if return_bins:
            return fano, _bins
        else:
            return fano


def fano_factor_multiunit(unit_trials, bins=1, return_rates=False,
                          return_bins=False):
    '''
    Calculate fano factor over several units with several trials as slopes from
    linear regression relating the variance to the mean of spike counts; see
    [1]_.

    Parameters
    ----------
    unit_trials : list of lists with trials
        That is unit_trials[0] = first unit, unit_trials[0][0] = first trial of
        first unit.
    bins : np.ndarray or int
        bins of where to calculate fano factor. Default is 1

    Returns
    -------
    (slopes, std_errors) : tuple
        Fano factor for each bin with corresponding standard error of the mean.

    To do
    -----
    Weighted regression (binsize/1000) and distribution matching as in [1]_.

    See also
    --------
    :func: `exana.statistics.fano_factor` : The function that calcuates mean
        and var.
    :func: `scipy.statistics.linregress` : The function that calcuates slopes
        and standard error.

    Note
    ----
    You need many neurons to get a decent output value as you only have one
    datapoint per neuron. If you have some neurons with many trials consider
    doing a weighted regression.

    To get 95 %% confidence interval you may use the standard error of teh mean
    by (fano - 2 * std_err, fano + 2 * std_err)

    Examples
    --------
    The fano factor is 1 for Poisson processes, thus we genereate 100 Poisson
    spiking neurons with each 10 trials. Not
    >>> from elephant.spike_train_generation import homogeneous_poisson_process
    >>> np.random.seed(12345)
    >>> units_trials = [
    ...     [homogeneous_poisson_process(
    ...          10 * pq.Hz, t_start=0.0 * pq.s, t_stop=1 * pq.s)
    ...      for _ in range(10)] for _ in range(100)]
    >>> fano_factor_multiunit(units_trials)
    ([0.91849103672074672], [0.041403689431354126])

    References
    ----------
    .. [1] Churchland, M. M., Byron, M. Y., Cunningham, J. P., Sugrue, L. P.,
       Cohen, M. R., Corrado, G. S., ... & Bradley, D. C. (2010). Stimulus onset
       quenches neural variability: a widespread cortical phenomenon. Nature
       neuroscience, 13(3), 369-378.
    '''
    from scipy.stats import linregress
    dim = 's'
    if isinstance(bins, int):
        nbins = bins
    else:
        nbins = len(bins) - 1
    nunits = len(unit_trials)
    means = np.zeros((nunits, nbins))
    varis = np.zeros((nunits, nbins))
    for unit_num, trials in enumerate(unit_trials):
        if len(trials) == 0:
            continue
        mean, var, bins = fano_factor(trials, bins, return_mean_var=True,
                                      return_bins=True)
        means[unit_num, :] = mean
        varis[unit_num, :] = var
    fanos = []
    std_errs = []
    for nb in range(nbins):
        slope, intercept, r_value, p_value, std_err = linregress(means[:, nb],
                                                                 varis[:, nb])
        std_errs.append(std_err / np.sqrt(nunits))
        fanos.append(slope)
    if return_rates:
        rates = np.mean(means, axis=0) / (bins[1] - bins[0])
        if return_bins:
            return fanos, std_errs, rates, bins
        else:
            return fanos, std_errs, rates
    else:
        if return_bins:
            return fanos, std_errs, bins
        else:
            return fanos, std_errs


def coeff_var(trials):
    """
    Calculate the coefficient of variation in inter spike interval (ISI)
    distribution over several trials

    Parameters
    ----------
    trials : list of neo.SpikeTrain or array like

    Returns
    -------
    out : list
        Coefficient of variations for each trial, nan if len(trial) == 0

    Examples
    --------
    >>> np.random.seed(12345)
    >>> trials = [np.arange(10), np.random.random((10))]
    >>> coeff_var(trials)
    [0.0, -9.533642434602724]

    """
    cvs = []
    for trial in trials:
        isi = np.diff(trial)
        if len(isi) > 0:
            cvs.append(np.std(isi) / np.mean(isi))
        else:
            cvs.append(np.nan)
    return cvs


def bootstrap(data, num_samples=10000, statistic=np.mean, alpha=0.05):
    """
    Returns bootstrap estimate of 100.0*(1-alpha) CI for statistic.
    Adapted from http://people.duke.edu/~ccc14/pcfb/analysis.html
    """
    import numpy.random as npr
    n = len(data)
    idx = npr.randint(0, n, (num_samples, n))
    samples = data[idx]
    stat = np.sort(statistic(samples, 1))
    return (stat[int((alpha / 2.0) * num_samples)],
            stat[int((1 - alpha / 2.0) * num_samples)])


def stat_test(tdict, test_func=None, nan_rule='remove'):
    '''performes statistic test between groups in tdict by given test function
    (test_func)'''
    import pandas as pd
    if test_func is None:
        from scipy import stats
        test_func = lambda g1, g2: stats.ttest_ind(g1, g2, equal_var=False)
    ps = {}
    sts ={}
    lib = []
    for key1, item1 in tdict.iteritems():
        for key2, item2 in tdict.iteritems():
            if key1 != key2:
                if set([key1, key2]) in lib:
                    continue
                lib.append(set([key1, key2]))
                one = np.array(item1, dtype=np.float64)
                two = np.array(item2, dtype=np.float64)
                if nan_rule == 'remove':
                    one = one[np.isfinite(one)]
                    two = two[np.isfinite(two)]
                assert len(one) > 0, 'Empty list of values'
                assert len(two) > 0, 'Empty list of values'
                stat, p = test_func(one, two)
                ps[key1+'--'+key2] = p
                sts[key1+'--'+key2] = stat
    return pd.DataFrame([ps, sts], index=['p-value','statistic'])


def pairwise_corrcoef(nrns, binsize=5*pq.ms):
    from elephant.conversion import BinnedSpikeTrain
    from elephant.spike_train_correlation import corrcoef
    cc = []
    for id1, st1 in enumerate(nrns):
        for id2, st2 in enumerate(nrns):
            if id1 != id2:
                cc_matrix = corrcoef(BinnedSpikeTrain([st1, st2],
                                                      binsize=binsize))
                cc.append(cc_matrix[0, 1])
    return cc
