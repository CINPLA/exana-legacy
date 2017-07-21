import quantities as pq
import numpy as np
import neo
from exana.misc.tools import is_quantities


def baysian_latency(count_data):
    import pymc3 as pm
    import theano.tensor as tt
    n_count_data = len(count_data)
    with pm.Model() as model:
        alpha = 1.0/count_data.mean()  # Recall count_data is the
                                       # variable that holds our txt counts
        lambda_1 = pm.Exponential("lambda_1", alpha)
        lambda_2 = pm.Exponential("lambda_2", alpha)

        tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)

    with model:
        idx = np.arange(n_count_data) # Index
        lambda_ = pm.math.switch(tau >= idx, lambda_1, lambda_2)
        observation = pm.Poisson("obs", lambda_, observed=count_data)
        step = pm.Metropolis()
        trace = pm.sample(10000, tune=5000,step=step)
    return trace


def generate_salt_trials(spike_train, epoch):
    """
    Generate test and baseline trials from spike train and epoch for salt.

    Test trial are trials within epoch times and durations, baseline trails
    are between time + duration and next time.

    Note
    ----
    Spikes before the first trial are disregarded in aseline trials.

    Parameters
    ----------
    spike_train : neo.SpikeTrain
    epoch : neo.Epoch

    Returns
    -------
    out : tuple
        (baseline_trials, test_trials)
    """
    from exana.stimulus import make_spiketrain_trials
    e = epoch
    test_trials = make_spiketrain_trials(spike_train=spike_train,
                                         epoch=e)
    durations = np.array(
        [t2 - t1 - d for t1, t2, d in zip(e.times,
                                          e.times[1:],
                                          e.durations)]) * e.times.units
    times = np.array(
        [t1 + d for t1, d in zip(e.times[:-1], e.durations[:-1])]) * e.times.units
    baseline_epoch = neo.Epoch(times=times, durations=durations)
    baseline_trials = make_spiketrain_trials(spike_train=spike_train,
                                             epoch=baseline_epoch)
    return baseline_trials, test_trials


def salt(baseline_trials, test_trials, winsize=0.01*pq.s,
         latency_step=0.01*pq.s):
    '''SALT   Stimulus-associated spike latency test.
    Calculates a modified version of Jensen-Shannon divergence (see [1]_)
    for spike latency histograms. Please cite [2]_ when using this program.

    Parameters
    ----------
    baseline_trials : Spike raster for stimulus-free baseline
       period. The baseline period has to excede the window size (winsize)
       multiple times, as the length of the baseline segment divided by the
       window size determines the sample size of the null
       distribution (see below).
    test_trials : Spike raster for test period, i.e. after
       stimulus. The test period has to excede the window size (winsize)
       multiple times, as the length of the test period divided by the
       latency_step size determines the number of latencies to be tested.
    winsize : quantities.Quantity
        Window size for baseline and test windows in seconds
        (optional default, 0.01 s).
    latency_step : quantities.Quantity
        Step size for test latencies in seconds
        (optional default, 0.01 s).


    Returns
    -------
    latencies : list
        latencies tested
    p_values : list
        Resulting P values for the Stimulus-Associated spike Latency Test.
    I_values : list
        Test statistic, difference between within baseline and test-to-baseline
        information distance values.

    Notes
    -----
    Briefly, the baseline binned spike raster (baseline_trials) is cut to
    non-overlapping epochs (window size determined by WN) and spike latency
    histograms for first spikes are computed within each epoch. A similar
    histogram is constructed for the test epoch (test_trials). Pairwise
    information distance measures are calculated for the baseline
    histograms to form a null-hypothesis distribution of distances. The
    distances of the test histogram and all baseline histograms are
    calculated and the median of these values is tested against the
    null-hypothesis distribution, resulting in a p value (P).

    References
    ----------
    .. [1] res DM, Schindelin JE (2003) A new metric for probability
       distributions. IEEE Transactions on Information Theory 49:1858-1860.

    .. [2] Kvitsiani D*, Ranade S*, Hangya B, Taniguchi H, Huang JZ, Kepecs A
       (2013) Distinct behavioural and network correlates of two interneuron
       types in prefrontal cortex. Nature 498:363?6.'''
    t_start = baseline_trials[0].t_start.rescale('s')
    t_stop = baseline_trials[0].t_stop.rescale('s')
    winsize = winsize.rescale('s')
    baseline_trials = [trial.rescale('s') for trial in baseline_trials]
    test_trials = [trial.rescale('s') for trial in test_trials]

    windows = np.arange(t_start, t_stop + winsize, winsize)
    binsize = winsize / 20
    bins = np.arange(- binsize, winsize + binsize, binsize)
    # Latency histogram - baseline
    nbtrials = len(baseline_trials)  # number of trials and number of baseline (pre-stim) data points
    nbins = len(bins)   # number of bins for latency histograms
    nwins = len(windows)
    hlsi = np.zeros((nbins - 1, nwins))   # preallocate latency histograms
    nhlsi = np.zeros((nbins - 1, nwins))    # preallocate latency histograms
    for i in range(nwins - 1):   # loop through baseline windows
        min_spike_times = []
        for j, trial in enumerate(baseline_trials):   # loop through trials
            mask = (trial < windows[i + 1]) & (trial > windows[i])
            spikes_in_win = trial[mask]
            if len(spikes_in_win) > 0:
                min_spike_times.append(spikes_in_win.min().magnitude - windows[i])   # latency from window
            else:
                min_spike_times.append(- binsize / 2)   # 0 if no spike in the window
        hlsi[:, i], _ = np.histogram(min_spike_times, bins)   # latency histogram
        nhlsi[:, i] = hlsi[:, i] / sum(hlsi[:, i])   # normalized latency histogram

    test_t_stop = test_trials[0].t_stop.rescale('s')
    latencies = np.arange(0 * pq.s, test_t_stop + latency_step, latency_step)
    p_values = []
    I_values = []
    nttrials = len(test_trials)   # number of trials
    lsi_tt = np.zeros((nttrials,1))*np.nan   # preallocate latency matrix
    for latency in latencies:
        min_spike_times = []
        for j, trial in enumerate(test_trials):   # loop through trials
            mask = (trial < latency + winsize.magnitude) & (trial > latency)
            spikes_in_win = trial[mask]
            if len(spikes_in_win) > 0:
                min_spike_times.append(spikes_in_win.min().magnitude - latency)   # latency from window
            else:
                min_spike_times.append(- binsize / 2)   # 0 if no spike in the window
        hlsi[:, nwins - 1], _ = np.histogram(min_spike_times, bins)   # latency histogram
        nhlsi[:, nwins - 1] = hlsi[:, nwins - 1] / sum(hlsi[:, nwins - 1])   # normalized latency histogram
        # JS-divergence
        kn = nwins   # number of all windows (nwins baseline win. + 1 test win.)
        jsd = np.zeros((kn, kn)) * np.nan
        for k1 in range(kn):
            D1 = nhlsi[:, k1]  # 1st latency histogram
            for k2 in range(k1+1, kn):
                D2 = nhlsi[:, k2]   # 2nd latency histogram
                jsd[k1, k2] = np.sqrt(JSdiv(D1, D2) * 2)  # pairwise modified JS-divergence (real metric!)

        # Calculate p-value and information difference
        p, I = makep(jsd, kn)
        p_values.append(p)
        I_values.append(I)
    return latencies * pq.s, p_values, I_values


def makep(kld, kn):
    '''Calculates p value from distance matrix.'''

    pnhk = kld[:kn - 1, :kn - 1]
    nullhypkld = pnhk[np.isfinite(pnhk)]   # nullhypothesis
    testkld = np.median(kld[:kn - 1, kn - 1])  # value to test
    sno = len(nullhypkld)   # sample size for nullhyp. distribution
    p_value = sum(nullhypkld >= testkld) / sno
    Idiff = testkld - np.median(nullhypkld)   # information difference between baseline and test min_spike_times
    return p_value, Idiff


def JSdiv(P, Q):
    '''JSDIV   Jensen-Shannon divergence.
    Calculates the Jensen-Shannon divergence of the two
    input distributions.'''
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # JS-divergence
    M = (P + Q) / 2
    D1 = KLdist(P, M)
    D2 = KLdist(Q, M)
    D = (D1 + D2) / 2
    return D


def KLdist(P, Q):
    '''KLDIST   Kullbach-Leibler distance.
    Calculates the Kullbach-Leibler distance (information
    divergence) of the two input distributions.'''
    assert abs(sum(P)-1) < 0.00001 or abs(sum(Q)-1) < 0.00001,\
        'Input arguments must be probability distributions.'

    assert P.size == Q.size, 'Input distributions must be of the same size.'

    # KL-distance
    P2 = P[P * Q > 0]     # restrict to the common support
    Q2 = Q[P * Q > 0]
    P2 = P2 / sum(P2)  # renormalize
    Q2 = Q2 / sum(Q2)

    D = sum(P2 * np.log(P2 / Q2))
    return D
