import pytest
import elephant
import neo
import quantities as pq
import numpy as np


def test_baysian():
    from exana.stimulus import baysian_latency, generate_salt_trials
    from exana.misc import concatenate_spiketrains
    from elephant.spike_train_generation import homogeneous_poisson_process as hpp
    np.random.seed(12345)
    N_trials = 100
    stim_duration = 100 * pq.ms
    stim_start = 1000 * pq.ms
    stim_latency = 50 * pq.ms
    trial_duration = 1150 * pq.ms
    trains = []
    stim_onsets = []
    for n in range(N_trials):
        offset = trial_duration * n
        stim_onsets.append(offset)
        trains.extend([hpp(rate=2 * pq.Hz,
                          t_start=offset,
                          t_stop=stim_start + stim_latency + offset),
                      hpp(rate=8 * pq.Hz,
                          t_start=stim_start + stim_latency + offset,
                          t_stop=stim_start + stim_duration + offset)])
    spike_train = concatenate_spiketrains(trains)

    epoch = neo.Epoch(
        times=np.array(stim_onsets) * pq.ms,
        durations=np.array([trial_duration] * len(stim_onsets)) * pq.ms)


    from exana.stimulus import make_spiketrain_trials
    trials = make_spiketrain_trials(spike_train=spike_train, epoch=epoch)
    from elephant.statistics import time_histogram
    t_start = trials[0].t_start.rescale('s')
    t_stop = trials[0].t_stop.rescale('s')

    binsize = (abs(t_start)+abs(t_stop))/float(100)
    time_hist = time_histogram(trials, binsize, t_start=t_start,
                               t_stop=t_stop, output='counts', binary=False)
    bins = np.arange(t_start.magnitude, t_stop.magnitude, binsize.magnitude)

    count_data = time_hist.magnitude

    trace = baysian_latency(count_data)
    return count_data, trace

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')
    import matplotlib.pyplot as plt

    count_data, trace = test_baysian()

    plt.figure()
    n_count_data = len(count_data)
    plt.bar(np.arange(n_count_data), count_data)
    plt.xlim(0, n_count_data);
    plt.show()

    lambda_1_samples = trace['lambda_1']
    lambda_2_samples = trace['lambda_2']
    tau_samples = trace['tau']

    plt.figure(figsize=(12.5, 10))
    #histogram of the samples:

    ax = plt.subplot(311)
    ax.set_autoscaley_on(False)

    plt.hist(lambda_1_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_1$", color="#A60628", normed=True)
    plt.legend(loc="upper left")
    plt.title(r"""Posterior distributions of the variables
        $\lambda_1,\;\lambda_2,\;\tau$""")
    plt.xlim([15, 30])
    plt.xlabel("$\lambda_1$ value")

    ax = plt.subplot(312)
    ax.set_autoscaley_on(False)
    plt.hist(lambda_2_samples, histtype='stepfilled', bins=30, alpha=0.85,
             label="posterior of $\lambda_2$", color="#7A68A6", normed=True)
    plt.legend(loc="upper left")
    plt.xlim([15, 30])
    plt.xlabel("$\lambda_2$ value")

    plt.subplot(313)
    w = 1.0 / tau_samples.shape[0] * np.ones_like(tau_samples)
    plt.hist(tau_samples, bins=n_count_data, alpha=1,
             label=r"posterior of $\tau$",
             color="#467821", weights=w, rwidth=2.)
    plt.xticks(np.arange(n_count_data))

    plt.legend(loc="upper left")
    plt.ylim([0, .75])
    plt.xlim([35, len(count_data)-20])
    plt.xlabel(r"$\tau$ (in days)")
    plt.ylabel("probability")
    plt.show()
