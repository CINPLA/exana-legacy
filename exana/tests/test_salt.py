import pytest
import elephant
import neo
import quantities as pq
import numpy as np


def _test_salt_inh():
    from exana.stimulus import salt, generate_salt_trials
    from exana.misc import concatenate_spiketrains
    from elephant.spike_train_generation import homogeneous_poisson_process as hpp
    np.random.seed(12345)
    N_trials = 100
    stim_duration = 100 * pq.ms
    stim_start = 1000 * pq.ms
    stim_latency = 50 * pq.ms
    trial_duration = 1500 * pq.ms
    trains = []
    stim_onsets = []
    for n in range(N_trials):
        offset = trial_duration * n
        stim_onsets.append(stim_start + offset)
        trains.extend([hpp(rate=8 * pq.Hz,
                          t_start=offset,
                          t_stop=stim_start + stim_latency + offset),
                      hpp(rate=0 * pq.Hz,
                          t_start=stim_start + stim_latency + offset,
                          t_stop=stim_start + stim_duration + offset),
                      hpp(rate=8 * pq.Hz,
                          t_start=stim_start + stim_duration + offset,
                          t_stop=trial_duration + offset)])
    spike_train = concatenate_spiketrains(trains)

    epoch = neo.Epoch(
        times=np.array(stim_onsets) * pq.ms,
        durations=np.array([stim_duration] * len(stim_onsets)) * pq.ms)
    baseline_trials, test_trials = generate_salt_trials(spike_train, epoch)


    latencies, p_values, I_values = salt(baseline_trials=baseline_trials,
                                         test_trials=test_trials,
                                         winsize=0.02*pq.s)
    print(latencies)
    print(p_values)
    idxs, = np.where(np.array(p_values) < 0.01)
    # assert latencies[min(idxs)] == stim_latency
    return baseline_trials, test_trials, spike_train, epoch


def test_salt_exc():
    from exana.stimulus import salt, generate_salt_trials
    from exana.misc import concatenate_spiketrains
    from elephant.spike_train_generation import homogeneous_poisson_process as hpp
    np.random.seed(12345)
    N_trials = 100
    stim_duration = 100 * pq.ms
    stim_start = 1000 * pq.ms
    stim_latency = 50 * pq.ms
    trial_duration = 1500 * pq.ms
    trains = []
    stim_onsets = []
    for n in range(N_trials):
        offset = trial_duration * n
        stim_onsets.append(stim_start + offset)
        trains.extend([hpp(rate=2 * pq.Hz,
                          t_start=offset,
                          t_stop=stim_start + stim_latency + offset),
                      hpp(rate=8 * pq.Hz,
                          t_start=stim_start + stim_latency + offset,
                          t_stop=stim_start + stim_duration + offset),
                      hpp(rate=2 * pq.Hz,
                          t_start=stim_start + stim_duration + offset,
                          t_stop=trial_duration + offset)])
    spike_train = concatenate_spiketrains(trains)

    epoch = neo.Epoch(
        times=np.array(stim_onsets) * pq.ms,
        durations=np.array([stim_duration] * len(stim_onsets)) * pq.ms)
    baseline_trials, test_trials = generate_salt_trials(spike_train, epoch)


    latencies, p_values, I_values = salt(baseline_trials=baseline_trials,
                                         test_trials=test_trials,
                                         winsize=0.01*pq.s,
                                         latency_step=0.01*pq.s)
    idxs, = np.where(np.array(p_values) < 0.01)
    print(latencies)
    print(p_values)
    assert latencies[min(idxs)] == stim_latency
    return baseline_trials, test_trials, spike_train, epoch


if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')

    # baseline_trials, test_trials, spike_train, epoch = _test_salt_inh()
    baseline_trials, test_trials, spike_train, epoch = test_salt_exc()

    binsize = 1 * pq.ms
    import matplotlib.pyplot as plt
    from exana.stimulus import plot_psth
    plot_psth(trials=test_trials, title='test', binsize=10*pq.ms)
    plot_psth(trials=baseline_trials, title='baseline', binsize=10*pq.ms)
    plot_psth(sptr=spike_train, epoch=epoch, t_start=-1 * pq.s,
              t_stop=0.5 * pq.s, title='full', binsize=10*pq.ms)
    plt.show()
    # NOTE for saving matlab var and test vs original matlab script
    from elephant.conversion import BinnedSpikeTrain
    test_binary = BinnedSpikeTrain(test_trials, binsize=binsize).to_array()
    baseline_binary = BinnedSpikeTrain(baseline_trials, binsize=binsize).to_array()

    import scipy
    scipy.io.savemat('/home/mikkel/apps/salt_data.mat',
                     {'spt_baseline': baseline_binary,
                      'spt_test': test_binary})
