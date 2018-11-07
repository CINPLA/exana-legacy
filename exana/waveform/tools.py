import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans, vq


def calculate_waveform_features(sptrs, calc_all_spikes=False):
    """Calculates waveform features for spiketrains; full-width half-maximum
    (half width) and minimum-to-maximum peak width (peak-to-peak width) for
    mean spike, and average firing rate. If calc_all_spikes is True it also
    calculates half width and peak-to-peak width for all the single spikes in
    each of the spiketrains.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains
    calc_all_spikes : bool
        returns half_width_all_spikes and peak_to_peak_all_spikes if True

    Returns
    ----------
    half_width_mean : array of floats
        full-width half-maximum (in ms) for mean spike of each spiketrain
    peak_to_peak_mean : array of floats
        minimum-to-maximum peak width (in ms) for mean spike of each spiketrain
    average_firing_rate : list of floats
        average firing rate (in Hz) for each spiketrain
    half_width_all_spikes : list of arrays with floats
        if calc_all_spikes is True, full-width half-maximum (in ms) for every
        single spike in each spiketrain
    peak_to_peak_all_spikes : list of arrays with floats
        if calc_all_spikes is True, minimum-to-maximum peak width (in ms) for
        every single spike in each spiketrain
    """
    for sptr in sptrs:
        if not hasattr(sptr.waveforms, 'shape'):
            raise AttributeError('Argument provided (sptr) has no attribute\
                                  waveforms.shape')

    average_firing_rate = calculate_average_firing_rate(sptrs)

    stime = np.arange(sptrs[0].waveforms.shape[2], dtype=np.float32) /\
        sptrs[0].sampling_rate
    stime.units = 'ms'

    mean_spikes_list = []
    all_spikes_list = []
    for i in range(len(sptrs)):
        mean_wf = np.mean(sptrs[i].waveforms, axis=0)
        max_amplitude_channel = np.argmin(mean_wf.min(axis=1))
        wf = mean_wf[max_amplitude_channel, :]
        mean_spikes_list.append(wf)
        if calc_all_spikes is True:
            wf2 = sptrs[i].waveforms[:, max_amplitude_channel, :]
            all_spikes_list.append(wf2)
    mean_spikes_list2 = [np.array(mean_spikes_list)]
    hw_list, ptp_list = calculate_spike_widths(mean_spikes_list2, stime)
    half_width_mean = hw_list[0]
    peak_to_peak_mean = ptp_list[0]

    if calc_all_spikes is True:
        half_width_all_spikes, peak_to_peak_all_spikes = \
            calculate_spike_widths(all_spikes_list, stime)
        return half_width_mean, peak_to_peak_mean, average_firing_rate,\
            half_width_all_spikes, peak_to_peak_all_spikes
    else:
        return half_width_mean, peak_to_peak_mean, average_firing_rate


def calculate_spike_widths(spikes_list, stime):
    """Calculates full-width half-maximum (half width) and minimum-to-maximum
    peak width (peak-to-peak width) for spikes.

    Parameters
    ----------
    spikes_list : list of arrays
        list of arrays with spike waveforms
    stime : array
        array of times for when the spikes waveform is measured

    Returns
    ----------
    half_width_list : list of arrays
        full-width half-maximum (in ms)
    peak_to_peak_list : list of arrays
        minimum-to-maximum peak width (in ms)
    """
    half_width_list = []
    peak_to_peak_list = []
    for wf in spikes_list:
        half_width = []
        peak_to_peak = []
        index_max_amplitude = np.argmin(wf, axis=1)
        value_half_amplitude = (wf.min(axis=1) * 0.5)[:, np.newaxis]
        new_wf = np.abs(wf - value_half_amplitude)
        for s in range(len(wf)):
            index_min = index_max_amplitude[s]
            if index_min == 0:
                pass
            else:
                p1 = np.argmin(new_wf[s, :index_min])
                p2 = index_min + np.argmin(new_wf[s, index_min:])
                half_width.append(stime[p2] - stime[p1])
                index_max = index_min + np.argmax(wf[s, index_min:])
                peak_to_peak.append(stime[index_max] - stime[index_min])
        half_width_list.append(np.array(half_width))
        peak_to_peak_list.append(np.array(peak_to_peak))
    return half_width_list, peak_to_peak_list


def calculate_average_firing_rate(sptrs):
    """Calculates average firing rate for spiketrains.

    Parameters
    ----------
    sptrs : list
        a list of neo spiketrains

    Returns
    ----------
    average_firing_rate : list of floats
        average firing rate (in Hz)
    """
    average_firing_rate = []
    for sptr in sptrs:
        nr_spikes = sptr.waveforms.shape[0]
        dt = sptr.t_stop - sptr.t_start
        average_firing_rate.append(nr_spikes/dt)
    return average_firing_rate


def cluster_waveform_features(feature1, feature2, nr_clusters=2):
    """Divides the spiketrains into groups using the k-means algorithm on the
    average waveform of each spiketrain. The average waveform is calculated
    based on feature1 and feature2 of mean spike to the spiketrain.

    Parameters
    ----------
    feature1 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    feature2 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    nr_clusters : int
        number of clusters you want, minimum 2, maximum 5

    Returns
    -------
    idx : list of integers
        for example when nr_clusters is 2 containts 0s and 1s
    red_group_index_list : list
        list of indexes (to features) belonging to red group
    blue_group_index_list : list
        list of indexes (to features) belonging to blue group
    green_group_index_list : list
        list of indexes belonging to green group, nr_clusters >= 3
    purple_group_index_list : list
        list of indexes belonging to purple group, nr_clusters >= 4
    black_group_index_list : list
        list of indexes belonging to black group, nr_clusters = 5
    """
    if nr_clusters < 2 or nr_clusters > 5:
        raise ValueError('Number of clusters must be from 2-5')

    features = np.stack(np.array([feature1, feature2]), axis=-1)
    centroids, _ = kmeans(features, nr_clusters)
    idx, _ = vq(features, centroids)

    red_group_index_list = []
    blue_group_index_list = []
    if nr_clusters == 3:
        green_group_index_list = []
    if nr_clusters == 4:
        green_group_index_list = []
        purple_group_index_list = []
    if nr_clusters == 5:
        green_group_index_list = []
        purple_group_index_list = []
        black_group_index_list = []

    for i in range(len(feature1)):
        if idx[i] == 0:
            red_group_index_list.append(i)
        if idx[i] == 1:
            blue_group_index_list.append(i)
        if nr_clusters == 3:
            if idx[i] == 2:
                green_group_index_list.append(i)
        if nr_clusters == 4:
            if idx[i] == 2:
                green_group_index_list.append(i)
            if idx[i] == 3:
                purple_group_index_list.append(i)
        if nr_clusters == 5:
            if idx[i] == 2:
                green_group_index_list.append(i)
            if idx[i] == 3:
                purple_group_index_list.append(i)
            if idx[i] == 4:
                black_group_index_list.append(i)

    if nr_clusters == 2:
        return idx, red_group_index_list, blue_group_index_list
    if nr_clusters == 3:
        return idx, red_group_index_list, blue_group_index_list,\
            green_group_index_list
    if nr_clusters == 4:
        return idx, red_group_index_list, blue_group_index_list,\
            green_group_index_list, purple_group_index_list
    if nr_clusters == 5:
        return idx, red_group_index_list, blue_group_index_list,\
            green_group_index_list, purple_group_index_list,\
            black_group_index_list


def plot_cluster_mean_spike(idx, feature1, feature2):
    """Plots the average waveform of each spiketrains mean spike in a cluster
    plot based on feature1 and feature2. The sptrs are previously separated in
    groups, from 2 to 5 (depending on given nr_clusters), with the function
    cluster_waveform_features().

    Parameters
    ----------
    idx : list of integers
        for example when nr_clusters is 2 containts 0s and 1s
    feature1 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    feature2 : array of floats
        array of floats describing a feature of the spiketrains mean spike
    """
    features = np.stack(np.array([feature1, feature2]), axis=-1)
    idx_max = np.max(idx)
    if idx_max == 1:
        plt.plot(features[idx == 0, 0], features[idx == 0, 1], 'Dr',
                 features[idx == 1, 0], features[idx == 1, 1], 'Db')
        plt.legend(["Red group", "Blue group"])
    if idx_max == 2:
        plt.plot(features[idx == 0, 0], features[idx == 0, 1], 'Dr',
                 features[idx == 1, 0], features[idx == 1, 1], 'Db',
                 features[idx == 2, 0], features[idx == 2, 1], 'Dg')
        plt.legend(["Red group", "Blue group", "Green group"])
    if idx_max == 3:
        plt.plot(features[idx == 0, 0], features[idx == 0, 1], 'Dr',
                 features[idx == 1, 0], features[idx == 1, 1], 'Db',
                 features[idx == 2, 0], features[idx == 2, 1], 'Dg',
                 features[idx == 3, 0], features[idx == 3, 1], 'Dm')
        plt.legend(["Red group", "Blue group", "Green group", "Purple group"])
    if idx_max == 4:
        plt.plot(features[idx == 0, 0], features[idx == 0, 1], 'Dr',
                 features[idx == 1, 0], features[idx == 1, 1], 'Db',
                 features[idx == 2, 0], features[idx == 2, 1], 'Dg',
                 features[idx == 3, 0], features[idx == 3, 1], 'Dm',
                 features[idx == 4, 0], features[idx == 4, 1], 'Dk')
        plt.legend(["Red group", "Blue group", "Green group", "Purple group",
                    "Black group"])
    plt.xlabel('feature 1')
    plt.ylabel('feature 2')
    plt.show()


def plot_histogram_waveform_feature(idx, feature, allspikes=False):
    """Plots the feature of the spiketrains in a histogram. The sptrs are
    previously separated in two groups, red and blue, by the function
    cluster_waveform_features(). Default is plotting a feature of the mean
    spike of the spiketrain. Set 'allspikes=True' if you want to plot a
    histogram over the feature for all the spikes from each spiketrain.

    Parameters
    ----------
    idx : list of integers
        for example when nr_clusters is 2 containts 0s and 1s
    feature : array of floats (single spikes) OR list of arrays (all spikes)
        (list of) array(s) of floats describing a feature of the spiketrain
    allspikes : bool
        set True if you want to plot all spikes and you give a list of arrays
    """
    idx_max = np.max(idx)
    if idx_max == 1:
        feature_red = []
        feature_blue = []
        for i in range(len(idx)):
            if idx[i] == 0:
                feature_red.append(feature[i])
            if idx[i] == 1:
                feature_blue.append(feature[i])
        if allspikes is False:
            red_ft = np.array(feature_red)
            blue_ft = np.array(feature_blue)
        if allspikes is True:
            red_ft = np.concatenate(feature_red)
            blue_ft = np.concatenate(feature_blue)
        weights_red = np.ones_like(red_ft)/float(len(red_ft))
        weights_blue = np.ones_like(blue_ft)/float(len(blue_ft))
        plt.hist([red_ft, blue_ft], bins=100,
                 weights=[weights_red, weights_blue], color=['r', 'b'])
        plt.legend(["Red group", "Blue group"])

    if idx_max == 2:
        feature_red = []
        feature_blue = []
        feature_green = []
        for i in range(len(idx)):
            if idx[i] == 0:
                feature_red.append(feature[i])
            if idx[i] == 1:
                feature_blue.append(feature[i])
            if idx[i] == 2:
                feature_green.append(feature[i])
        if allspikes is False:
            red_ft = np.array(feature_red)
            blue_ft = np.array(feature_blue)
            green_ft = np.array(feature_green)
        if allspikes is True:
            red_ft = np.concatenate(feature_red)
            blue_ft = np.concatenate(feature_blue)
            green_ft = np.concatenate(feature_green)
        weights = []
        for i in [red_ft, blue_ft, green_ft]:
            weights.append(np.ones_like(i)/float(len(i)))
        plt.hist([red_ft, blue_ft, green_ft], bins=100, weights=weights,
                 color=['r', 'b', 'g'])
        plt.legend(["Red group", "Blue group", "Green group"])

    if idx_max == 3:
        feature_red = []
        feature_blue = []
        feature_green = []
        feature_purple = []
        for i in range(len(idx)):
            if idx[i] == 0:
                feature_red.append(feature[i])
            if idx[i] == 1:
                feature_blue.append(feature[i])
            if idx[i] == 2:
                feature_green.append(feature[i])
            if idx[i] == 3:
                feature_purple.append(feature[i])
        if allspikes is False:
            red_ft = np.array(feature_red)
            blue_ft = np.array(feature_blue)
            green_ft = np.array(feature_green)
            purple_ft = np.array(feature_purple)
        if allspikes is True:
            red_ft = np.concatenate(feature_red)
            blue_ft = np.concatenate(feature_blue)
            green_ft = np.concatenate(feature_green)
            purple_ft = np.concatenate(feature_purple)
        weights = []
        for i in [red_ft, blue_ft, green_ft, purple_ft]:
            weights.append(np.ones_like(i)/float(len(i)))
        plt.hist([red_ft, blue_ft, green_ft, purple_ft], bins=100,
                 weights=weights, color=['r', 'b', 'g', 'm'])
        plt.legend(["Red group", "Blue group", "Green group", "Purple group"])

    if idx_max == 4:
        feature_red = []
        feature_blue = []
        feature_green = []
        feature_purple = []
        feature_black = []
        for i in range(len(idx)):
            if idx[i] == 0:
                feature_red.append(feature[i])
            if idx[i] == 1:
                feature_blue.append(feature[i])
            if idx[i] == 2:
                feature_green.append(feature[i])
            if idx[i] == 3:
                feature_purple.append(feature[i])
            if idx[i] == 4:
                feature_black.append(feature[i])
        if allspikes is False:
            red_ft = np.array(feature_red)
            blue_ft = np.array(feature_blue)
            green_ft = np.array(feature_green)
            purple_ft = np.array(feature_purple)
            black_ft = np.array(feature_black)
        if allspikes is True:
            red_ft = np.concatenate(feature_red)
            blue_ft = np.concatenate(feature_blue)
            green_ft = np.concatenate(feature_green)
            purple_ft = np.concatenate(feature_purple)
            black_ft = np.concatenate(feature_black)
        weights = []
        for i in [red_ft, blue_ft, green_ft, purple_ft, black_ft]:
            weights.append(np.ones_like(i)/float(len(i)))
        plt.hist([red_ft, blue_ft, green_ft, purple_ft, black_ft], bins=100,
                 weights=weights, color=['r', 'b', 'g', 'm', 'k'])
        plt.legend(["Red group", "Blue group", "Green group", "Purple group",
                    "Black group"])
    plt.xlabel('Feature')
    plt.show()
