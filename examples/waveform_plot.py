import numpy as np
import matplotlib.pyplot as plt
import quantities as pq


def plot_waveforms(sptr, color='r', fig=None, title='waveforms', lw=2, gs=None):
    """
    Visualize waveforms on respective channels

    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure

    Returns
    -------
    out : fig
    """
    import matplotlib.gridspec as gridspec
    nrc = sptr.waveforms.shape[1]
    if fig is None:
        fig = plt.figure()
    axs = []
    ax = None
    for c in range(nrc):
        if gs is None:
            ax = fig.add_subplot(1, nrc, c+1, sharex=ax, sharey=ax)
        else:
            gs0 = gridspec.GridSpecFromSubplotSpec(1, nrc, subplot_spec=gs)
            ax = fig.add_subplot(gs0[:, c], sharex=ax, sharey=ax)
        axs.append(ax)
    for c in range(nrc):
        wf = sptr.waveforms[:, c, :]
        m = np.mean(wf, axis=0)
        stime = np.arange(m.size, dtype=np.float32)/sptr.sampling_rate
        stime.units = 'ms'
        sd = np.std(wf, axis=0)
        axs[c].plot(stime, m, color=color, lw=lw)
        axs[c].fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
        if sptr.left_sweep is not None:
            sptr.left_sweep.units = 'ms'
            axs[c].axvspan(sptr.left_sweep, sptr.left_sweep, color='k',
                           ls='--')
        axs[c].set_xlabel(stime.dimensionality)
        axs[c].set_xlim([stime.min(), stime.max()])
        if c > 0:
            plt.setp(axs[c].get_yticklabels(), visible=False)
    axs[0].set_ylabel(r'amplitude $\pm$ std [%s]' % wf.dimensionality)
    fig.suptitle(title)
    return fig


def plot_largest_waveform(sptr, color='r', ax=None, title='waveforms', lw=2,
                          ylabel=True, xlabel=True):
    """
    Visualize waveforms on respective channels

    Parameters
    ----------
    sptr : neo.SpikeTrain
    color : color of waveforms
    title : figure title
    fig : matplotlib figure

    Returns
    -------
    out : fig
    """
    nrc = sptr.waveforms.shape[1]
    if ax is None:
        fig, ax = plt.subplots(111)
    maxs = []
    for c in range(nrc):
        maxs.append(np.mean(sptr.waveforms[:, c, :], axis=0).max())
    c = np.argmax(maxs)
    wf = sptr.waveforms[:, c, :]
    m = np.mean(wf, axis=0)
    stime = np.arange(m.size, dtype=np.float32)/sptr.sampling_rate
    stime.units = 'ms'
    sd = np.std(wf, axis=0)
    ax.plot(stime, m, color=color, lw=lw)
    ax.fill_between(stime, m-sd, m+sd, alpha=.1, color=color)
    if sptr.left_sweep is not None:
        sptr.left_sweep.units = 'ms'
        ax.axvspan(sptr.left_sweep.rescale('ms'), sptr.left_sweep.rescale('ms'),
                   color='k', ls='--')
    if xlabel:
        ax.set_xlabel(stime.dimensionality)
    ax.set_xlim([stime.min(), stime.max()])
    if ylabel:
        ax.set_ylabel(r'amplitude $\pm$ std [%s]' % m.dimensionality)


def plot_amp_clusters(sptrs, colors=None, fig=None, title=None, gs=None):
    """
    Visualize clustering on amplitude at detection point

    Parameters
    ----------
    sptrs : list of neo.SpikeTrains with same number of recording channels
    color : color of spikes
    title : figure title
    fig : matplotlib figure

    Returns
    -------
    out : fig
    """
    nrc = sptrs[0].waveforms.shape[1]
    if fig is None:
        fig = plt.figure()
    import matplotlib.gridspec as gridspec
    if gs is None:
        gs0 = gridspec.GridSpec(nrc-1, nrc-1)
    else:
        gs0 = gridspec.GridSpecFromSubplotSpec(nrc-1, nrc-1, subplot_spec=gs)
    axs = []
    for x in range(nrc-1):
        for y in range(nrc-1):
            if y <= x:
                ax = fig.add_subplot(gs0[x, y])
                axs.append(ax)
                ax.set_xticks([])
                ax.set_yticks([])
                if x == nrc-2:
                    ax.set_xlabel('channel %i' % (range(nrc)[y-1]))
                if y == 0:
                    ax.set_ylabel('channel %i' % (x))
    if colors is None:
        from matplotlib.pyplot import cm
        colors = cm.rainbow(np.linspace(0, 1, len(sptrs)))
    for idx, sptr in enumerate(sptrs):
        cnt = 0
        wf = sptr.waveforms
        if sptr.left_sweep is None:
            sptr.left_sweep = 0.2 * pq.ms
        mask = int(sptr.sampling_rate*sptr.left_sweep.rescale('s'))
        color = colors[idx]
        for x in range(nrc-1):
            for y in range(nrc-1):
                if y <= x:
                    axs[cnt].plot(wf[:, y-1, mask], wf[:, x, mask], ls='None',
                                  marker='.', color=color)
                    cnt += 1
    if title is not None:
        fig.suptitle(title)
    return fig


def plot_cluster_mean_spike(idx, feature1, feature2):
    """Plots the average waveform of each spiketrains mean spike in a cluster
    plot based on feature1 and feature2. The sptrs are previously separated in
    groups, from 2 to 5 (depending on given n_clusters), with the function
    cluster_waveform_features().

    Parameters
    ----------
    idx : list of integers
        for example when n_clusters is 2 containts 0s and 1s
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

    plt.xlabel('feature 1')
    plt.ylabel('feature 2')


def plot_histogram_waveform_feature(idx, feature, allspikes=False):
    """Plots the feature of the spiketrains in a histogram. The sptrs are
    previously separated in two groups, red and blue, by the function
    cluster_waveform_features(). Default is plotting a feature of the mean
    spike of the spiketrain. Set 'allspikes=True' if you want to plot a
    histogram over the feature for all the spikes from each spiketrain.

    Parameters
    ----------
    idx : list of integers
        for example when n_clusters is 2 containts 0s and 1s
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

    plt.xlabel('Feature')
