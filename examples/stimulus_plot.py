import neo
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from .tools import *
from utils import simpleaxis
from statistics_plot import (plot_spike_histogram)
from general_plot import (plot_raster)
from exana.statistics.tools import (fano_factor_multiunit)


def polar_tuning_curve(orients, rates, ax=None, params={}):
    """
    Direction polar tuning curve
    """
    import numpy as np
    import math
    from exana.misc import pretty_plotting

    assert len(orients) == len(rates)

    if ax is None:
        fig, ax = plt.subplots()
        ax = plt.subplot(111, projection='polar')

    ax.plot(orients, rates, '-', **params)
    ax.fill(orients, rates, alpha=1)
    ax.set_yticklabels([])
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    return ax


def plot_tuning_overview(trials, spontan_rate=None):
    """
    Makes orientation tuning plots (line and polar plot)
    for each stimulus orientation.

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    spontan_rates : defaultdict(dict), optional
        rates[channel_index_name][unit_id] = spontaneous firing rate trials.
    """
    import seaborn
    from exana.stimulus.tools import (make_orientation_trials,
                                      compute_orientation_tuning,
                                      compute_osi)
    fig = plt.figure()

    ax1 = fig.add_subplot(1, 2, 1)
    trials = make_orientation_trials(trials)
    rates, orients = compute_orientation_tuning(trials)
    preferred_orient, index = compute_osi(rates, orients)

    ax1.set_title("Preferred orientation={},\n OSI={}".format(preferred_orient,
                                                              round(index, 2)))
    ax1.plot(orients, rates, "-o", label="with bkg")
    ax1.set_xlabel("Orientation")
    ax1.set_ylabel("Rate (1/s)")

    ax2 = fig.add_subplot(1, 2, 2, projection="polar")
    polar_tuning_curve(orients.rescale("rad"), rates, ax=ax2)

    if spontan_rate is not None:
        ax1.plot(orients, rates - spontan_rate, "-o", label="without bkg")
        ax1.legend()

    fig.tight_layout()

    return fig


def orient_raster_plots(trials):
    """
    Makes raster plot for each stimulus orientation

    Parameters
    ----------
    trials : list
        list of neo.SpikeTrain
    """
    import seaborn
    orient_trials = make_orientation_trials(trials)
    col_count = 4
    row_count = int(np.ceil(len(orient_trials))/col_count)
    fig = plt.figure(figsize=(2*col_count, 2*row_count))
    for i, (orient, trials) in enumerate(orient_trials.items()):
        ax = fig.add_subplot(row_count, col_count, i+1)
        ax = plot_raster(trials, ax=ax)
        ax.set_title(orient)
        ax.grid(False)
    fig.tight_layout()

    return fig


def plot_psth(sptr=None, epoch=None, t_start=None, t_stop=None, trials=None,
              output='counts', binsize=None, bins=100, fig=None,
              color='b', title='plot_psth', stim_color='b', edgecolor='k',
              alpha=.2, label='stim on', legend_loc=1, legend_style='patch',
              axs=None, hist_ylabel=True, rast_ylabel='trials',
              ylim=None, offset=0 * pq.s):
    """
    Visualize clustering on amplitude at detection point

    Parameters
    ----------
    sptr : neo.SpikeTrain
    trials : list of cut neo.SpikeTrains with same number of recording channels
    color : color of spikes
    title : figure title
    fig : matplotlib figure
    axs : matplotlib axes (must be 2)
    legend_loc : 'outside' or matplotlib standard loc
    legend_style : 'patch' or 'line'

    Returns
    -------
    out : fig
    """
    if fig is None and axs is None:
        fig, (ax, ax2) = plt.subplots(2, 1, sharex=True)
    elif fig is not None and axs is None:
        ax = fig.add_subplot(2, 1, 1)
        ax2 = fig.add_subplot(2, 1, 2, sharex=ax)
    else:
        assert len(axs) == 2
        ax, ax2 = axs
    if trials is None:
        assert sptr is not None
        assert epoch is not None
        t_start = t_start or 0 * pq.s
        t_stop = t_stop or epoch.durations[0]
        trials = make_spiketrain_trials(epoch=epoch, t_start=t_start, t_stop=t_stop,
                                        spike_train=sptr)
        dim = sptr.times.dimensionality
        stim_duration = epoch.durations.rescale(dim).magnitude.max()
    else:
        dim = trials[0].times.dimensionality
        if legend_style == 'patch':
            if epoch is not None:
                stim_duration = epoch.durations.rescale(dim).magnitude.max()
            else:
                import warnings
                warnings.warn('Unable to acquire stimulus duration, setting ' +
                              'legend_style to "line". Please provede "epoch"' +
                              ' in order to use legend_style "patch".')
                legend_style = 'line'

    plot_spike_histogram(trials, color=color, ax=ax, output=output,
                         binsize=binsize, bins=bins, edgecolor=edgecolor,
                         ylabel=hist_ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    plot_raster(trials, color=color, ax=ax2, ylabel=rast_ylabel)
    if legend_style == 'patch':
        fill_stop = stim_duration
        import matplotlib.patches as mpatches
        line = mpatches.Patch([], [], color=stim_color, label=label, alpha=alpha)
    elif legend_style == 'line':
        fill_stop = 0
        import matplotlib.lines as mlines
        line = mlines.Line2D([], [], color=stim_color, label=label)
    offset = offset.rescale('s').magnitude
    ax.axvspan(offset, fill_stop + offset, color=stim_color, alpha=alpha)
    ax2.axvspan(offset, fill_stop + offset, color=stim_color, alpha=alpha)
    if legend_loc == 'outside':
        ax.legend(handles=[line], bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                  ncol=2, borderaxespad=0.)
    else:
        ax.legend(handles=[line], loc=legend_loc, ncol=2, borderaxespad=0.)
    if title is not None:
        ax.set_title(title)
    return fig


def plot_stimulus_overview(unit_trials, t_start, t_stop, binsize, title=None,
                           axs=None):
    '''plots an overview of many units where each unit has several trials,
    output is rate, raster and fano factor'''
    dim = 's'
    t_start = t_start.rescale(dim)
    t_stop = t_stop.rescale(dim)
    binsize = binsize.rescale(dim)
    bins = np.arange(t_start, t_stop + binsize, binsize)
    fano, ci, rates, bins = fano_factor_multiunit(unit_trials=unit_trials,
                                                  bins=bins,
                                                  return_rates=True,
                                                  return_bins=True)
    all_trials = [trial for trials in unit_trials for trial in trials]
    if axs is None:
        f = plt.figure()
        ax = f.add_subplot(3, 1, 1)
        ax2 = f.add_subplot(3, 1, 2)
        ax3 = f.add_subplot(3, 1, 3)
    else:
        ax, ax2, ax3 = axs
    if title is not None:
        ax.set_title(title)
    ax.bar(bins[0:-1], rates, width=bins[1]-bins[0])
    ax.set_xlim(t_start, t_stop)
    ax.set_ylabel('mean rate')
    ax.axvspan(0,0,color='r')
    import matplotlib.lines as mlines
    line = mlines.Line2D([], [], color='r', label='stim on')
    ax.legend(handles=[line], bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
                ncol=2, borderaxespad=0.)
    simpleaxis(ax, bottom=False, ticks=False)
    plot_raster(all_trials, ax=ax2)
    ax2.axvspan(0,0, color='r')
    simpleaxis(ax2, bottom=False, ticks=False)
    ax3.bar(bins[0:-1], fano, width=bins[1]-bins[0])
    ax3.set_xlim(t_start, t_stop)
    ax3.set_ylabel('Fano factor')
    ax3.set_xlabel('time [%s]' % bins.dimensionality)
    ax3.axvspan(0, 0, color='r')
    simpleaxis(ax3)
    plt.tight_layout()
