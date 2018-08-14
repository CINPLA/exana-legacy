import neo
import numpy as np
import matplotlib.pyplot as plt
import quantities as pq
from exana.tools import *
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
    import pretty_plotting

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


def plot_psth(spike_train=None, epoch=None, trials=None, xlim=[None, None],
              fig=None, axs=None, legend_loc=1, color='b',
              title='', stim_alpha=.2, stim_color=None,
              stim_label='Stim on', stim_style='patch', stim_offset=0*pq.s,
              rast_ylabel='Trials', rast_size=10,
              hist_color=None, hist_edgecolor=None,
              hist_ylim=None,  hist_ylabel=None,
              hist_output='counts', hist_binsize=None, hist_nbins=100,
              hist_alpha=1.):
    """
    Visualize clustering on amplitude at detection point
    Parameters
    ----------
    spike_train : neo.SpikeTrain
    epoch : neo.Epoch
    trials : list of cut neo.SpikeTrains with same number of recording channels
    xlim : list
        limit of x axis
    fig : matplotlib figure
    axs : matplotlib axes (must be 2)
    legend_loc : 'outside' or matplotlib standard loc
    color : color of spikes
    title : figure title
    stim_alpha : float
    stim_color : str
    stim_label : str
    stim_style : 'patch' or 'line'
    stim_offset : pq.Quantity
        The amount of offset for the stimulus relative to epoch.
    rast_ylabel : str
    hist_color : str
    hist_edgecolor : str
    hist_ylim : list
    hist_ylabel : str
    hist_output : str
        Accepts 'counts', 'rate' or 'mean'.
    hist_binsize : pq.Quantity
    hist_nbins : int
    Returns
    -------
    out : fig
    """
    if fig is None and axs is None:
        fig, (hist_ax, rast_ax) = plt.subplots(2, 1, sharex=True)
    elif fig is not None and axs is None:
        hist_ax = fig.add_subplot(2, 1, 1)
        rast_ax = fig.add_subplot(2, 1, 2, sharex=hist_ax)
    else:
        assert len(axs) == 2
        hist_ax, rast_ax = axs

    if trials is None:
        assert spike_train is not None and epoch is not None
        t_start = xlim[0] or 0 * pq.s
        t_stop = xlim[1] or epoch.durations[0]
        trials = make_spiketrain_trials(epoch=epoch, t_start=t_start,
                                        t_stop=t_stop, spike_train=spike_train)
    else:
        assert spike_train is None
    dim = trials[0].times.dimensionality
    if stim_style == 'patch':
        if epoch is not None:
            stim_duration = epoch.durations.rescale(dim).magnitude.max()
        else:
            warnings.warn('Unable to acquire stimulus duration, setting ' +
                          'stim_style to "line". Please provede "epoch"' +
                          ' in order to use stim_style "patch".')
            stim_style = 'line'
    # raster
    plot_raster(trials, color=color, ax=rast_ax, ylabel=rast_ylabel,
                marker_size=rast_size)
    # histogram
    hist_color = color if hist_color is None else hist_color
    hist_ylabel = hist_output if hist_ylabel is None else hist_ylabel
    plot_spike_histogram(trials, color=hist_color, ax=hist_ax,
                         output=hist_output, binsize=hist_binsize,
                         nbins=hist_nbins, edgecolor=hist_edgecolor,
                         ylabel=hist_ylabel, alpha=hist_alpha)
    if hist_ylim is not None: hist_ax.set_ylim(hist_ylim)
    # stim representation
    stim_color = color if stim_color is None else stim_color
    if stim_style == 'patch':
        fill_stop = stim_duration
        import matplotlib.patches as mpatches
        line = mpatches.Patch([], [], color=stim_color, label=stim_label,
                              alpha=stim_alpha)
    elif stim_style == 'line':
        fill_stop = 0
        import matplotlib.lines as mlines
        line = mlines.Line2D([], [], color=stim_color, label=stim_label)
    stim_offset = stim_offset.rescale(dim).magnitude
    hist_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                    alpha=stim_alpha, zorder=0)
    rast_ax.axvspan(stim_offset, fill_stop + stim_offset, color=stim_color,
                    alpha=stim_alpha, zorder=0)
    if legend_loc == 'outside':
        hist_ax.legend(handles=[line], bbox_to_anchor=(0., 1.02, 1., .102),
                       loc=4, ncol=2, borderaxespad=0.)
    else:
        hist_ax.legend(handles=[line], loc=legend_loc, ncol=2, borderaxespad=0.)
    if title is not None: hist_ax.set_title(title)
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
