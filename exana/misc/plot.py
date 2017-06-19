import matplotlib.pyplot as plt

def simpleaxis(ax, left=True, right=True, top=True, bottom=True, ticks=True):
    """
    Removes axis lines
    """
    try:
        ax.spines['top'].set_visible(top)
        ax.spines['right'].set_visible(right)
        ax.spines['left'].set_visible(left)
        ax.spines['bottom'].set_visible(bottom)
    except AttributeError:
        pass
    except:
        raise
    if not bottom and not ticks:
        ax.get_xaxis().tick_bottom()
        plt.setp(ax.get_xticklabels(), visible=False)
    if not left and not ticks:
        ax.get_yaxis().tick_left()
        plt.setp(ax.get_yticklabels(), visible=False)


def floating_axis(ax, origin=[0, 0], ending=[1, 1], xlabel=None, ylabel=None,
                  xhpadding=1, yhpadding=1.01, xvpadding=1.05, yvpadding=1.75,
                  **kwargs):
    """
    Floating axis for axes scale.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes return by matplotlib
    origin : list
        The origin [x, y] of your axis, relative to normalized axes i.e. between
        zero and one.
    ending : list
        The length [x, y] of the axis arms relative to normalized axes i.e.
        between zero and one.
    xlabel : str
        The label on the horizontal axis.
    ylabel : str
        The label on the vertical axis.
    xhpadding : float
        The amount of horizontal padding on the x label.
    yhpadding : float
        The amount of horizontal padding on the y label.
    xvpadding : float
        The amount of vertical padding on the x label.
    xvpadding : float
        The amount of vertical padding on the y label.
    kwargs : dict
        Keyword arguments set to matplotlib.axes.Axes.axhline/axvline.
    """
    simpleaxis(ax, bottom=False, left=False, top=False, right=False,
               ticks=False)

    def trans(val, lim):
        return (val - lim[0])/abs(lim[1] - lim[0])
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.axhline(y=origin[1], xmin=trans(ending[0], xlim),
               xmax=trans(origin[0], xlim), **kwargs)
    ax.axvline(x=origin[0], ymin=trans(ending[1], ylim),
               ymax=trans(origin[1], xlim), **kwargs)
    if xlabel is None:
        xlabel = '{}'.format(abs(origin[0] - ending[0]))
    else:
        xlabel = '{} {}'.format(abs(origin[0] - ending[0]), xlabel)
    if ylabel is None:
        ylabel = '%s' % abs(origin[1] - ending[1])
    else:
        ylabel = '{} {}'.format(abs(origin[1] - ending[1]), ylabel)
    ax.text(ending[0]*xhpadding, origin[1]*xvpadding, xlabel)
    ax.text(origin[0]*yhpadding, ending[1]*yvpadding, ylabel, rotation=270)


def upper_legend(ax, label='', color='b'):
    """
    Legend on the top right side of your axes.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes return by matplotlib.
    label : str
        Label text.
    color : str
        Color of line.
    """
    import matplotlib.lines as mlines
    line = mlines.Line2D([], [], color=color, label=label)
    ax.legend(handles=[line],
              bbox_to_anchor=(0., 1.02, 1., .102), loc=4,
              ncol=2, borderaxespad=0.)


def addticks(ax, newLocs, newLabels, pos='x'):
    # Draw to get ticks
    plt.draw()

    # Get existing ticks
    if pos == 'x':
        locs = ax.get_xticks().tolist()
        labels = [x.get_text() for x in ax.get_xticklabels()]
    elif pos == 'y':
        locs = ax.get_yticks().tolist()
        labels = [x.get_text() for x in ax.get_yticklabels()]
    else:
        raise AssertionError("WRONG pos. Use 'x' or 'y'")

    # Build dictionary of ticks
    Dticks = dict(zip(locs, labels))

    # Add/Replace new ticks
    for Loc, Lab in zip(newLocs, newLabels):
        Dticks[Loc] = Lab

    # Get back tick lists
    locs = list(Dticks.keys())
    labels = list(Dticks.values())

    # Generate new ticks
    if pos == 'x':
        ax.set_xticks(locs)
        ax.set_xticklabels(labels)
    elif pos == 'y':
        ax.set_yticks(locs)
        ax.set_yticklabels(labels)
