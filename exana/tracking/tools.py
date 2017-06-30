import numpy as np
import quantities as pq
from ..misc.tools import is_quantities, normalize


def _cut_to_same_len(*args):
    out = []
    lens = []
    for arg in args:
        lens.append(len(arg))
    minlen = min(lens)
    for arg in args:
        out.append(arg[:minlen])
    return out


def monotonously_increasing(var):
    return all(x < y for x, y in zip(var , var[1:]))


def remove_eqal_times(time, *args):
    idxs, = np.where([x == y for x, y in zip(time , time[1:])])
    out = []
    for arg in args:
        out.append(np.delete(arg, idxs+1))
    return np.delete(time, idxs+1), out


def remove_smaller_times(time, *args):
    idxs, = np.where([x > y for x, y in zip(time , time[1:])])
    out = []
    for arg in args:
        out.append(np.delete(arg, idxs+1))
    return np.delete(time, idxs+1), out


def get_processed_position(exdir_path):
    """
    Get postion data from exdir position group

    Parameters
    ----------
    exdir_path : str
        Path to exdir file.

    Returns
    ----------
    out : tuple
        Positions and head direction(x, y, t, ang, ang_t)
    """
    import exdir
    exdir_group = exdir.File(exdir_path)
    position_group = exdir_group['processing']['tracking']['camera_0']['Position']
    x1, y1, t1 = tr.get_raw_position(position_group['led_0'])
    x2, y2, t2 = tr.get_raw_position(position_group['led_1'])
    x1, y1, t1 = fix_not_monotonous_timestamps(x1, y1, t1)
    x2, y2, t2 = fix_not_monotonous_timestamps(x2, y2, t2)
    x, y, t = tr.select_best_position(x1, y1, t1, x2, y2, t2)
    ang, ang_t = tr.head_direction(x1, y1, x2, y2, t1,
                                             return_rad=False)
    x, y, t = tr.interp_filt_position(x, y, t, pos_fs=par['pos_fs'], f_cut=par['f_cut'])
    return x, y, t, ang, ang_t


def get_raw_position(spot_group):
        """
        Get postion data from exdir led group

        Parameters
        ----------
        spot_group : exdir.group

        Returns
        ----------
        out : x, y, t
            1d vectors with position and time from LED
        """
        coords = spot_group["data"]
        t = spot_group["timestamps"].data
        x = coords[:, 0]
        y = coords[:, 1]

        return x, y, t


def fix_not_monotonous_timestamps(x, y, t):
    if not monotonously_increasing(t):
        t_unit = t.units
        x_unit = x.units
        y_unit = y.units
        import warnings
        warnings.warn('Time is not monotonously increasing, ' +
                      'removing equal and smaller timestamps.')
        t, (x, y) = remove_eqal_times(t, x, y)
        if min(t) != t[0]:
            idx = np.where(t == min(t))
            t = t[idx:]
            x = x[idx:]
            y = y[idx:]
        if not monotonously_increasing(t):
            raise ValueError('Unable to fix timestamps please revise them.')
    return x * x_unit, y * y_unit, t * t_unit


def get_tracking(postion_group):
        """
        Get postion data from exdir position group

        Parameters
        ----------
        position_group : exdir.group

        Returns
        ----------
        out : dict
            dictionary with position and time from leds in position group
        """

        tracking = {}
        for name, group in postion_group.items():
            x, y, t = get_raw_position(spot_group=group)
            # TODO: Remove nans etc
            tracking[name] = {"x": x, "y": y, "t": t}
        return tracking


def select_best_position(x1, y1, t1, x2, y2, t2, speed_filter=5 * pq.m / pq.s):
    """
    selects position data with least nan after speed filtering

    Parameters
    ----------
    x1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    y1 : quantities.Quantity array in m
        1d vector of x positions from LED 1
    t1 : quantities.Quantity array in s
        1d vector of times from LED 1 at x, y positions
    x2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    y2 : quantities.Quantity array in m
        1d vector of x positions from LED 2
    t2 : quantities.Quantity array in s
        1d vector of times from LED 2 at x, y positions
    speed_filter : None or quantities in m/s
        threshold filter for translational speed
    """
    is_quantities([x1, y1, t1, x2, y2, t2], 'vector')
    x1, y1, t1, x2, y2, t2 = _cut_to_same_len(x1, y1, t1, x2, y2, t2)
    is_quantities(speed_filter, 'scalar')
    measurements1 = len(x1)
    measurements2 = len(x2)
    x1, y1, t1 = rm_nans(x1, y1, t1)
    x2, y2, t2 = rm_nans(x2, y2, t2)
    if speed_filter is not None:
        x1, y1, t1 = velocity_threshold(x1, y1, t1, speed_filter)
        x2, y2, t2 = velocity_threshold(x2, y2, t2, speed_filter)

    if len(x1) > len(x2):
        print('Removed %.2f %% invalid measurements in path' %
              ((1. - len(x1) / float(measurements1)) * 100.))
        x = x1
        y = y1
        t = t1
    else:
        print('Removed %.2f %% invalid measurements in path' %
              ((1. - len(x2) / float(measurements2)) * 100.))
        x = x2
        y = y2
        t = t2
    return x, y, t


def interp_filt_position(x, y, tm, box_xlen=1 * pq.m, box_ylen=1 * pq.m,
                         pos_fs=100 * pq.Hz, f_cut=10 * pq.Hz):
    """
    Calculeate head direction in angles or radians for time t

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    tm : quantities.Quantity array in s
        1d vector of times at x, y positions
    pos_fs : quantities scalar in Hz
        return radians

    Returns
    -------
    out : angles, resized t
    """
    import scipy.signal as ss
    assert len(x) == len(y) == len(tm), 'x, y, t must have same length'
    is_quantities([x, y, tm], 'vector')
    is_quantities([pos_fs, box_xlen, box_ylen, f_cut], 'scalar')
    spat_dim = x.units
    t = np.arange(tm.min(), tm.max() + 1. / pos_fs, 1. / pos_fs) * tm.units
    x = np.interp(t, tm, x)
    y = np.interp(t, tm, y)
    # rapid head movements will contribute to velocity artifacts,
    # these can be removed by low-pass filtering
    # see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    # code addapted from Espen Hagen
    b, a = ss.butter(N=1, Wn=f_cut * 2 / pos_fs)
    # zero phase shift filter
    x = ss.filtfilt(b, a, x) * spat_dim
    y = ss.filtfilt(b, a, y) * spat_dim
    # we tolerate small interpolation errors
    x[(x > -1e-3) & (x < 0.0)] = 0.0 * spat_dim
    y[(y > -1e-3) & (y < 0.0)] = 0.0 * spat_dim
    if np.isnan(x).any() and np.isnan(y).any():
        raise ValueError('nans found in  position, ' +
            'x nans = %i, y nans = %i' % (sum(np.isnan(x)), sum(np.isnan(y))))
    if (x.min() < 0 or x.max() > box_xlen or y.min() < 0 or y.max() > box_ylen):
        raise ValueError(
            "Interpolation produces path values " +
            "outside box: min [x, y] = [{}, {}], ".format(x.min(), y.min()) +
            "max [x, y] = [{}, {}]".format(x.max(), y.max()))

    R = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    V = R / np.diff(t)
    print('Maximum speed {}'.format(V.max()))
    return x, y, t


def rm_nans(*args):
    """
    Removes nan from all corresponding arrays

    Parameters
    ----------
    args : arrays, lists or quantities which should have removed nans in
           all the same indices

    Returns
    -------
    out : args with removed nans
    """
    nan_indices = []
    for arg in args:
        nan_indices.extend(np.where(np.isnan(arg))[0].tolist())
    nan_indices = np.unique(nan_indices)
    out = []
    for arg in args:
        if isinstance(arg, pq.Quantity):
            unit = arg.units
        else:
            unit = 1
        out.append(np.delete(arg, nan_indices) * unit)
    return out


def velocity_threshold(x, y, t, threshold):
    """
    Removes values above threshold

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of y positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    threshold : float
    """
    assert len(x) == len(y) == len(t), 'x, y, t must have same length'
    is_quantities([x, y, t], 'vector')
    is_quantities(threshold, 'scalar')
    r = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    v = np.divide(r, np.diff(t))
    speed_lim = np.concatenate(([False], v > threshold), axis=0)
    x[speed_lim] = np.nan * x.units
    y[speed_lim] = np.nan * y.units
    x, y, t = rm_nans(x, y, t)
    return x, y, t
