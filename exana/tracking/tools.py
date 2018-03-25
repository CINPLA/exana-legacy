import numpy as np
import quantities as pq
from ..misc.tools import is_quantities, normalize
from ..misc.peakdetect import peakdetect
from .head import head_direction
import pdb
import scipy.signal as sig


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
    return all(x < y for x, y in zip(var, var[1:]))


def remove_eqal_times(time, *args):
    idxs, = np.where([x == y for x, y in zip(time, time[1:])])
    out = []
    for arg in args:
        out.append(np.delete(arg, idxs + 1))
    return np.delete(time, idxs + 1), out


def remove_smaller_times(time, *args):
    idxs, = np.where([x > y for x, y in zip(time, time[1:])])
    out = []
    for arg in args:
        out.append(np.delete(arg, idxs + 1))
    return np.delete(time, idxs + 1), out


def get_processed_tracking(exdir_path, par, return_rad=False):
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
    processing = exdir_group['processing']
    assert 'tracking' in processing, 'No tracking recorded.'
    position_group = processing['tracking']['camera_0']['Position']
    x1, y1, t1 = get_raw_position(position_group['led_0'])
    x1, y1, t1 = fix_nonmonotonous_timestamps(x1, y1, t1)
    if 'led_1' in position_group:
        x2, y2, t2 = get_raw_position(position_group['led_1'])
        x2, y2, t2 = fix_nonmonotonous_timestamps(x2, y2, t2)
        ang, ang_t = head_direction(x1, y1, x2, y2, t1, return_rad=return_rad)
        x, y, t = select_best_position(x1, y1, t1, x2, y2, t2)
    else:
        x, y, t, ang, ang_t = x1, y1, t1, None, None
    x, y, t = interp_filt_position(x, y, t, pos_fs=par['pos_fs'],
                                      f_cut=par['f_cut'])
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


def fix_nonmonotonous_timestamps(x, y, t):
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
            t, (x, y) = remove_smaller_times(t, x, y)
        if not monotonously_increasing(t):
            raise ValueError('Unable to fix timestamps please revise them.')
        return x * x_unit, y * y_unit, t * t_unit
    else:
        return x, y, t


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
    rapid head movements will contribute to velocity artifacts,
    these can be removed by low-pass filtering
    see http://www.ncbi.nlm.nih.gov/pmc/articles/PMC1876586/
    code addapted from Espen Hagen

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


def unit_vector(v):
    """ Return unit vector of v
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    return v / np.linalg.norm(v)


def angle_between_vectors(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'
    modified from David Wolever,
    https://stackoverflow.com/questions/2827393/angles
    -between-two-n-dimensional-vectors-in-python
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def rescale_linear_track_2d_to_1d(x, y, end_0=[], end_1=[]):
    """ Take x, y coordinates of linear track data, rescale to 1-d.

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    y : quantities.Quantity array in m
        1d vector of x positions
    t : quantities.Quantity array in s
        1d vector of times at x, y positions
    end_0: quantities.Quantity array in m
        linear track endpoint 1, in x, y
    end_1: quantities.Quantity array in m
        linear track endpoint 2, in x, y

    Returns
    -------
    out : 1d vector
    """
    from exana.misc.tools import is_quantities
    if not all([len(var) == len(var2) for var in
                [x, y] for var2 in [x, y]]):
        raise ValueError('x, y, t must have same number of elements')
    is_quantities([x, y], 'vector')
    x = x.rescale('m').magnitude
    y = y.rescale('m').magnitude
    if len(end_0) != 2 or len(end_1) != 2:
        raise ValueError('end_0 and end_1 must be 2d vectors')
    end_0 = end_0.rescale('m').magnitude
    end_1 = end_1.rescale('m').magnitude
    # shift coordinate system to have end_0 as origin
    x -= end_0[0]
    y -= end_0[1]

    # calculate angle of track
    v_x_axis = np.array([1, 0])
    theta = angle_between_vectors(end_1-end_0, v_x_axis)
    # rotate clockwise
    rot_mat = np.array([[np.cos(-theta), -np.sin(-theta)],
                        [np.sin(-theta),  np.cos(-theta)]])
    x_rot = []
    for x_i, y_i in zip(x, y):
        [x_rot_i, _] = np.dot(rot_mat,
                              np.array([[x_i],
                                        [y_i]]))
        x_rot.append(x_rot_i.item())
    # shift x_rot so that np.min(x_rot) == 0
    x_rot -= np.min(x_rot)
    # only consider x_rot in output
    return x_rot*pq.m


def find_laps(peaks_start, peaks_stop, valid_start, valid_stop):
    laps = []
    for t_start, x_start in peaks_start:
        # check if current peak is in valid start position
        if not valid_start[0] <= x_start <= valid_start[1]:
            continue
        # find next maxpeak in time
        res = np.where(peaks_stop[:, 0] > t_start)[0]
        if len(res) == 0:
            continue
        id_stop = res[0]
        t_stop = peaks_stop[id_stop, 0]
        x_stop = peaks_stop[id_stop, 1]
        # check if stop peak is in its valid start zone
        if not valid_stop[0] <= x_stop <= valid_stop[1]:
            continue
        # add start and end time of lap
        laps.append([[t_start, t_stop], [x_start, x_stop]])
    return laps


def identify_laps_on_linear_track(x,
                                  t,
                                  kernel='auto',
                                  val_margin=.3,
                                  track_len='max'):
    """  Individual laps on linear track are identified by,
      a) smoothing trajectory
      b) find peaks of trajectory
      c) connect minimal and maximal peaks, if they are located in
         the respective region at the end of the track.
         The region extends from the end of the track to max_length*val_margin

    Parameters
    ----------
    x : quantities.Quantity array in m
        1d vector of x positions
    t : quantities.Quantity array in s
        1d vector of times at x positions
    kernel : numpy.ndarray|str['auto']
        Kernel for smoothing of trajectory
        If 'auto', boxcar kernel will be used
    val_margin : float
        Fraction of environment that defines valid zone of return
        points at beginning and end of track
    track_len : float|str['max']
        Lenght of linear track
        If 'max', maximal pos value will be used

    Returns
    -------
    laps_start2end, list of [x_start, x_stop],[t_start, t_stop]
    laps_end2start, list of [x_start, x_stop],[t_start, t_stop]

    """
    t = t.rescale('s')
    sampling_rate = np.median(np.diff(t))
    x = x.rescale('m')

    if kernel == 'auto':
        filter_width = int(np.ceil(5*sampling_rate))
        kernel = np.ones(filter_width)/filter_width
    x_filt = sig.convolve(x, kernel, mode='same') * pq.m

    max_peaks, min_peaks = peakdetect(x_filt, t)
    max_peaks = np.array(max_peaks)
    min_peaks = np.array(min_peaks)

    if track_len == 'max':
        x_max = np.max(x_filt)
    else:
        x_max = track_len

    valid_start = [0. * pq.m,
                   val_margin*x_max]
    valid_stop = [x_max - val_margin * x_max, x_max]

    laps_start2end = find_laps(min_peaks, max_peaks, valid_start, valid_stop)
    laps_end2start = find_laps(max_peaks, min_peaks, valid_stop, valid_start)

    return laps_start2end, laps_end2start


def gaussian2D(amp, x, y, xc, yc, s):
    return amp * np.exp(- 0.5 * (((x - xc) / s)**2 + ((y - yc) / s)**2))


def make_test_grid_rate_map(sigma=0.05*np.ones(7), spacing=0.3,
                            amplitude=np.ones(7), dpos=0,
                            box_xlen=1*pq.m, box_ylen=1*pq.m):
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    x = np.linspace(0, box_xlen, 50)
    y = np.linspace(0, box_ylen, 50)
    x,y = np.meshgrid(x,y)

    p0 = np.array((0.5, 0.5)) + dpos
    pos = [p0]

    angles = np.linspace(0, 2 * np.pi, 7)[:-1]

    rate_map = np.zeros_like(x)
    rate_map += gaussian2D(1, x, y, *p0, sigma[0])

    for i, a in enumerate(angles):
        p = p0 + [spacing * f(a) for f in [np.cos, np.sin]]
        rate_map += gaussian2D(amplitude[i], x, y, *p, sigma[i])
        pos.append(p)
    return rate_map, np.array(pos)


def make_test_grid_spike_path(t_stop=10*pq.min, dt=1/(30*pq.Hz), box_xlen=1*pq.m,
                              box_ylen=1*pq.m):
    from elephant.spike_train_generation import homogeneous_poisson_process as hpp
    rate_map, grid_pos = make_test_grid_rate_map(box_xlen=box_xlen,
                                                 box_ylen=box_ylen)
    box_xlen = box_xlen.rescale('m').magnitude
    box_ylen = box_ylen.rescale('m').magnitude
    rate_map = rate_map > 0.1
    ny, nx = rate_map.shape
    xref = np.linspace(0, box_xlen, nx)
    yref = np.linspace(0, box_ylen, ny)
    t_stop = t_stop.rescale('s').magnitude
    dt = dt.rescale('s').magnitude
    time = np.arange(0, t_stop, dt)

    def speed_good(x1, y1, x2, y2, threshold=1.5):
        if any(x is None for x in [x1, y1, x2, y2]):
            return False
        return (np.sqrt((x2 - x1)**2 + (y2 - y1)**2)) / dt < threshold
    x, y, spikes = [0], [0], []
    while len(x) < len(time):
        x2, y2 = None, None
        while not speed_good(x[-1], y[-1], x2, y2):
            x2, y2 = np.random.uniform(0, 1, 2)
            x2, y2 = x2 * box_xlen, y2 * box_ylen
        x.append(x2)
        y.append(y2)
        if in_rate_map(rate_map, x2, y2, xref, yref):
            curr_t = time[len(x) - 1]
            st = hpp(rate=30.0 * pq.Hz, t_start=curr_t * pq.s,
                     t_stop=(curr_t + dt) * pq.s)
            spikes.extend(st.times.magnitude.tolist())


    return x, y, time, spikes


def in_rate_map(rate_map, x, y, xref, yref):
    assert rate_map.dtype == bool
    xdiff = xref - x
    xdiff[xdiff < 0] = np.inf
    xidx = np.argmin(xdiff)
    ydiff = yref - y
    ydiff[ydiff < 0] = np.inf
    yidx = np.argmin(ydiff)
    return rate_map[yidx, xidx]


def gaussian2D_asym(pos, amplitude, xc, yc, sigma_x, sigma_y, theta):
    x,y = pos

    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = amplitude*np.exp( - (a*((x-xc)**2) + 2*b*(x-xc)*(y-yc)
                            + c*((y-yc)**2)))
    return g.ravel()


def fit_gauss_asym(data, p0 = None, return_data=True):
    """Fits an asymmetric 2D gauss function to the given data set, with optional guess
    parameters. Optimizes amplitude, center coordinates, sigmax, sigmay and
    angle. If no guess parameters, initializes with a thin gauss bell
    centered at the data maxima

    Parameters
    -----------
    data        : 2D np array
    p0 (optional): arraylike
                  initial parameters [amplitude,x_center,y_center,sigma_x, sigma_y,angle]
    return_data : bool



    Returns
    --------
    params      : tuple of params: (amp,xc,yc,sigmax, sigmay, angle)
    (if return_data) data_fitted : 2D np array
                                   the fitted gauss data

    """
    from scipy.optimize import curve_fit
    # Create x and y indices
    sx, sy = data.shape
    xmin, xmax = 0, 1
    ymin, ymax = 0, 1
    x = np.linspace(xmin, xmax, sx)
    y = np.linspace(ymin, ymax, sy)
    x, y = np.meshgrid(x, y)

    if p0 is None:
        # initial guesses, use small gaussian at maxima as initial guess
        ia =     np.max(data)                                # amplitude
        index = np.unravel_index(np.argmax(data), (sx, sy))  # center
        ix, iy = x[index], y[index]
        isig =   0.01
        iang = 0

        p0 = (ia, ix, iy, isig, isig, iang)

    popt, pcov = curve_fit(gaussian2D_asym, (x, y), data.ravel(), p0=p0)
    # TODO : Add test for pcov
    if return_data:
        data_fitted = gaussian2D_asym((x, y), *popt)
        return popt, data_fitted.reshape(sx,sy)
    else:
        return popt


def separation_error_func(smoothing, lpl_thrsh, rate_map):
    """
    Gives a measure of how well the smoothing and laplace-threshold factors
    separates a rate_map into hexagonal fields.
    Measures the deviation of the distance from each bump to its two
    closest neighbors from the average distance as gotten from
    tr.fields.find_avg_dist, and the relative difference in area of each of
    the fields.

    Parameters
    -----------
        smoothing : float
            size of the smoothing kernel relative to the box
        lpl_thrsh : float

    laplace_thrsh : float
        value of laplacian to separate fields by relative to the minima.
        see exana.tracking.fields.separate_fields

    Returns
    -------
        err : float
            0 if all fields exact same size and distance from two closest
            neighbors
    """
    from astropy.convolution import Gaussian2DKernel, convolve_fft

    if np.isnan(smoothing):
        return np.inf

    import exana.tracking as tr

    rate_map[np.isnan(rate_map)] = 0.

    csize = rate_map.shape[0] * smoothing
    kernel = Gaussian2DKernel(csize)
    rm_smooth = convolve_fft(rate_map, kernel)  # TODO edge correction

    f, nf, bc = tr.fields.separate_fields(rm_smooth, laplace_thrsh=lpl_thrsh,
                                            cutoff_method = 'median',
                                            center_method = 'maxima')

    avg_dist = tr.fields.find_avg_dist(rm_smooth, thrsh = 0.1)

    if nf < 3:
        return np.inf
    if np.isnan(avg_dist):
        return np.inf

    indx = np.arange(1,nf+1)
    err = 0


    # Slower:
    # areas = np.zeros(nf)
    # for i in range(nf):
    #     areas[i] = np.sum(f==(i+1))
    # Faster: (~ 2x, depends on indx.size, bigger loop = more gain )
    areas = np.sum(f.ravel() == indx[:,None], axis = 1)

    # Slower:
    # area_deviation = 0
    # for i in range(nf):
    #     for j in range(i+1,nf):
    #         Ai = areas[i]
    #         Aj = areas[j]
    #         area_diff = (Ai - Aj)**2/(Ai*Aj)
    #         area_deviation += area_diff
    # Faster: (~ 4x)
    area_deviation = np.sum(((areas[:,None] - areas)**2/(areas[:,None] * areas)))
    err += area_deviation

    # Slower:
    # dist_deviations = np.zeros(nf)
    # for i in indx:
    #     bump = bc[i-1]

    #     rel = bc - bump
    #     dist = np.linalg.norm(rel, axis=1)
    #     sort = np.argsort(dist)

    #     # add relative difference in area to all other fields
    #     dist_diff = (dist[sort][1:3] - avg_dist)
    #     dist_deviations[i-1] = np.sum(dist_diff**2)/2
    # err += np.sum(dist_deviations**2)/nf
    # Faster (~ 4x)
    dists = np.linalg.norm(bc[:,None,:] - bc, axis = -1)
    dist_diffs = np.sort(dists)[:,1:3] - avg_dist
    dist_deviation = np.sum(np.sum(dist_diffs**2, axis = 1)**2)
    #err += dist_deviation
    # oneliner forzelulz
    # err += np.sum(np.sum((np.sort(np.linalg.norm(bc[:,None,:] - bc, axis = -1))[:,1:3] - avg)**2, axis = 1)**2)

    field_mask = f > 0

    # measure of the spike rates covered by the fields
    # TWO WAYS TO DO THIS: measure over original rate map, or over smoothed rate map
    #field_coverage =  np.sum(rm_smooth) / np.sum(rm_smooth[field_mask])
    field_coverage =  np.sum(rate_map) / np.sum(rate_map[field_mask])
    # total spike rate

    err = err*field_coverage/nf
    return err
