import numpy as np
import neo
import quantities as pq
from ..misc.tools import find_first_peak, is_quantities


###############################################################################
#                      functions for organizing data
###############################################################################
def _rescale_orients(trials, unit=pq.deg):
    """
    Rescales all orient annotations to the same unit
    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation 'orient' (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit. Default is degree.
    """
    if unit not in [pq.deg, pq.rad]:
        raise ValueError("unit can only be deg or rad, ", str(unit))

    for trial in trials:
        orient = trial.annotations["orient"]
        trial.annotations["orient"] = orient.rescale(unit)


def _convert_quantity_scalar_to_string(value):
    """
    converts quantity scalar to string
    Parameters
    ----------
    value : quantity scalar
    Returns
    -------
    out : str
        magnitude and unit are separated with space.
    """
    return str(value.magnitude)+" "+value.dimensionality.string


def _convert_string_to_quantity_scalar(value):
    """
    converts string to quantity scalar
    Parameters
    ----------
    value : str
        magnitude and unit are assumed to be separated with space.
    Returns
    -------
    out : quantity scalar
    """
    v = value.split(" ")
    return pq.Quantity(float(v[0]), v[1])


def add_orientation_to_trials(trials, orients):
    """
    Adds annotation 'orient' to trials
    Parameters
    ----------
    trials : list of neo.SpikeTrains
    orients : quantity array
        orientation array
    """
    assert len(trials) == len(orients)
    for trial, orient in zip(trials, orients):
        trial.annotations["orient"] = orient


def make_stimulus_trials(chxs, stim_epoch):
    '''
    makes stimulus trials for every units (good) in each channel
    ----------
    chxs : list
        list of neo.core.ChannelIndex
    stim_epoch : neo.core.Epoch
        stimulus epoch
    Returns
    -------
    out : defaultdict(dict)
        trials[channel_index_name][unit_id] = list of spike_train trials.
    '''
    from collections import defaultdict
    stim_trials = defaultdict(dict)

    for chx in chxs:
        for un in chx.units:
            cluster_group = un.annotations.get('cluster_group') or 'noise'
            if cluster_group.lower() != "noise":
                sptr = un.spiketrains[0]
                trials = make_spiketrain_trials(epoch=stim_epoch,
                                                t_start=0 * pq.s,
                                                t_stop=stim_epoch.durations,
                                                spike_train=sptr)
                unit_id = un.annotations["cluster_id"]
                stim_trials[chx.name][unit_id] = trials

    # Add orientation value to each trial as annotation
    for chx in stim_trials.values():
        for trials in chx.values():
            add_orientation_to_trials(trials, stim_epoch.labels)

    return stim_trials


def make_orientation_trials(trials, unit=pq.deg):
    """
    Makes trials based on stimulus orientation
    Parameters
    ----------
    trials : neo.SpikeTrains
        list of spike trains where orientation is given as
        annotation 'orient' (quantity scalar) on each spike train.
    unit : Quantity, optional
        scaling unit (default is degree) used for orients
        used as keys in dictionary.
    Returns
    -------
    trials : collections.OrderedDict
        OrderedDict with orients as keys and trials as values.
    """
    from collections import defaultdict, OrderedDict
    sorted_trials = defaultdict(list)
    _rescale_orients(trials, unit)

    for trial in trials:
        orient = trial.annotations["orient"]
        key = _convert_quantity_scalar_to_string(orient)
        sorted_trials[key].append(trial)

    return OrderedDict(sorted(sorted_trials.items(),
                              key=lambda x: _convert_string_to_quantity_scalar(x[0]).magnitude))


def make_spiketrain_trials(spike_train, epoch, t_start=None, t_stop=None):
    '''
    Makes trials based on an Epoch and given temporal bound
    Parameters
    ----------
    spike_train : neo.SpikeTrain, neo.Unit, numpy.array, quantities.Quantity
    epoch : neo.Epoch
    t_start : quantities.Quantity
        time before epochs, default is 0 s
    t_stop : quantities.Quantity
        time after epochs default is duration of epoch
    Returns
    -------
    out : list of neo.SpikeTrains
    '''
    from neo.core import SpikeTrain
    if t_start is None:
        t_start = 0 * pq.s
    if t_start.ndim == 0:
        t_starts = t_start * np.ones(len(epoch.times))
    else:
        t_starts = t_start
        assert len(epoch.times) == len(t_starts), 'epoch.times and t_starts have different size'
    if t_stop is None:
        t_stop = epoch.durations
    if t_stop.ndim == 0:
        t_stops = t_stop * np.ones(len(epoch.times))
    else:
        t_stops = t_stop
        assert len(epoch.times) == len(t_stops), 'epoch.times and t_stops have different size'

    dim = 's'

    if isinstance(spike_train, neo.Unit):
        sptr = []
        for st in unit.spiketrains:
            sptr.append(spike_train.rescale(dim).magnitude)
        sptr = np.sort(sptr) * pq.s
    elif isinstance(spike_train, neo.SpikeTrain):
        sptr = spike_train.times.rescale(dim)
    elif isinstance(spike_train, pq.Quantity):
        assert is_quantities(spike_train, 'vector')
        sptr = spike_train.rescale(dim)
    elif isinstance(spike_train, np.array):
        sptr = spike_train * pq.s
    else:
        raise TypeError('Expected (neo.Unit, neo.SpikeTrain, ' +
                        'quantities.Quantity, numpy.array), got "' +
                        str(type(spike_train)) + '"')
    if not isinstance(epoch, neo.Epoch):
        raise TypeError('Expected "neo.Epoch" got "' + str(type(epoch)) + '"')

    trials = []
    for j, t in enumerate(epoch.times.rescale(dim)):
        t_start = t_starts[j].rescale(dim)
        t_stop = t_stops[j].rescale(dim)
        spikes = []
        for spike in sptr[(t+t_start < sptr) & (sptr < t+t_stop)]:
            spikes.append(spike-t)
        trials.append(SpikeTrain(times=spikes * pq.s,
                                 t_start=t_start,
                                 t_stop=t_stop))
    return trials


def make_analog_trials(ana, epoch, t_start, t_stop):
    '''
    Makes trials based on an Epoch and given temporal bound
    Parameters
    ----------
    epoch : neo.Epoch
    t_start : quantities.Quantity
        time before epochs
    t_stop : quantities.Quantity
        time after epochs
    ana : neo.AnalogSignal
    Returns
    -------
    out : list of neo.AnalogSignal
    '''
    assert t_start != t_stop, 't_start cannot be equal to t_stop'
    from neo.core import AnalogSignal
    dim = 's'
    t_start = t_start.rescale(dim)
    t_stop = t_stop.rescale(dim)
    times = epoch.times.rescale(dim)
    trials = []
    nsamp = int(abs(t_start - t_stop) * ana.sampling_rate)-1
    for j, t in enumerate(times):
        sig = ana.magnitude[(t+t_start <= ana.times) & (ana.times <= t+t_stop), :]
        trials.append(AnalogSignal(signal=sig[:nsamp, :]*ana.units,
                                   sampling_rate=ana.sampling_rate,
                                   t_start=t_start,
                                   t_stop=t_stop))
    return trials


def get_epoch(epochs, epoch_type):
    '''
    returns epoch with matching name
    ----------
    epochs : list
        list of neo.core.Epoch
    epoch_type : str
        epoch type (name)
    Returns
    -------
    out : neo.core.Epoch
    '''
    for epoch in epochs:
        if epoch_type == epoch.annotations.get("type", None):
            return epoch
    else:
        raise ValueError("epoch not found", epoch_type)


def make_stimulus_off_epoch(epo, include_boundary=False):
    '''
    Creates a neo.Epoch of off periods.
    Parameters
    ----------
    epo : neo.Epoch
        stimulus epoch
    include_boundary :
        add 0 to be first off period
    Returns
    ------
    out : neo.Epoch
    '''

    from neo.core import Epoch
    times = epo.times[:-1] + epo.durations[:-1]
    durations = epo.times[1:] - times
    if(include_boundary):
        times = np.append([0], times)*pq.s
        durations = np.append(epo.times[0], durations)*pq.s

    off_epoch = Epoch(labels=[None]*len(times),
                      durations=durations,
                      times=times)

    return off_epoch


def epoch_overview(epo, period, expected_num_epochs=None):
    '''
    Makes a new Epoch with start and stop time as first and last event in
    a burst of epochs, bursts are separated by > period + stim duration*2
    Parameters
    ----------
    epo : neo.Epoch
    Returns
    -------
    out : neo.Epoch
    '''
    is_quantities(period, dtype='scalar')
    if len(epo.times) == 1:
        return epo
    from neo import Epoch
    pause = np.diff(epo.times)
    pause = pause > period + np.median(epo.durations) * 2
    start_ind = np.concatenate((np.array([1]), pause))
    stop_ind = np.concatenate((pause, np.array([1])))
    stop_times = epo.times[stop_ind == 1]
    start_times = epo.times[start_ind == 1]
    if expected_num_epochs is not None:
        assert len(start_times) == expected_num_epochs
    return Epoch(times=start_times,
                 durations=stop_times-start_times,
                 description=epo.description)


def print_epo(epo, N=20):
    '''
    Print the N first epochs
    Parameters
    ----------
    epo : neo.Epoch
    N : number of epochs to print
    Returns
    ------
    prints : print of epoch
    '''
    cnt = 0
    for i, t, d in zip(range(epo.times.size), epo.times, epo.durations):
        d.units = 'ms'
        p = epo.times[i+1]-t-d
        p.units = 'ms'
        print('%.2f %.2f %.2f %.2f' % (t, d, t+d, p))
        cnt += 1
        if cnt == N:
            break


###############################################################################
#                           Analysis
###############################################################################
def wrap_angle(angle, wrap_range=360.):
    '''
    wraps angle in to the interval [0, wrap_range]
    ----------
    angle : numpy.array/float
        input array/float
    wrap_range : float
        wrap range (eg. 360 or 2pi)
    Returns
    -------
    out : numpy.array/float
        angle in interval [0, wrap_range]
    '''
    return angle - wrap_range * np.floor(angle/float(wrap_range))


def compute_osi(rates, orients):
    # TODO: write tests
    '''
    calculates orientation selectivity index
    Parameters
    ----------
    rates : quantity array
        array of mean firing rates
    orients : quantity array
        array of orientations
    Returns
    -------
    out : quantity scalar
        preferred orientation
    out : float
        selectivity index
    '''

    orients = orients.rescale(pq.deg)
    preferred = np.where(rates == rates.max())
    null_angle = wrap_angle(orients[preferred] + 180*pq.deg, wrap_range=360.)

    null = np.where(orients == null_angle)
    if len(null[0]) == 0:
        raise Exception("orientation not found: "+str(null_angle))

    orth_angle_p = wrap_angle(orients[preferred] + 90*pq.deg, wrap_range=360.)
    orth_angle_n = wrap_angle(orients[preferred] - 90*pq.deg, wrap_range=360.)
    orth_p = np.where(orients == orth_angle_p)
    orth_n = np.where(orients == orth_angle_n)

    if len(orth_p[0]) == 0:
        raise Exception("orientation not found: " + str(orth_angle_p))
    if len(orth_n[0]) == 0:
        raise Exception("orientation not found: " + str(orth_angle_n))

    index = 1. - (rates[orth_p] + rates[orth_n]) / (rates[preferred]+rates[null])

    return float(orients[preferred])*orients.units, float(index)


def compute_spontan_rate(chxs, stim_off_epoch):
    # TODO: test
    '''
    Calculates spontaneous firing rate
    Parameters
    ----------
    chxs : list
        list of neo.core.ChannelIndex
    stim_off_epoch : neo.core.Epoch
        stimulus epoch
    Returns
    -------
    out : defaultdict(dict)
        rates[channel_index_name][unit_id] = spontaneous rate
    '''
    from collections import defaultdict
    from elephant.statistics import mean_firing_rate

    rates = defaultdict(dict)
    unit_rates = pq.Hz

    for chx in chxs:
        for un in chx.units:
            cluster_group = un.annotations.get('cluster_group') or 'noise'
            if cluster_group.lower() != "noise":
                sptr = un.spiketrains[0]
                unit_id = un.annotations["cluster_id"]
                trials = make_spiketrain_trials(epoch=stim_off_epoch,
                                                t_start=0 * pq.s,
                                                t_stop=stim_off_epoch.durations,
                                                spike_train=sptr)
                rate = 0 * unit_rates
                for trial in trials:
                    rate += mean_firing_rate(trial, trial.t_start, trial.t_stop)

                rates[chx.name][unit_id] = rate / len(trials)

    return rates


def compute_orientation_tuning(orient_trials):
    from exana.stimulus.tools import (make_orientation_trials,
                                      _convert_string_to_quantity_scalar)
    '''
    Calculates the mean firing rate for each orientation
    Parameters
    ----------
    trials : collections.OrderedDict
        OrderedDict with orients as keys and trials as values.
    Returns
    -------
    rates : quantity array
        average rates
    orients : quantity array
        sorted stimulus orientations
    '''
    from elephant.statistics import mean_firing_rate

    unit_orients = pq.deg
    unit_rates = pq.Hz
    orient_count = len(orient_trials)

    rates = np.zeros((orient_count)) * unit_rates
    orients = np.zeros((orient_count)) * unit_orients

    for i, (orient, trials) in enumerate(orient_trials.items()):
        orient = _convert_string_to_quantity_scalar(orient)
        rate = 0 * unit_rates

        for trial in trials:
            rate += mean_firing_rate(trial, trial.t_start, trial.t_stop)

        rates[i] = rate / len(trials)
        orients[i] = orient.rescale(unit_orients)

    return rates, orients


def rate_latency(trials=None, epo=None, unit=None, t_start=None, t_stop=None,
                 kernel=None, search_stop=None, sampling_period=None):
    assert trials != unit
    import neo
    import elephant
    if trials is None:
        trials = make_spiketrain_trials(epo=epo, unit=unit, t_start=t_start,
                                        t_stop=t_stop)
    else:
        t_start = trials[0].t_start
        t_stop = trials[0].t_stop
        if search_stop is None:
            search_stop = t_stop
    trial = neo.SpikeTrain(times=np.array([st for trial in trials
                                           for st in trial.times.rescale('s')])*pq.s,
                           t_start=t_start, t_stop=t_stop)
    rate = elephant.statistics.instantaneous_rate(trial, sampling_period,
                                                  kernel=kernel, trim=True)/len(trials)
    rate_mag = rate.rescale('Hz').magnitude.reshape(len(rate))
    if not any(rate_mag):
        return np.nan, rate
    else:
        mask = (rate.times > 0*pq.ms) & (rate.times < 250*pq.ms)
        spont_mask = (rate.times > -250*pq.ms) & (rate.times < 0*pq.ms)
        # spk, ind = find_max_peak(rate_mag[mask])
        krit1 = rate_mag[mask].mean() + rate_mag[mask].std() > rate_mag[spont_mask].mean() + rate_mag[spont_mask].std()
        spike_mask = (trial.times > 0*pq.ms) & (trial.times < search_stop)
        krit2 = len(trial.times[spike_mask])/search_stop.rescale('s') > 1.*pq.Hz
        if not krit1 and krit2:
            return np.nan, rate
        t0 = 0*pq.ms
        while t0 < search_stop:
            mask = (rate.times > t0) & (rate.times < search_stop)
            pk, ind = find_first_peak(rate_mag[mask])
            if len(pk) == 0:
                break
            krit3 = pk > rate_mag[mask].mean() + rate_mag[mask].std()
            krit4 = pk > 1.*pq.Hz
            krit5 = pk != 0
            lat_time = rate.times[mask][ind]
            assert len(lat_time) == 1
            if krit3 and krit4 and krit5:
                return lat_time, rate
            else:
                t0 = lat_time
        return np.nan, rate
