import numpy as np
import quantities as pq
import pytest


def test_make_stimulus_off_epoch():
    from neo.core import Epoch
    from exana.stimulus.tools import (make_stimulus_off_epoch)
    
    times = np.linspace(0, 10, 11) * pq.s
    durations = np.ones(len(times)) * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.linspace(1, 10, 10)).all()
    assert(stim_off_epoch.durations == np.zeros(10)).all()
    assert(stim_off_epoch.labels == [None]*10)

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)
    assert(stim_off_epoch.times == np.linspace(0, 10, 11)).all()
    assert(stim_off_epoch.durations == np.zeros(11)).all()
    assert(stim_off_epoch.labels == [None]*11)

    times = np.arange(0.5, 11, 0.5)[::2] * pq.s
    durations = np.ones(len(times)) * 0.5 * pq.s
    labels = np.ones(len(times))

    stim_epoch = Epoch(labels=labels, durations=durations, times=times)
    stim_off_epoch = make_stimulus_off_epoch(stim_epoch)

    assert(stim_off_epoch.times == np.arange(1, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(10) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*10)

    stim_off_epoch = make_stimulus_off_epoch(stim_epoch, include_boundary=True)

    assert(stim_off_epoch.times == np.arange(0, 11, 1)).all()
    assert(stim_off_epoch.durations == np.ones(11) * 0.5).all()
    assert(stim_off_epoch.labels == [None]*11)


def test_compute_orientation_tuning():
    from neo.core import SpikeTrain
    import quantities as pq
    from exana.stimulus.tools import (make_orientation_trials,
                                      compute_orientation_tuning)

    trials = [SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315. * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 1)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]
    sorted_orients = np.array([0, (np.pi/3 * pq.rad).rescale(pq.deg)/pq.deg, 315]) * pq.deg
    rates_e = np.array([1., 2.7, 0.5]) / pq.s

    trials = make_orientation_trials(trials)
    rates, orients = compute_orientation_tuning(trials)
    assert((rates == rates_e).all())
    assert(rates.units == rates_e.units)
    assert((orients == sorted_orients).all())
    assert(orients.units == sorted_orients.units)


def test_make_orientation_trials():
    from neo.core import SpikeTrain
    from exana.stimulus.tools import (make_orientation_trials,
                                      _convert_string_to_quantity_scalar)

    trials = [SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315. * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 1)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]

    sorted_trials = [[trials[2]], [trials[1], trials[3]], [trials[0]]]
    sorted_orients = [0 * pq.deg, (np.pi/3 * pq.rad).rescale(pq.deg), 315 * pq.deg]
    orient_trials = make_orientation_trials(trials, unit=pq.deg)

    for (key, value), trial, orient in zip(orient_trials.items(),
                                           sorted_trials,
                                           sorted_orients):
        key = _convert_string_to_quantity_scalar(key)
        assert(key == orient.magnitude)
        for t, st in zip(value, trial):
            assert((t == st).all())
            assert(t.t_start == st.t_start)
            assert(t.t_stop == st.t_stop)
            assert(t.annotations["orient"] == orient)


def test_rescale_orients():
    from neo.core import SpikeTrain
    import quantities as pq
    from exana.stimulus.tools import _rescale_orients

    trials = [SpikeTrain(np.arange(0, 10, 1.)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315 * pq.deg)]
    scaled_trials = trials.copy()
    _rescale_orients(scaled_trials[:])
    assert(scaled_trials is not trials)
    for t, st in zip(trials, scaled_trials):
        orient = list(t.annotations.values())[0].rescale(pq.deg)
        scaled_orient = list(st.annotations.values())[0]
        assert(scaled_orient.units == pq.deg)
        assert(scaled_orient == orient)
        assert((t == st).all())
        assert(t.t_start == st.t_start)
        assert(t.t_stop == st.t_stop)

    trials = [SpikeTrain(np.arange(0, 10, 1.)*pq.s, t_stop=10*pq.s,
                         orient=0 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.5)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad),
              SpikeTrain(np.arange(0, 10, 2)*pq.s, t_stop=10*pq.s,
                         orient=315 * pq.deg),
              SpikeTrain(np.arange(0, 10, 0.3)*pq.s, t_stop=10*pq.s,
                         orient=np.pi/3 * pq.rad)]

    scaled_trials = trials.copy()
    _rescale_orients(scaled_trials[:], unit=pq.rad)
    assert(scaled_trials is not trials)
    for t, st in zip(trials, scaled_trials):
        orient = list(t.annotations.values())[0].rescale(pq.rad)
        scaled_orient = list(st.annotations.values())[0]
        assert(scaled_orient.units == pq.rad)
        assert(scaled_orient == orient)
        assert((t == st).all())
        assert(t.t_start == st.t_start)
        assert(t.t_stop == st.t_stop)
